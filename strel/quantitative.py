"""
quantitative.py — STREL quantitative (real-valued robustness) semantics.

Signal tensor convention:
    signal : [B, N, F, T]   batch × agents × features × time steps
    lab    : [B, N, T]      integer class label per agent per time step

Every formula's evaluate() returns robustness [B, N, T]:
    positive  →  property satisfied, margin = robustness value
    negative  →  property violated, |margin| = robustness value

Semantics mode (controlled per evaluate() call):
    differentiable=False  (default)
        Uses torch.amin / torch.amax for temporal aggregation and
        torch.minimum / torch.maximum for pointwise operations.
        Results are EXACT.  No gradient flows through min/max.

    differentiable=True
        Uses smooth_min / smooth_max (conservative lower bounds of the
        exact operators).  Results satisfy:
            smooth_rho <= exact_rho   always
        so the smooth robustness is a provable under-approximation of the
        true robustness.  Gradients flow through all operations.
        Required for AutoLiRPA certification (CROWN / CROWN-IBP).
"""

from typing import Optional

import torch

from .smooth_ops import smooth_max, smooth_min, exact_max, exact_min


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _agg_max(x: torch.Tensor, beta: float, differentiable: bool) -> torch.Tensor:
    """Aggregate (max) over the last dimension, mode-aware."""
    return smooth_max(x, beta) if differentiable else exact_max(x)


def _agg_min(x: torch.Tensor, beta: float, differentiable: bool) -> torch.Tensor:
    """Aggregate (min) over the last dimension, mode-aware."""
    return smooth_min(x, beta) if differentiable else exact_min(x)


def _pw_max(a: torch.Tensor, b: torch.Tensor, differentiable: bool) -> torch.Tensor:
    """Pointwise max(a, b), mode-aware."""
    if differentiable:
        return smooth_max(torch.stack([a, b], dim=-1))
    return torch.maximum(a, b)


def _pw_min(a: torch.Tensor, b: torch.Tensor, differentiable: bool) -> torch.Tensor:
    """Pointwise min(a, b), mode-aware."""
    if differentiable:
        return smooth_min(torch.stack([a, b], dim=-1))
    return torch.minimum(a, b)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class STRELFormula:
    """Base class for all STREL formulas."""

    def evaluate(
        self,
        signal: torch.Tensor,
        lab: torch.Tensor,
        differentiable: bool = False,
    ) -> torch.Tensor:
        """
        Evaluate quantitative robustness.

        Args:
            signal:         [B, N, F, T]
            lab:            [B, N, T] integer labels
            differentiable: False (default) → exact semantics with
                            torch.amin/amax/minimum/maximum.
                            True → conservative smooth lower bounds,
                            required for backprop and AutoLiRPA.

        Returns:
            robustness [B, N, T]
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Atomic propositions
# ---------------------------------------------------------------------------

class AtomicPredicate(STRELFormula):
    """
    Linear threshold predicate on one signal feature, optionally restricted
    to a subset of agent classes.

    Evaluates to:
        sign * signal[..., var_ind, :] - |threshold|

    where sign = -1 if lte (≤ threshold), +1 if gte (≥ threshold).
    The predicate is exact regardless of the differentiable flag (it is a
    linear function with no min/max).

    Args:
        var_ind:   Index of the signal feature.
        threshold: Threshold value.
        labels:    1-D tensor of class ids this applies to. None → all.
        lte:       True → signal ≤ threshold; False → signal ≥ threshold.
    """

    NEG_INF: float = -1e2

    def __init__(
        self,
        var_ind: int,
        threshold: float,
        labels: Optional[torch.Tensor] = None,
        lte: bool = False,
    ):
        self.var_ind = var_ind
        self.sign = -1.0 if lte else 1.0
        self.bias = threshold if lte else -threshold
        self.labels = torch.ones(1) if labels is None else labels.reshape(-1)

    def _predicate(self, signal: torch.Tensor) -> torch.Tensor:
        return self.sign * signal[:, :, self.var_ind, :] + self.bias

    def _mask(self, x: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        labels = self.labels.to(x.device).view(1, 1, 1, -1)
        n_cls = self.labels.shape[0]
        class_indices = torch.arange(n_cls, device=x.device).view(1, 1, 1, -1)
        one_hot = (lab.unsqueeze(-1) == class_indices).to(x.dtype)
        mask = torch.sum(one_hot * labels, dim=-1)
        return torch.clamp(mask, min=0.0, max=1.0)

    def evaluate(self, signal, lab, differentiable=False):
        m = self._mask(signal, lab)
        z = self._predicate(signal)
        return z * m + (1.0 - m) * self.NEG_INF


class TrueFormula(STRELFormula):
    def __init__(self, pos_inf: float = 1e2):
        self.pos_inf = pos_inf

    def evaluate(self, signal, lab, differentiable=False):
        B, N, _, T = signal.shape
        return torch.full((B, N, T), self.pos_inf, dtype=signal.dtype, device=signal.device)


class FalseFormula(STRELFormula):
    def __init__(self, neg_inf: float = -1e2):
        self.neg_inf = neg_inf

    def evaluate(self, signal, lab, differentiable=False):
        B, N, _, T = signal.shape
        return torch.full((B, N, T), self.neg_inf, dtype=signal.dtype, device=signal.device)


# ---------------------------------------------------------------------------
# Boolean connectives
# ---------------------------------------------------------------------------

class Not(STRELFormula):
    """ρ(¬φ) = −ρ(φ)  — exact regardless of mode."""

    def __init__(self, phi: STRELFormula):
        self.phi = phi

    def evaluate(self, signal, lab, differentiable=False):
        return -self.phi.evaluate(signal, lab, differentiable)


class And(STRELFormula):
    """
    ρ(φ₁ ∧ φ₂)
        exact:         min(ρ₁, ρ₂)
        differentiable: smooth_min(ρ₁, ρ₂)  [≤ exact]
    """

    def __init__(self, left: STRELFormula, right: STRELFormula, beta: float = 20.0):
        self.left = left
        self.right = right
        self.beta = beta

    def evaluate(self, signal, lab, differentiable=False):
        l = self.left.evaluate(signal, lab, differentiable)
        r = self.right.evaluate(signal, lab, differentiable)
        return _pw_min(l, r, differentiable)


class Or(STRELFormula):
    """
    ρ(φ₁ ∨ φ₂)
        exact:         max(ρ₁, ρ₂)
        differentiable: smooth_max(ρ₁, ρ₂)  [≤ exact]
    """

    def __init__(self, left: STRELFormula, right: STRELFormula, beta: float = 20.0):
        self.left = left
        self.right = right
        self.beta = beta

    def evaluate(self, signal, lab, differentiable=False):
        l = self.left.evaluate(signal, lab, differentiable)
        r = self.right.evaluate(signal, lab, differentiable)
        return _pw_max(l, r, differentiable)


class Implies(STRELFormula):
    """φ₁ → φ₂  ≡  ¬φ₁ ∨ φ₂."""

    def __init__(self, ante: STRELFormula, cons: STRELFormula, beta: float = 20.0):
        self._impl = Or(Not(ante), cons, beta=beta)

    def evaluate(self, signal, lab, differentiable=False):
        return self._impl.evaluate(signal, lab, differentiable)


# ---------------------------------------------------------------------------
# Temporal operators  (aggregate over T)
# ---------------------------------------------------------------------------

class Always(STRELFormula):
    """
    G φ  — globally / always.
        exact:         amin over T (broadcast back to [B, N, T])
        differentiable: smooth_min over T  [≤ exact]
    """

    def __init__(self, phi: STRELFormula, beta: float = 20.0):
        self.phi = phi
        self.beta = beta

    def evaluate(self, signal, lab, differentiable=False):
        phi_rob = self.phi.evaluate(signal, lab, differentiable)   # [B, N, T]
        agg = _agg_min(phi_rob, self.beta, differentiable)         # [B, N]
        return agg.unsqueeze(-1).expand_as(phi_rob)


class Eventually(STRELFormula):
    """
    F φ  — eventually.
        exact:         amax over T
        differentiable: smooth_max over T  [≤ exact]
    """

    def __init__(self, phi: STRELFormula, beta: float = 20.0):
        self.phi = phi
        self.beta = beta

    def evaluate(self, signal, lab, differentiable=False):
        phi_rob = self.phi.evaluate(signal, lab, differentiable)
        agg = _agg_max(phi_rob, self.beta, differentiable)
        return agg.unsqueeze(-1).expand_as(phi_rob)


class Until(STRELFormula):
    """
    φ₁ U φ₂  — until.
    Backward scan:
        result[T-1] = ρ₂[T-1]
        result[t]   = max(ρ₂[t], min(ρ₁[t], result[t+1]))

        exact:         torch.maximum / torch.minimum
        differentiable: smooth_max / smooth_min
    """

    def __init__(self, phi1: STRELFormula, phi2: STRELFormula, beta: float = 20.0):
        self.phi1 = phi1
        self.phi2 = phi2
        self.beta = beta

    def evaluate(self, signal, lab, differentiable=False):
        r1 = self.phi1.evaluate(signal, lab, differentiable)   # [B, N, T]
        r2 = self.phi2.evaluate(signal, lab, differentiable)
        B, N, T = r1.shape
        result = torch.zeros_like(r1)
        result[:, :, -1] = r2[:, :, -1]
        for t in range(T - 2, -1, -1):
            carry = _pw_min(r1[:, :, t], result[:, :, t + 1], differentiable)
            result[:, :, t] = _pw_max(r2[:, :, t], carry, differentiable)
        return result


# ---------------------------------------------------------------------------
# Spatial operators
# ---------------------------------------------------------------------------

class _SpatialBase(STRELFormula):
    """Shared distance-matrix and mask utilities for Reach / Escape."""

    NEG_INF: float = -1e2
    POS_INF: float = 1e2
    EPS: float = 1e-8

    def __init__(self, pos_x_ind: int = 2, pos_y_ind: int = 3):
        self.pos_x_ind = pos_x_ind
        self.pos_y_ind = pos_y_ind

    def _dist_mat(self, signal: torch.Tensor) -> torch.Tensor:
        """Euclidean distance matrix [B, T, N, N]."""
        pos = signal[:, :, [self.pos_x_ind, self.pos_y_ind], :].permute(0, 3, 1, 2)
        diff = pos.unsqueeze(2) - pos.unsqueeze(3)
        return torch.sqrt(torch.sum(diff * diff, dim=-1) + self.EPS)

    @staticmethod
    def _label_mask(x, lab, labels):
        if labels is None:
            return torch.ones_like(lab, dtype=x.dtype)
        labels = labels.to(x.device).view(1, 1, 1, -1)
        n_cls = labels.shape[-1]
        class_indices = torch.arange(n_cls, device=x.device).view(1, 1, 1, -1)
        one_hot = (lab.unsqueeze(-1) == class_indices).to(x.dtype)
        return torch.clamp(torch.sum(one_hot * labels, dim=-1), 0.0, 1.0)

    @staticmethod
    def _fw_min(cost: torch.Tensor, n: int) -> torch.Tensor:
        """
        All-pairs shortest path (min-cost).
        Uses abs-trick: min(a,b) = 0.5*(a+b-|a-b|) — exact in both modes.
        auto_LiRPA supports torch.abs, so this stays on the smooth path too.
        """
        d = cost.clone()
        for k in range(n):
            dik = d[:, :, :, k].unsqueeze(-1)
            dkj = d[:, :, k, :].unsqueeze(-2)
            cand = dik + dkj
            d = 0.5 * (d + cand - torch.abs(d - cand))
        return d

    @staticmethod
    def _fw_max_bottleneck(cost: torch.Tensor, n: int) -> torch.Tensor:
        """
        All-pairs bottleneck-max path (max of min-weights along path).
        Uses abs-trick: max(a,b) = 0.5*(a+b+|a-b|) — exact in both modes.
        """
        w = cost.clone()
        for k in range(n):
            wik = w[:, :, :, k].unsqueeze(-1)
            wkj = w[:, :, k, :].unsqueeze(-2)
            cand = 0.5 * (wik + wkj - torch.abs(wik - wkj))  # min(wik, wkj)
            w = 0.5 * (w + cand + torch.abs(w - cand))        # max(w, cand)
        return w


class Reach(_SpatialBase):
    """
    Reach(φ₁, φ₂, d₁, d₂):
    From source i, there exists destination j reachable via φ₁-satisfying
    intermediate nodes, with shortest-path distance ∈ [d₁, d₂], where j
    satisfies φ₂.

    The Floyd-Warshall inner loops use the abs-trick (exact min/max) in
    both modes.  The differentiable flag affects only the aggregation of
    sub-formula robustness values and the final max-over-j.

    Returns robustness [B, N, T].
    """

    def __init__(
        self,
        phi1: STRELFormula,
        phi2: STRELFormula,
        d1: float,
        d2: float,
        labels1: Optional[torch.Tensor] = None,
        labels2: Optional[torch.Tensor] = None,
        pos_x_ind: int = 2,
        pos_y_ind: int = 3,
    ):
        super().__init__(pos_x_ind, pos_y_ind)
        self.phi1 = phi1
        self.phi2 = phi2
        self.d1 = d1
        self.d2 = d2
        self.labels1 = None if labels1 is None else labels1.reshape(-1)
        self.labels2 = None if labels2 is None else labels2.reshape(-1)

    def evaluate(self, signal, lab, differentiable=False):
        B, N, _, T = signal.shape
        dist = self._dist_mat(signal)              # [B, T, N, N]
        adj = dist / (torch.abs(dist) + self.EPS)
        idx = torch.arange(N, device=signal.device)
        eye = (idx.unsqueeze(0) == idx.unsqueeze(1)).unsqueeze(0).unsqueeze(0).to(signal.dtype)

        m1 = self._label_mask(signal, lab, self.labels1)
        m2 = self._label_mask(signal, lab, self.labels2)
        s1 = self.phi1.evaluate(signal, lab, differentiable) * m1 + (1.0 - m1) * self.NEG_INF
        s2 = self.phi2.evaluate(signal, lab, differentiable) * m2 + (1.0 - m2) * self.NEG_INF

        s1_bt = s1.permute(0, 2, 1)    # [B, T, N]
        s2_bt = s2.permute(0, 2, 1)

        # Bottleneck-weight matrix (intermediate node satisfaction)
        c = s1_bt.unsqueeze(-1).expand(B, T, N, N)
        c = c * adj + self.NEG_INF * (1.0 - adj)
        c = self.POS_INF * eye + c * (1.0 - eye)
        wcap = self._fw_max_bottleneck(c, N)       # exact abs-trick

        # All-pairs shortest distances
        d = dist * adj + self.POS_INF * (1.0 - adj)
        d = d * (1.0 - eye)
        d = self._fw_min(d, N)                     # exact abs-trick

        # Distance range eligibility
        low = d - self.d1 + 1e-6
        high = self.d2 - d + 1e-6
        in_range = 0.5 * (low + high - torch.abs(low - high))
        elig = 0.5 * (1.0 + in_range / (torch.abs(in_range) + self.EPS))

        s2_dest = s2_bt.unsqueeze(-2).expand(B, T, N, N)
        pair = 0.5 * (wcap + s2_dest - torch.abs(wcap - s2_dest))   # min(wcap, s2)
        pair = pair * elig + self.NEG_INF * (1.0 - elig)
        pair = pair * (1.0 - eye) + self.NEG_INF * eye

        # Max over destinations j  (mode-aware)
        best = pair[:, :, :, 0].clone()
        for j in range(1, N):
            best = _pw_max(best, pair[:, :, :, j], differentiable)

        return best.permute(0, 2, 1)    # [B, N, T]


class Escape(_SpatialBase):
    """
    Escape(φ, d₁, d₂):
    There exists a path from i to destination j staying in φ-satisfying
    nodes, with distance ∈ [d₁, d₂].

    Returns robustness [B, N, T].
    """

    def __init__(
        self,
        phi: STRELFormula,
        d1: float,
        d2: float,
        labels: Optional[torch.Tensor] = None,
        pos_x_ind: int = 2,
        pos_y_ind: int = 3,
    ):
        super().__init__(pos_x_ind, pos_y_ind)
        self.phi = phi
        self.d1 = d1
        self.d2 = d2
        self.labels = None if labels is None else labels.reshape(-1)

    def evaluate(self, signal, lab, differentiable=False):
        B, N, _, T = signal.shape
        dist = self._dist_mat(signal)
        adj = dist / (torch.abs(dist) + self.EPS)
        idx = torch.arange(N, device=signal.device)
        eye = (idx.unsqueeze(0) == idx.unsqueeze(1)).unsqueeze(0).unsqueeze(0).to(signal.dtype)

        m = self._label_mask(signal, lab, self.labels)
        s = self.phi.evaluate(signal, lab, differentiable) * m + (1.0 - m) * self.NEG_INF
        s_bt = s.permute(0, 2, 1)

        c = s_bt.unsqueeze(-1).expand(B, T, N, N)
        c = c * adj + self.NEG_INF * (1.0 - adj)
        c = self.POS_INF * eye + c * (1.0 - eye)
        wcap = self._fw_max_bottleneck(c, N)

        d = dist * adj + self.POS_INF * (1.0 - adj)
        d = d * (1.0 - eye)
        d = self._fw_min(d, N)

        low  = d - self.d1 + 1e-6
        high = self.d2 - d + 1e-6
        in_range = 0.5 * (low + high - torch.abs(low - high))
        elig = 0.5 * (1.0 + in_range / (torch.abs(in_range) + self.EPS))

        dest_rob = s_bt.unsqueeze(-2).expand(B, T, N, N)
        path_dest = 0.5 * (wcap + dest_rob - torch.abs(wcap - dest_rob))
        pair = path_dest * elig + self.NEG_INF * (1.0 - elig)
        pair = pair * (1.0 - eye) + self.NEG_INF * eye

        best = pair[:, :, :, 0].clone()
        for j in range(1, N):
            best = _pw_max(best, pair[:, :, :, j], differentiable)

        return best.permute(0, 2, 1)
