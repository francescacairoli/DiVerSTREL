"""
parametric.py — Parametric quantitative STREL semantics.

Formula classes here inherit from both nn.Module and the quantitative
STRELFormula hierarchy.  Threshold values (and optionally distance bounds)
are exposed as nn.Parameter, so gradients flow through the robustness
computation back into the specification parameters.

This enables:
    - learning thresholds from data (specification mining)
    - joint optimisation of network weights and specification parameters
    - AutoLiRPA certification over a parametric formula

Parametric classes always use smooth (differentiable) operators internally,
since they are designed for gradient-based training.  However, they accept
the differentiable kwarg and pass it down, so mixing parametric and
non-parametric nodes in the same tree is supported.

NOTE: When using AutoLiRPA, only wrap the *outermost* parametric formula in
the BoundedModule (see examples/03_certified_training.py).  Inner formulas
that are plain STRELFormulas (non-parametric) are opaque to LiRPA — embed
their robustness as a fixed constant input if needed.
"""

from typing import Optional

import torch
import torch.nn as nn

from .smooth_ops import smooth_max, smooth_min
from .quantitative import STRELFormula, _SpatialBase, _pw_max


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _soft_clamp(x: torch.Tensor) -> torch.Tensor:
    """Keep a learned parameter loosely positive (used for distance bounds)."""
    return torch.clamp(x, min=1e-3)


# ---------------------------------------------------------------------------
# Parametric atomic predicate
# ---------------------------------------------------------------------------

class ParametricAtomicPredicate(nn.Module, STRELFormula):
    """
    Atomic predicate with a learnable threshold.

    The threshold is an nn.Parameter, so `optimizer.step()` can update it.
    All other bookkeeping is identical to the quantitative AtomicPredicate.

    The predicate is linear in the threshold (no min/max), so the
    differentiable flag has no effect here — gradients always flow.

    Args:
        var_ind:           Signal feature index.
        init_threshold:    Initial value of the learnable threshold.
        labels:            Class label mask (fixed, not learned).
        lte:               Direction of inequality.
    """

    NEG_INF: float = -1e2

    def __init__(
        self,
        var_ind: int,
        init_threshold: float = 0.0,
        labels: Optional[torch.Tensor] = None,
        lte: bool = False,
    ):
        nn.Module.__init__(self)
        self.var_ind = var_ind
        self.sign = -1.0 if lte else 1.0
        self.lte = lte
        self.threshold = nn.Parameter(torch.tensor(init_threshold, dtype=torch.float32))
        self.labels = torch.ones(1) if labels is None else labels.reshape(-1)

    @property
    def bias(self) -> torch.Tensor:
        return self.threshold if self.lte else -self.threshold

    def _predicate(self, signal: torch.Tensor) -> torch.Tensor:
        return self.sign * signal[:, :, self.var_ind, :] + self.bias

    def _mask(self, x: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        labels = self.labels.to(x.device).view(1, 1, 1, -1)
        n_cls = self.labels.shape[0]
        class_indices = torch.arange(n_cls, device=x.device).view(1, 1, 1, -1)
        one_hot = (lab.unsqueeze(-1) == class_indices).to(x.dtype)
        mask = torch.sum(one_hot * labels, dim=-1)
        return torch.clamp(mask, min=0.0, max=1.0)

    def evaluate(
        self,
        signal: torch.Tensor,
        lab: torch.Tensor,
        differentiable: bool = True,   # default True: parametric classes are for training
    ) -> torch.Tensor:
        # Linear in threshold — differentiable flag has no effect, gradients always flow
        m = self._mask(signal, lab)
        z = self._predicate(signal)
        return z * m + (1.0 - m) * self.NEG_INF

    def forward(self, signal: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        """nn.Module forward — same as evaluate(differentiable=True)."""
        return self.evaluate(signal, lab, differentiable=True)


# ---------------------------------------------------------------------------
# Parametric temporal operators
# ---------------------------------------------------------------------------

class ParametricAlways(nn.Module, STRELFormula):
    """
    Always with learnable beta (inverse temperature).

    A larger beta means the smooth_min is tighter (closer to the true min),
    so it can also be annealed during training via gradient descent.

    Always uses smooth_min internally (gradients must flow through beta).
    """

    def __init__(self, phi: STRELFormula, init_beta: float = 20.0):
        nn.Module.__init__(self)
        self.phi = phi
        self.log_beta = nn.Parameter(torch.tensor(float(init_beta)).log())

    @property
    def beta(self) -> torch.Tensor:
        return self.log_beta.exp()

    def evaluate(
        self,
        signal: torch.Tensor,
        lab: torch.Tensor,
        differentiable: bool = True,
    ) -> torch.Tensor:
        phi_rob = self.phi.evaluate(signal, lab, differentiable)
        # Always smooth: beta is a parameter, gradients must flow through it
        agg = smooth_min(phi_rob, beta=self.beta.item())
        return agg.unsqueeze(-1).expand_as(phi_rob)

    def forward(self, signal: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        return self.evaluate(signal, lab, differentiable=True)


class ParametricEventually(nn.Module, STRELFormula):
    """Eventually with learnable beta."""

    def __init__(self, phi: STRELFormula, init_beta: float = 20.0):
        nn.Module.__init__(self)
        self.phi = phi
        self.log_beta = nn.Parameter(torch.tensor(float(init_beta)).log())

    @property
    def beta(self) -> torch.Tensor:
        return self.log_beta.exp()

    def evaluate(
        self,
        signal: torch.Tensor,
        lab: torch.Tensor,
        differentiable: bool = True,
    ) -> torch.Tensor:
        phi_rob = self.phi.evaluate(signal, lab, differentiable)
        # Always smooth: beta is a parameter, gradients must flow through it
        agg = smooth_max(phi_rob, beta=self.beta.item())
        return agg.unsqueeze(-1).expand_as(phi_rob)

    def forward(self, signal: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        return self.evaluate(signal, lab, differentiable=True)


# ---------------------------------------------------------------------------
# Parametric spatial operators
# ---------------------------------------------------------------------------

class ParametricReach(nn.Module, _SpatialBase):
    """
    Reach with learnable distance bounds d1, d2.

    Both bounds are stored as unconstrained parameters and soft-clamped to
    be ≥ 1e-3 at evaluation time, so gradient flow is never blocked by a
    hard clamp during training.

    The Floyd-Warshall abs-trick (exact min/max) is used unconditionally for
    path distance; only the final max-over-j aggregation obeys the
    differentiable flag.

    Args:
        phi1, phi2:   STREL formulas for intermediate / destination nodes.
        init_d1:      Initial lower distance bound.
        init_d2:      Initial upper distance bound.
        labels1/2:    Fixed class label masks.
        pos_x_ind/y:  Feature indices for (x, y) position.
    """

    NEG_INF: float = -1e2
    POS_INF: float = 1e2
    EPS: float = 1e-8

    def __init__(
        self,
        phi1: STRELFormula,
        phi2: STRELFormula,
        init_d1: float = 0.0,
        init_d2: float = 1.0,
        labels1: Optional[torch.Tensor] = None,
        labels2: Optional[torch.Tensor] = None,
        pos_x_ind: int = 2,
        pos_y_ind: int = 3,
    ):
        nn.Module.__init__(self)
        _SpatialBase.__init__(self, pos_x_ind, pos_y_ind)
        self.phi1 = phi1
        self.phi2 = phi2
        self.d1_raw = nn.Parameter(torch.tensor(float(init_d1)))
        self.d2_raw = nn.Parameter(torch.tensor(float(init_d2)))
        self.labels1 = None if labels1 is None else labels1.reshape(-1)
        self.labels2 = None if labels2 is None else labels2.reshape(-1)

    @property
    def d1(self) -> torch.Tensor:
        return _soft_clamp(self.d1_raw)

    @property
    def d2(self) -> torch.Tensor:
        return _soft_clamp(self.d2_raw)

    def evaluate(
        self,
        signal: torch.Tensor,
        lab: torch.Tensor,
        differentiable: bool = True,
    ) -> torch.Tensor:
        B, N, _, T = signal.shape
        dist = self._dist_mat(signal)
        adj = dist / (torch.abs(dist) + self.EPS)
        idx = torch.arange(N, device=signal.device)
        eye = (idx.unsqueeze(0) == idx.unsqueeze(1)).unsqueeze(0).unsqueeze(0).to(signal.dtype)

        m1 = self._label_mask(signal, lab, self.labels1)
        m2 = self._label_mask(signal, lab, self.labels2)
        s1 = self.phi1.evaluate(signal, lab, differentiable) * m1 + (1.0 - m1) * self.NEG_INF
        s2 = self.phi2.evaluate(signal, lab, differentiable) * m2 + (1.0 - m2) * self.NEG_INF

        s1_bt = s1.permute(0, 2, 1)
        s2_bt = s2.permute(0, 2, 1)

        c = s1_bt.unsqueeze(-1).expand(B, T, N, N)
        c = c * adj + self.NEG_INF * (1.0 - adj)
        c = self.POS_INF * eye + c * (1.0 - eye)
        wcap = self._fw_max_bottleneck(c, N)      # abs-trick, exact

        d = dist * adj + self.POS_INF * (1.0 - adj)
        d = d * (1.0 - eye)
        d = self._fw_min(d, N)                    # abs-trick, exact

        # Use parametric d1, d2 — gradient flows through them
        low = d - self.d1 + 1e-6
        high = self.d2 - d + 1e-6
        in_range = 0.5 * (low + high - torch.abs(low - high))
        elig = 0.5 * (1.0 + in_range / (torch.abs(in_range) + self.EPS))

        s2_dest = s2_bt.unsqueeze(-2).expand(B, T, N, N)
        pair = 0.5 * (wcap + s2_dest - torch.abs(wcap - s2_dest))   # min
        pair = pair * elig + self.NEG_INF * (1.0 - elig)
        pair = pair * (1.0 - eye) + self.NEG_INF * eye

        # Max over destinations j (mode-aware)
        best = pair[:, :, :, 0].clone()
        for j in range(1, N):
            best = _pw_max(best, pair[:, :, :, j], differentiable)

        return best.permute(0, 2, 1)

    def forward(self, signal: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        return self.evaluate(signal, lab, differentiable=True)


# ---------------------------------------------------------------------------
# Composite parametric formula container
# ---------------------------------------------------------------------------

class ParametricFormula(nn.Module):
    """
    Thin nn.Module wrapper for any STRELFormula tree that may contain
    ParametricAtomicPredicate / ParametricReach / etc. nodes.

    Collects all nn.Parameters from child modules automatically.
    Use this as the top-level module when certifying with AutoLiRPA.

    Args:
        formula: Root of the STREL formula tree (may mix parametric and
                 non-parametric nodes).

    Usage::

        phi = ParametricFormula(
            And(
                ParametricAtomicPredicate(0, init_threshold=0.3),
                Eventually(AtomicPredicate(1, 0.0), beta=10.0),
            )
        )
        rho = phi(signal, lab)           # smooth semantics, [B, N, T]
        rho_exact = phi(signal, lab, differentiable=False)   # exact

        # For AutoLiRPA:
        from auto_LiRPA import BoundedModule
        lirpa_phi = BoundedModule(phi, (dummy_signal, dummy_lab))
    """

    def __init__(self, formula: STRELFormula):
        super().__init__()
        self.formula = formula
        # Register any nn.Module children so their parameters appear in
        # self.parameters() / self.state_dict()
        self._register_submodules(formula)

    def _register_submodules(self, node, _seen=None):
        if _seen is None:
            _seen = set()
        if id(node) in _seen:
            return
        _seen.add(id(node))
        if isinstance(node, nn.Module):
            name = f"_sub_{len(list(self.children()))}"
            self.add_module(name, node)
        for attr in ("phi", "phi1", "phi2", "left", "right", "ante", "cons"):
            child = getattr(node, attr, None)
            if child is not None and isinstance(child, STRELFormula):
                self._register_submodules(child, _seen)

    def forward(
        self,
        signal: torch.Tensor,
        lab: torch.Tensor,
        differentiable: bool = True,
    ) -> torch.Tensor:
        """
        Evaluate the formula.

        Args:
            differentiable: True (default for parametric) → smooth conservative LBs.
                            False → exact semantics (for eval / reporting).
                            STRELMonitor hard-codes True so LiRPA traces the
                            smooth path.
        """
        return self.formula.evaluate(signal, lab, differentiable)
