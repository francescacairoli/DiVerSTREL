"""
Example 3 — AutoLiRPA certification and backpropagation through the lower bound.

This example shows the full pipeline:

  y0  →  RolloutModule  →  signal  →  STRELFormula  →  ρ_nom
                                                       ↓
                                         BoundedModule (LiRPA)
                                                       ↓
                                         lb  (certified lower bound on ρ)
                                                       ↓
                                         loss = -[λ·lb + (1-λ)·ρ_nom]
                                                       ↓
                                         backward() through lb AND ρ_nom

Requirements:
    pip install auto_LiRPA torch

Architecture decisions for LiRPA compatibility:
  - Use smooth_max / smooth_min (NOT torch.logsumexp or torch.amax) — the
    implementations in smooth_ops.py are already IBP/CROWN-compatible.
  - Use torch.abs, not torch.relu, for the min/max identities in Reach/Escape.
  - Wrap only the STREL monitor (not the rollout) in BoundedModule: the
    certified bound is over STREL robustness given a ε-ball around the
    nominal signal, which is exactly the threat model for sensor noise.
  - For policy disturbances (action noise), wrap the rollout + monitor
    together — see note at the bottom.

Signal shapes: [B, N, F, T]
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Import STREL library
# ---------------------------------------------------------------------------
from strel import AtomicPredicate, And, Always, Eventually, Reach
from strel.parametric import ParametricFormula   # nn.Module wrapper

# ---------------------------------------------------------------------------
# Rollout (same as example 02)
# ---------------------------------------------------------------------------

class AgentPolicy(nn.Module):
    def __init__(self, F: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(F, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, F),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RolloutModule(nn.Module):
    def __init__(self, policy: nn.Module, horizon: int):
        super().__init__()
        self.policy = policy
        self.T = horizon

    def forward(self, y0: torch.Tensor, eps: torch.Tensor | None = None) -> torch.Tensor:
        B, N, F = y0.shape
        if eps is None:
            eps = torch.zeros(B, N, F, self.T, device=y0.device, dtype=y0.dtype)
        states = []
        state = y0
        for t in range(self.T):
            states.append(state)
            delta = self.policy(state.reshape(B * N, F)).reshape(B, N, F)
            state = state + delta + eps[:, :, :, t]
        return torch.stack(states, dim=-1)


# ---------------------------------------------------------------------------
# The STREL monitor as an nn.Module
# ---------------------------------------------------------------------------

class STRELMonitor(nn.Module):
    """
    Wraps a STREL formula tree as an nn.Module so BoundedModule can trace it.

    The BoundedModule treats `signal` as the perturbed input; `lab` is fixed
    (class labels don't change with noise).

    To make the forward signature compatible with LiRPA (single input), we
    fix `lab` at construction time and expose only `signal` as the forward arg.
    """

    def __init__(self, formula, lab_fixed: torch.Tensor):
        super().__init__()
        self.formula = formula
        # Register lab as a buffer (not a parameter — not perturbed)
        self.register_buffer("lab", lab_fixed)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Returns mean robustness over agents and time: scalar per batch.

        Hard-codes differentiable=True so BoundedModule traces the smooth
        (log-mean-exp / log-sum-exp) path, which is IBP/CROWN-compatible.
        The smooth robustness is a conservative lower bound of the exact
        robustness, so the certified lb is still a valid bound on the true ρ.
        """
        rho = self.formula.evaluate(signal, self.lab, differentiable=True)
        return rho.mean(dim=(1, 2))                      # [B]


# ---------------------------------------------------------------------------
# Certification loop
# ---------------------------------------------------------------------------

def run_certified_training():
    # Hyper-parameters
    B, N, F, T = 4, 5, 6, 12
    EPS_MAX = 0.05       # maximum sensor-noise magnitude
    K_ITERS = 10         # training iterations
    BETA    = 20.0
    HIDDEN  = 64

    try:
        from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
    except ImportError:
        print("auto_LiRPA not installed. Run:  pip install auto_LiRPA")
        print("Showing the code structure only (no actual certification).")
        _show_structure_only(B, N, F, T, BETA)
        return

    # --- Build models ---
    policy  = AgentPolicy(F, HIDDEN)
    rollout = RolloutModule(policy, T)

    # --- STREL property ---
    vel_pos  = AtomicPredicate(var_ind=0, threshold=0.0)
    pos_safe = AtomicPredicate(var_ind=2, threshold=0.3)
    formula  = Always(And(vel_pos, pos_safe), beta=BETA)

    lab = torch.zeros(B, N, T, dtype=torch.long)   # all normal agents
    monitor = STRELMonitor(formula, lab)

    # --- Wrap monitor in BoundedModule ---
    dummy_signal = torch.zeros(B, N, F, T)
    lirpa_monitor = BoundedModule(monitor, (dummy_signal,))

    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

    for k in range(K_ITERS):
        lam = k / max(K_ITERS - 1, 1)       # curriculum: 0 → 1
        eps_cur = EPS_MAX * lam

        y0 = torch.randn(B, N, F) * 0.1

        # Nominal rollout
        signal_nom = rollout(y0)             # [B, N, F, T]

        # Nominal robustness (differentiable)
        rho_nom = monitor(signal_nom)        # [B]

        # Certified lower bound via IBP / CROWN
        ptb = PerturbationLpNorm(norm=float("inf"), eps=eps_cur)
        signal_bnd = BoundedTensor(signal_nom, ptb)

        method = "CROWN-IBP" if lam < 0.8 else "CROWN"
        lb, _ = lirpa_monitor.compute_bounds(
            x=(signal_bnd,),
            method=method,
        )   # lb : [B]  — certified lower bound on ρ

        # Mixed objective: optimise certified + nominal robustness
        mixed = lam * lb + (1.0 - lam) * rho_nom
        loss  = -mixed.mean()

        opt.zero_grad()
        loss.backward()   # gradients flow through BOTH lb and rho_nom
        opt.step()

        cert_sat_rate = (lb > 0).float().mean().item()
        print(
            f"iter {k:3d}  ε={eps_cur:.3f}  method={method:10s}  "
            f"loss={loss.item():7.4f}  "
            f"ρ_nom={rho_nom.mean().item():6.3f}  "
            f"lb={lb.mean().item():6.3f}  "
            f"cert_sat={cert_sat_rate:.3f}"
        )

    print("\nDone.  Gradients flowed through the certified lower bound.")


def _show_structure_only(B, N, F, T, BETA):
    """Demonstrate the shapes without auto_LiRPA."""
    vel_pos  = AtomicPredicate(var_ind=0, threshold=0.0)
    pos_safe = AtomicPredicate(var_ind=2, threshold=0.3)
    formula  = Always(And(vel_pos, pos_safe), beta=BETA)

    signal = torch.randn(B, N, F, T, requires_grad=True)
    lab    = torch.zeros(B, N, T, dtype=torch.long)

    # Smooth path (differentiable=True) for backprop
    rho_smooth = formula.evaluate(signal, lab, differentiable=True)
    loss = -rho_smooth.mean()
    loss.backward()

    # Exact path for reporting (no gradient needed)
    with torch.no_grad():
        rho_exact = formula.evaluate(signal, lab, differentiable=False)

    print(f"signal.shape      = {list(signal.shape)}")
    print(f"rho.shape         = {list(rho_smooth.shape)}")
    print(f"ρ_smooth (LB)     = {rho_smooth.mean().item():.4f}")
    print(f"ρ_exact           = {rho_exact.mean().item():.4f}")
    print(f"ρ_smooth ≤ ρ_exact: {(rho_smooth.detach() <= rho_exact).all().item()}")
    print(f"signal.grad not None: {signal.grad is not None}")
    print()
    print("To add LiRPA certification, install auto_LiRPA and run:")
    print("  python3 examples/03_certified_training.py")
    print()
    print("The BoundedModule wraps the STRELMonitor and computes:")
    print("  lb, ub = lirpa_monitor.compute_bounds(")
    print("      x=(BoundedTensor(signal, PerturbationLpNorm(inf, eps=ε)),),")
    print("      method='CROWN-IBP'")
    print("  )")
    print("  loss = -(λ * lb + (1-λ) * rho_nom).mean()")
    print("  loss.backward()   # gradients flow through lb too")


# ---------------------------------------------------------------------------
# Usage note: wrapping the rollout + monitor together
# ---------------------------------------------------------------------------
#
# To certify over *action* disturbances (perturbations applied to policy
# outputs rather than observations), wrap RolloutModule + STRELMonitor in
# a single nn.Module whose input is y0 (the initial state) and whose output
# is the mean robustness, then BoundedModule wraps that joint module.
#
# class RolloutAndMonitor(nn.Module):
#     def forward(self, y0: Tensor) -> Tensor:
#         signal = rollout(y0)       # y0 → trajectory
#         return monitor(signal)     # trajectory → scalar ρ per batch
#
# ptb = PerturbationLpNorm(norm=inf, eps=eps_y0)
# y0_bnd = BoundedTensor(y0, ptb)
# lb, _ = lirpa_joint.compute_bounds(x=(y0_bnd,), method="CROWN-IBP")
#
# This propagates bounds from y0 perturbations all the way through the
# rollout and the STREL monitor in one shot.

if __name__ == "__main__":
    run_certified_training()
