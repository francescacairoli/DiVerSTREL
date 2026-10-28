"""
Example 4 — Parametric quantitative semantics (specification mining).

Demonstrates learning formula parameters (thresholds, distance bounds) from
labelled satisfaction data.  This is useful for:

  - Specification mining: find the "best" threshold that separates
    satisfying from violating traces.
  - Joint optimisation: learn policy AND specification parameters together.
  - Parametric certification: AutoLiRPA can propagate through learned
    thresholds inside ParametricFormula just as it propagates through
    the signal tensor.
"""

import torch
import torch.nn as nn

from strel import AtomicPredicate, Always, BooleanEval
from strel.parametric import (
    ParametricAtomicPredicate,
    ParametricAlways,
    ParametricReach,
    ParametricFormula,
)

torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------
B, N, F, T = 32, 4, 6, 10

# Generate "positive" traces: pos_x (feature 2) stays above 0.5
def make_traces(satisfying: bool):
    signal = torch.randn(B, N, F, T) * 0.1
    if satisfying:
        signal[:, :, 2, :] += 0.8    # pos_x high → satisfying
    else:
        signal[:, :, 2, :] -= 0.2    # pos_x low → violating
    return signal

signal_pos = make_traces(True)
signal_neg = make_traces(False)
lab = torch.zeros(B, N, T, dtype=torch.long)

# Ground-truth labels (+1 / -1 for positive/negative traces)
y_pos = torch.ones(B)
y_neg = -torch.ones(B)


# ---------------------------------------------------------------------------
# Parametric formula (to be mined)
# ---------------------------------------------------------------------------

# We want to learn the threshold θ such that:
#   G (pos_x ≥ θ)  separates pos from neg traces
phi = ParametricFormula(
    ParametricAlways(
        ParametricAtomicPredicate(var_ind=2, init_threshold=0.0),
        init_beta=10.0,
    )
)

print("Initial threshold:", phi.formula.phi.threshold.item())
print("Initial beta     :", phi.formula.phi.log_beta.exp().item())


# ---------------------------------------------------------------------------
# Mining loop
# ---------------------------------------------------------------------------

opt = torch.optim.Adam(phi.parameters(), lr=0.05)

for step in range(50):
    rho_pos = phi(signal_pos, lab).mean(dim=(1, 2))   # [B]
    rho_neg = phi(signal_neg, lab).mean(dim=(1, 2))   # [B]

    # Hinge loss: push ρ_pos > 0, ρ_neg < 0
    loss = torch.relu(1.0 - rho_pos).mean() + torch.relu(1.0 + rho_neg).mean()

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 10 == 0:
        theta = phi.formula.phi.threshold.item()
        beta  = phi.formula.phi.log_beta.exp().item()
        print(f"step {step:3d}  loss={loss.item():.4f}  θ={theta:.4f}  β={beta:.2f}")

print(f"\nLearned threshold θ = {phi.formula.phi.threshold.item():.4f}  "
      f"(true separating value ≈ 0.5)")


# ---------------------------------------------------------------------------
# Evaluate mined specification
# ---------------------------------------------------------------------------

bool_phi = BooleanEval(phi.formula.phi)
sat_pos  = bool_phi.satisfaction_rate(signal_pos, lab, reduce="all")
sat_neg  = bool_phi.satisfaction_rate(signal_neg, lab, reduce="all")
print(f"Satisfaction on positive traces: {sat_pos.item():.3f}  (should be ~1.0)")
print(f"Satisfaction on negative traces: {sat_neg.item():.3f}  (should be ~0.0)")


# ---------------------------------------------------------------------------
# Parametric Reach: learn the distance bound
# ---------------------------------------------------------------------------

print("\n--- Parametric Reach: learning distance bound d2 ---")

# Agents: pos_x spread over [0, 3]
signal2 = torch.randn(B, N, F, T) * 0.1
signal2[:, :, 2, :] = torch.linspace(0, 3, N).view(1, N, 1).expand(B, N, T) + signal2[:, :, 2, :]
signal2[:, :, 3, :] = torch.zeros(B, N, T)     # all agents at y=0

lab2 = torch.zeros(B, N, T, dtype=torch.long)

target_pred = AtomicPredicate(var_ind=2, threshold=1.5)   # pos_x > 1.5
reach_phi   = ParametricFormula(
    ParametricReach(
        phi1=AlwaysTrueHack(),   # see below
        phi2=target_pred,
        init_d1=0.0,
        init_d2=0.5,             # start with small range
        pos_x_ind=2,
        pos_y_ind=3,
    )
)


class AlwaysTrueHack:
    """Non-parametric True for intermediate nodes."""
    NEG_INF = -1e2
    def evaluate(self, signal, lab):
        B, N, _, T = signal.shape
        return torch.full((B, N, T), 1e2, dtype=signal.dtype, device=signal.device)


reach_phi = ParametricFormula(
    ParametricReach(
        phi1=AlwaysTrueHack(),
        phi2=target_pred,
        init_d1=0.0,
        init_d2=0.5,
        pos_x_ind=2,
        pos_y_ind=3,
    )
)
opt2 = torch.optim.Adam(reach_phi.parameters(), lr=0.1)

for step in range(30):
    rho = reach_phi(signal2, lab2)      # [B, N, T]
    loss = -rho.mean()                  # maximise reachability

    opt2.zero_grad()
    loss.backward()
    opt2.step()

    if step % 10 == 0:
        d2 = reach_phi.formula.d2.item()
        print(f"  step {step:3d}  d2={d2:.4f}  loss={loss.item():.4f}")

print(f"\nLearned d2 = {reach_phi.formula.d2.item():.4f}  "
      f"(agents span ~3 units; effective reach should converge toward that)")
