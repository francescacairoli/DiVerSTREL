"""
Example 1 — Defining STREL properties and evaluating them.

Signal convention:
    signal : [B, N, F, T]
        B=2  batch
        N=4  agents
        F=6  features: [vel_x, vel_y, pos_x, pos_y, dist_to_goal, class_feat]
        T=10 time steps

    lab    : [B, N, T]  integer class id per agent per step
             0 = normal agent, 1 = obstacle
"""

import torch
from strel import (
    AtomicPredicate, TrueFormula, Not, And, Or, Always, Eventually, Until,
    Reach, Escape, BooleanEval,
)

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# Synthetic trace
# ---------------------------------------------------------------------------
B, N, F, T = 2, 4, 6, 10
signal = torch.randn(B, N, F, T) * 0.3
# Feature layout: [vel_x=0, vel_y=1, pos_x=2, pos_y=3, dist_goal=4, feat=5]

# pos_x/y spread agents across [0, 2]
signal[:, :, 2, :] = torch.linspace(0, 2, N).view(1, N, 1).expand(B, N, T)
signal[:, :, 3, :] = torch.rand(B, N, T) * 2.0

# Labels: agent 0,1,2 are normal (0), agent 3 is an obstacle (1)
lab = torch.zeros(B, N, T, dtype=torch.long)
lab[:, 3, :] = 1

# ---------------------------------------------------------------------------
# 1. Atomic predicates
# ---------------------------------------------------------------------------
# "velocity in x is positive"
vel_positive = AtomicPredicate(var_ind=0, threshold=0.0)

# "distance to goal is below 0.5"
close_to_goal = AtomicPredicate(var_ind=4, threshold=0.5, lte=True)

# "distance to goal above 1.0  (i.e. NOT close)"
far_from_goal = AtomicPredicate(var_ind=4, threshold=1.0)

# ---------------------------------------------------------------------------
# 2. Boolean connectives
# ---------------------------------------------------------------------------
moving_and_close = And(vel_positive, close_to_goal)
moving_or_far    = Or(vel_positive, far_from_goal)

# ---------------------------------------------------------------------------
# 3. Temporal operators
# ---------------------------------------------------------------------------
# "Always eventually close to goal" — standard reach-avoid style
always_close   = Always(close_to_goal, beta=20.0)
eventually_close = Eventually(close_to_goal, beta=20.0)

# "Vel positive UNTIL close to goal"
vel_until_close = Until(vel_positive, close_to_goal, beta=10.0)

# ---------------------------------------------------------------------------
# 4. Spatial operators (Reach / Escape)
# ---------------------------------------------------------------------------
normal_agents = torch.tensor([1.0])   # label 0 is "normal"

# "From any normal agent, there exists a normal agent within distance [0, 1.5]
#  that is close to goal"
reach_goal_nbr = Reach(
    phi1=TrueFormula(),      # intermediate nodes: any
    phi2=close_to_goal,      # destination: close to goal
    d1=0.0,
    d2=1.5,
    labels1=None,
    labels2=None,
    pos_x_ind=2,
    pos_y_ind=3,
)

# "There exists an escape path of length ∈ [0, 2] through non-obstacle nodes
#  where agents maintain positive velocity"
escape_safe = Escape(
    phi=vel_positive,
    d1=0.0,
    d2=2.0,
    labels=None,
    pos_x_ind=2,
    pos_y_ind=3,
)

# ---------------------------------------------------------------------------
# 5. Evaluate — quantitative semantics
# ---------------------------------------------------------------------------
print("=== Quantitative semantics ===")
for name, formula in [
    ("vel_positive",    vel_positive),
    ("close_to_goal",   close_to_goal),
    ("moving_and_close",moving_and_close),
    ("always_close",    always_close),
    ("vel_until_close", vel_until_close),
    ("reach_goal_nbr",  reach_goal_nbr),
    ("escape_safe",     escape_safe),
]:
    rho = formula.evaluate(signal, lab)   # [B, N, T]
    print(f"  {name:22s}  rho.shape={list(rho.shape)}  "
          f"mean={rho.mean():.3f}  min={rho.min():.3f}  max={rho.max():.3f}")

# ---------------------------------------------------------------------------
# 6. Boolean semantics
# ---------------------------------------------------------------------------
print("\n=== Boolean semantics ===")
bool_phi = BooleanEval(And(vel_positive, reach_goal_nbr))
sat = bool_phi.is_satisfied(signal, lab)          # BoolTensor [B, N, T]
rate = bool_phi.satisfaction_rate(signal, lab, reduce="all")
print(f"  satisfaction rate (over B,N,T): {rate.item():.3f}")
print(f"  satisfied shape: {list(sat.shape)}")

# ---------------------------------------------------------------------------
# 7. Nesting spatial and temporal operators
# ---------------------------------------------------------------------------
print("\n=== Nested property ===")
# "It is always the case that there is a reachable neighbour eventually close"
nested = Always(
    Eventually(reach_goal_nbr, beta=10.0),
    beta=10.0,
)
rho_nested = nested.evaluate(signal, lab)
print(f"  nested rho: mean={rho_nested.mean():.3f}")
