"""
Example 2 — Connecting a deep learning trajectory generator with STREL.

We build a minimal RolloutModule (nn.Module) that:
  1. Takes initial states  y0 : [B, N, F]
  2. Runs a learned policy for T steps
  3. Returns the full trajectory  signal : [B, N, F, T]

The STREL monitor is then chained inline, so the entire computation graph
  y0  →  rollout  →  signal  →  STREL  →  ρ  →  loss  →  backward()
is differentiable end-to-end.

KEY: `differentiable` flag
  evaluate(signal, lab)                   # exact semantics — for reporting
  evaluate(signal, lab, differentiable=True)  # smooth conservative LBs — for training

The smooth operators (smooth_min / smooth_max) are conservative lower bounds
of the exact min/max, so the gradient descent signal is valid and the true
robustness is never overstated.

Shapes:
    B = batch,  N = agents,  F = features,  T = horizon
    signal[b, n, :, t] = state of agent n at time t in batch b
    lab    : [B, N, T]  fixed integer labels (not differentiated)
"""

import torch
import torch.nn as nn

from strel import (
    AtomicPredicate, And, Always, Eventually, Reach,
    BooleanEval,
)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
B, N, F, T = 4, 5, 6, 12   # batch, agents, features, horizon
HIDDEN = 64
BETA = 20.0


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------

class AgentPolicy(nn.Module):
    """
    Per-agent MLP policy: maps the current state of ONE agent to its next-step
    delta.  The same weights are shared across agents (homogeneous policy).

    Input:  [B*N, F]
    Output: [B*N, F]  (state delta)
    """

    def __init__(self, feature_dim: int, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, feature_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


# ---------------------------------------------------------------------------
# Rollout module
# ---------------------------------------------------------------------------

class RolloutModule(nn.Module):
    """
    Unrolls the policy for T steps and stacks the trajectory.

    Args:
        policy:  AgentPolicy (or any nn.Module mapping [B*N, F] → [B*N, F])
        horizon: Number of time steps T.

    Forward:
        y0    : [B, N, F]  initial state
        eps   : [B, N, F, T]  optional additive disturbance (set to 0 for
                nominal; set to bounded noise for robust training)

    Returns:
        signal : [B, N, F, T]  full trajectory
    """

    def __init__(self, policy: nn.Module, horizon: int):
        super().__init__()
        self.policy = policy
        self.T = horizon

    def forward(
        self,
        y0: torch.Tensor,
        eps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, F = y0.shape
        if eps is None:
            eps = torch.zeros(B, N, F, self.T, device=y0.device, dtype=y0.dtype)

        states = []
        state = y0  # [B, N, F]

        for t in range(self.T):
            states.append(state)
            flat = state.reshape(B * N, F)           # merge batch+agent
            delta = self.policy(flat).reshape(B, N, F)
            state = state + delta + eps[:, :, :, t]  # step + disturbance

        # Stack: list of T × [B, N, F]  →  [B, N, F, T]
        signal = torch.stack(states, dim=-1)
        return signal


# ---------------------------------------------------------------------------
# STREL property (over the generated traces)
# ---------------------------------------------------------------------------

# Agents maintain positive x-velocity (feature 0)
vel_x_pos = AtomicPredicate(var_ind=0, threshold=0.0)

# Position x > 0.3 (feature 2)
pos_safe   = AtomicPredicate(var_ind=2, threshold=0.3)

# Always: vel_x positive AND pos_x safe
safety_formula = Always(And(vel_x_pos, pos_safe), beta=BETA)

# Eventually: some reachable neighbour within distance 1.5 is at pos_x > 1.0
at_target = AtomicPredicate(var_ind=2, threshold=1.0)
reach_target = Reach(
    phi1=vel_x_pos,
    phi2=at_target,
    d1=0.0,
    d2=1.5,
    pos_x_ind=2,
    pos_y_ind=3,
)
liveness_formula = Eventually(reach_target, beta=BETA)

bool_reach = BooleanEval(reach_target)


# ---------------------------------------------------------------------------
# Training loop skeleton
# ---------------------------------------------------------------------------

def robustness_loss(rho: torch.Tensor) -> torch.Tensor:
    """
    Loss = -mean(ρ) : minimising this maximises the average robustness.
    Only applied at t=0 (initial time step), propagating gradients back
    through the entire rollout.
    """
    return -rho[:, :, 0].mean()


def run_example():
    policy   = AgentPolicy(feature_dim=F)
    rollout  = RolloutModule(policy, horizon=T)

    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # Fixed labels for this example (agent 0 is "goal", rest are normal)
    lab = torch.zeros(B, N, T, dtype=torch.long)
    lab[:, 0, :] = 1

    for step in range(5):
        y0 = torch.randn(B, N, F) * 0.1   # random initial states

        # --- nominal rollout (eps=0) ---
        signal = rollout(y0)                              # [B, N, F, T]

        # --- TRAINING pass: smooth conservative lower bounds ---
        # differentiable=True → smooth_min/smooth_max (≤ exact), gradient flows
        rho_safety   = safety_formula.evaluate(signal, lab, differentiable=True)
        rho_liveness = liveness_formula.evaluate(signal, lab, differentiable=True)

        # --- combined loss: maximise both smooth robustnesses ---
        loss = robustness_loss(rho_safety) + robustness_loss(rho_liveness)

        opt.zero_grad()
        loss.backward()   # gradients flow: loss → ρ_smooth → signal → policy
        opt.step()

        # --- REPORTING pass: exact semantics (non-differentiable, use detach) ---
        with torch.no_grad():
            rho_exact = safety_formula.evaluate(signal, lab)       # exact
            # BooleanEval always uses differentiable=False internally
            sat_rate  = bool_reach.satisfaction_rate(signal, lab, reduce="all")

        print(
            f"step {step:3d}  loss={loss.item():7.4f}  "
            f"ρ_smooth={rho_safety[:,:,0].mean().item():6.3f}  "
            f"ρ_exact={rho_exact[:,:,0].mean().item():6.3f}  "
            f"sat_reach={sat_rate.item():.3f}"
        )

    print("\nGradient check — policy.net[0].weight.grad is not None:",
          policy.net[0].weight.grad is not None)
    print("Note: ρ_smooth ≤ ρ_exact always (conservative lower bound).")


if __name__ == "__main__":
    run_example()
