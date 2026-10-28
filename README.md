# DiVerSTREL

**Di**fferentiable and **Ver**ifiable **STREL** monitoring.

A clean Python library for evaluating STREL (Spatial-Temporal Reach and Escape
Logic) properties on multi-agent trajectories, with three compatible semantics:
exact quantitative, boolean, and parametric.  The library is designed to be
plugged directly into PyTorch training loops and AutoLiRPA certification.

## At a glance

| Semantics | Class / wrapper | Differentiable | Learnable params | LiRPA-safe |
|-----------|----------------|:--------------:|:----------------:|:----------:|
| Quantitative | `STRELFormula` subclasses | ✓ (smooth mode) | — | ✓ |
| Boolean | `BooleanEval` | — | — | — |
| Parametric quantitative | `Parametric*` + `ParametricFormula` | ✓ | ✓ | ✓ |

## Signal convention

```
signal : [B, N, F, T]
    B  — batch size
    N  — number of agents / nodes
    F  — number of signal features per agent
    T  — number of time steps

lab    : [B, N, T]   integer class label per agent per time step
```

Every formula's `evaluate(signal, lab)` returns robustness `[B, N, T]`:
- **positive** → property satisfied at that agent / time
- **negative** → violated
- **magnitude** → margin / degree of satisfaction

## The `differentiable` flag

The same formula object can be evaluated in two modes:

```python
rho_exact  = phi.evaluate(signal, lab)                      # default: exact
rho_smooth = phi.evaluate(signal, lab, differentiable=True) # conservative LB
```

| Mode | Operators | Gradients | When to use |
|------|-----------|-----------|-------------|
| `differentiable=False` (default) | `torch.amin`, `torch.amax`, `torch.minimum`, `torch.maximum` | limited (argmin/argmax) | evaluation, reporting, Boolean check |
| `differentiable=True` | `smooth_min`, `smooth_max` (log-mean-exp / log-sum-exp) | ✓ everywhere | training, certification, LiRPA |

**Conservative lower bound guarantee:**

```
smooth_max(x)  ≤  max(x)    (log-mean-exp under-approximates max)
smooth_min(x)  ≤  min(x)    (log-sum-exp variant under-approximates min)
```

This means `rho_smooth ≤ rho_exact` always, so any certified lower bound
computed on top of `rho_smooth` is still a valid lower bound on the true
robustness.  Gradients never over-claim satisfaction.

## Operators

### Atomic
| Class | Meaning |
|-------|---------|
| `AtomicPredicate(var_ind, threshold, lte=False)` | `signal[var_ind] ≥ threshold` |
| `TrueFormula()` | always satisfied |
| `FalseFormula()` | always violated |

### Boolean connectives
`Not`, `And`, `Or`, `Implies`

### Temporal (over T)
| Class | Meaning |
|-------|---------|
| `Always(φ, beta=20)` | G φ — globally / always |
| `Eventually(φ, beta=20)` | F φ — eventually |
| `Until(φ₁, φ₂, beta=20)` | φ₁ U φ₂ — until |

### Spatial
| Class | Meaning |
|-------|---------|
| `Reach(φ₁, φ₂, d1, d2)` | reach φ₂ via φ₁-satisfying path, distance ∈ [d1, d2] |
| `Escape(φ, d1, d2)` | escape via φ-satisfying path, distance ∈ [d1, d2] |

## Quick start

```python
from strel import AtomicPredicate, Always, And, BooleanEval

# 1. Define a spatio-temporal property
phi = Always(And(
    AtomicPredicate(var_ind=0, threshold=0.0),   # vel_x > 0
    AtomicPredicate(var_ind=2, threshold=0.3),   # pos_x > 0.3
))

# 2a. Exact quantitative robustness (for evaluation)
rho_exact = phi.evaluate(signal, lab)   # [B, N, T]

# 2b. Smooth robustness for training (conservative LB, gradients flow)
rho_train = phi.evaluate(signal, lab, differentiable=True)

# 3. Boolean check (always exact)
bool_phi = BooleanEval(phi)
satisfied = bool_phi.is_satisfied(signal, lab)        # BoolTensor
rate      = bool_phi.satisfaction_rate(signal, lab)   # scalar
```

## Deep-learning integration

```python
# Training loop — use differentiable=True so gradients reach the policy
signal = rollout_module(y0)                             # DL model → trajectory
rho    = phi.evaluate(signal, lab, differentiable=True) # smooth STREL robustness
loss   = -rho.mean()
loss.backward()   # gradients flow through smooth STREL → rollout

# Evaluation — use exact semantics
with torch.no_grad():
    rho_exact = phi.evaluate(signal, lab)   # exact, no smoothing bias
```

See `examples/02_dl_integration.py` for the full pattern.

## AutoLiRPA certification

```python
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

# STRELMonitor hard-codes differentiable=True in its forward() so that
# BoundedModule traces the smooth (IBP/CROWN-compatible) path.
monitor      = STRELMonitor(phi, lab_fixed)
lirpa_model  = BoundedModule(monitor, (dummy_signal,))

signal_bnd   = BoundedTensor(signal, PerturbationLpNorm(norm=float("inf"), eps=ε))
lb, _        = lirpa_model.compute_bounds(x=(signal_bnd,), method="CROWN-IBP")

# lb is a certified lower bound on smooth_rho ≤ true_rho,
# so lb is also a valid lower bound on the true robustness.
loss = -(λ * lb + (1 - λ) * rho_nom).mean()
loss.backward()   # gradients flow through the certified lower bound
```

See `examples/03_certified_training.py` for the full certified training loop.

## Parametric semantics (specification mining)

```python
from strel.parametric import ParametricAtomicPredicate, ParametricAlways, ParametricFormula

phi = ParametricFormula(
    ParametricAlways(
        ParametricAtomicPredicate(var_ind=2, init_threshold=0.0),
    )
)

opt = torch.optim.Adam(phi.parameters(), lr=0.05)

for step in range(100):
    # Parametric classes default to differentiable=True
    rho_pos = phi(signal_positive, lab).mean()
    rho_neg = phi(signal_negative, lab).mean()
    loss = torch.relu(1 - rho_pos) + torch.relu(1 + rho_neg)   # hinge
    opt.zero_grad(); loss.backward(); opt.step()

# Reporting after training (exact)
with torch.no_grad():
    rho_report = phi(signal_test, lab, differentiable=False)

print("Learned threshold:", phi.formula.phi.threshold.item())
```

See `examples/04_parametric_semantics.py` for the full example.

## LiRPA compatibility notes

- **smooth_max / smooth_min** use log-mean-exp and log-sum-exp with
  `torch.sum(torch.exp(...))` — _not_ `torch.logsumexp` or `torch.amax`,
  which break IBP propagation.
- All min/max identities in `Reach` / `Escape` use `|a−b|` arithmetic, not
  `torch.maximum` / `torch.minimum`, which are not supported by CROWN.  This
  `abs`-trick gives mathematically exact min/max and is LiRPA-safe.
- `BoundedModule` should wrap the **STREL monitor only** (with signal as the
  perturbed input) for sensor-noise certification, or the joint
  rollout + monitor for action-noise certification.
- `smooth_max` uses log-**mean**-exp (not log-sum-exp) to ensure
  `smooth_max(x) ≤ max(x)` — a conservative lower bound, so certified
  robustness never overstates the true value.
- `STRELMonitor.forward` hard-codes `differentiable=True` so `BoundedModule`
  always traces the smooth path.

## Repository layout

```
strel_monitor/
├── strel/
│   ├── __init__.py          # public API
│   ├── smooth_ops.py        # smooth_max, smooth_min (LiRPA-safe)
│   ├── quantitative.py      # all STRELFormula classes
│   ├── boolean.py           # BooleanEval wrapper
│   └── parametric.py        # Parametric* nn.Module classes
├── examples/
│   ├── 01_define_properties.py
│   ├── 02_dl_integration.py
│   ├── 03_certified_training.py
│   └── 04_parametric_semantics.py
├── tests/
│   └── test_semantics.py
├── requirements.txt
└── README.md
```
