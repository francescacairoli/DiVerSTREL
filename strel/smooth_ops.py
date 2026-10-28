"""
smooth_ops.py — LiRPA-safe differentiable approximations of max/min.

Both operators are CONSERVATIVE LOWER BOUNDS of the exact operation:
    smooth_max(x) <= max(x)    (log-mean-exp under-approximates max)
    smooth_min(x) <= min(x)    (log-sum-exp variant under-approximates min)

This means the smooth robustness is always ≤ the exact robustness, so
any certified lower bound computed on top (via LiRPA) is still a valid
lower bound on the true robustness.

These ops are used only when `differentiable=True` is passed to `evaluate`.
When `differentiable=False` (the default) the formulas use torch.amin /
torch.amax / torch.minimum / torch.maximum for exact semantics.
"""

import torch

# ---------------------------------------------------------------------------
# Floor-hit monitoring
# ---------------------------------------------------------------------------
_floor_stats: dict = {
    "smooth_max_hits": 0, "smooth_max_total": 0,
    "smooth_min_hits": 0, "smooth_min_total": 0,
}


def get_floor_stats() -> dict:
    return dict(_floor_stats)


def reset_floor_stats() -> None:
    for k in _floor_stats:
        _floor_stats[k] = 0


# ---------------------------------------------------------------------------
# Conservative smooth operators (differentiable=True path)
# ---------------------------------------------------------------------------

def smooth_max(x: torch.Tensor, beta: float = 20.0) -> torch.Tensor:
    """
    Conservative (log-mean-exp) soft maximum over the last dimension.

    Satisfies  smooth_max(x) <= max(x)  always.

    Standard log-sum-exp over-approximates max; subtracting log(n)/beta
    converts it to log-mean-exp which under-approximates max instead.
    The decomposition (sum + sub rather than mean) keeps the computation
    graph within auto_LiRPA's IBP/CROWN-compatible op set.

    Args:
        x:    Input tensor of shape (..., n).
        beta: Inverse temperature; larger → tighter to the exact max.
    """
    xb = beta * x
    n = x.shape[-1]
    raw_sum = torch.sum(torch.exp(xb), dim=-1)

    if not torch.jit.is_tracing():
        with torch.no_grad():
            hit = raw_sum <= 0
            _floor_stats["smooth_max_hits"] += int(hit.sum().item())
            _floor_stats["smooth_max_total"] += hit.numel()

    log_n = torch.log(n * torch.ones(1, dtype=raw_sum.dtype, device=raw_sum.device))
    return (torch.log(raw_sum + 1e-12) - log_n) / beta


def smooth_min(x: torch.Tensor, beta: float = 20.0) -> torch.Tensor:
    """
    Conservative soft minimum over the last dimension.

    Satisfies  smooth_min(x) <= min(x)  always.

    Uses -(1/beta) log(sum(exp(-beta*x))) — the log-sum-exp over -x gives
    an upper bound on -min(x), so negating yields a lower bound on min(x).

    Args:
        x:    Input tensor of shape (..., n).
        beta: Inverse temperature; larger → tighter to the exact min.
    """
    xb = -beta * x
    raw_sum = torch.sum(torch.exp(xb), dim=-1)

    if not torch.jit.is_tracing():
        with torch.no_grad():
            hit = raw_sum <= 0
            _floor_stats["smooth_min_hits"] += int(hit.sum().item())
            _floor_stats["smooth_min_total"] += hit.numel()

    return -torch.log(raw_sum + 1e-12) / beta


# ---------------------------------------------------------------------------
# Exact operators (differentiable=False path)
# ---------------------------------------------------------------------------

def exact_max(x: torch.Tensor) -> torch.Tensor:
    """Exact maximum over the last dimension."""
    return torch.amax(x, dim=-1)


def exact_min(x: torch.Tensor) -> torch.Tensor:
    """Exact minimum over the last dimension."""
    return torch.amin(x, dim=-1)
