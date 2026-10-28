"""
strel_monitor.strel
===================

STREL (Spatial-Temporal Reach and Escape Logic) monitoring with three
compatible semantics:

    quantitative  — real-valued robustness (differentiable, AutoLiRPA-safe)
    boolean       — {-1, +1} based on sign of robustness (non-differentiable)
    parametric    — quantitative with learnable threshold / distance parameters

All three share the same signal convention:
    signal : [B, N, F, T]  — batch × agents × features × time
    lab    : [B, N, T]     — integer class label per agent per time step
"""

from .smooth_ops import smooth_max, smooth_min, get_floor_stats, reset_floor_stats

from .quantitative import (
    STRELFormula,
    AtomicPredicate,
    TrueFormula,
    FalseFormula,
    Not,
    And,
    Or,
    Implies,
    Always,
    Eventually,
    Until,
    Reach,
    Escape,
)

from .boolean import BooleanEval

from .parametric import (
    ParametricAtomicPredicate,
    ParametricAlways,
    ParametricEventually,
    ParametricReach,
    ParametricFormula,
)

__all__ = [
    # operators
    "smooth_max", "smooth_min", "get_floor_stats", "reset_floor_stats",
    # quantitative
    "STRELFormula", "AtomicPredicate", "TrueFormula", "FalseFormula",
    "Not", "And", "Or", "Implies", "Always", "Eventually", "Until",
    "Reach", "Escape",
    # boolean
    "BooleanEval",
    # parametric
    "ParametricAtomicPredicate", "ParametricAlways", "ParametricEventually",
    "ParametricReach", "ParametricFormula",
]
