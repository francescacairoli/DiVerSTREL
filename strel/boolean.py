"""
boolean.py — Boolean STREL semantics.

Wraps any quantitative formula: `evaluate` returns {-1, +1} (as float)
based on the sign of the quantitative robustness.  Positive robustness → +1
(satisfied); non-positive → -1 (violated).

Because the boolean layer is a non-differentiable step function, it breaks
the gradient graph.  Always calls the underlying formula with
differentiable=False (exact semantics).  Use quantitative or parametric
semantics for training; use BooleanEval for final evaluation / metrics.
"""

import torch
from .quantitative import STRELFormula


class BooleanEval:
    """
    Converts a quantitative STREL formula into a boolean monitor.

    Args:
        formula:   Any STRELFormula (quantitative or parametric).
        threshold: Decision boundary (default 0). Satisfied iff ρ > threshold.

    Usage::

        quant_phi = Always(AtomicPredicate(0, 0.5))
        bool_phi  = BooleanEval(quant_phi)

        # Returns FloatTensor with values in {-1, +1}, shape [B, N, T]
        result = bool_phi.evaluate(signal, lab)

        # Or get a boolean mask:
        satisfied = bool_phi.is_satisfied(signal, lab)   # BoolTensor [B, N, T]

        # Fraction satisfied (scalar):
        rate = bool_phi.satisfaction_rate(signal, lab, reduce="all")
    """

    def __init__(self, formula: STRELFormula, threshold: float = 0.0):
        self.formula = formula
        self.threshold = threshold

    def evaluate(self, signal: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        """
        Returns +1 where robustness > threshold, -1 otherwise.
        Shape matches robustness: [B, N, T].

        Always uses exact (non-differentiable) semantics.  The step function
        breaks the gradient graph regardless, so smooth approximations would
        give no benefit.
        """
        # Boolean eval always runs exact semantics — step function is non-diff anyway
        rho = self.formula.evaluate(signal, lab, differentiable=False)
        return torch.where(rho > self.threshold, torch.ones_like(rho), -torch.ones_like(rho))

    def is_satisfied(self, signal: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        """Returns a bool tensor True where the property holds."""
        rho = self.formula.evaluate(signal, lab, differentiable=False)
        return rho > self.threshold

    def satisfaction_rate(
        self,
        signal: torch.Tensor,
        lab: torch.Tensor,
        reduce: str = "batch_agent",
    ) -> torch.Tensor:
        """
        Fraction of (batch, agent, time) triples where the property holds.

        Args:
            reduce: One of 'all', 'batch', 'batch_agent'.
                - 'all'         : scalar, mean over B×N×T
                - 'batch'       : [N, T], mean over B
                - 'batch_agent' : [T], mean over B×N
        """
        sat = self.is_satisfied(signal, lab).float()
        if reduce == "all":
            return sat.mean()
        elif reduce == "batch":
            return sat.mean(dim=0)
        elif reduce == "batch_agent":
            return sat.mean(dim=(0, 1))
        else:
            raise ValueError(f"Unknown reduce mode: {reduce!r}")
