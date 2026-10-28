"""
tests/test_semantics.py — Sanity-check the three semantics layers.

Run with:  python -m pytest tests/  or  python tests/test_semantics.py
"""

import torch
import pytest

from strel import (
    AtomicPredicate, TrueFormula, FalseFormula,
    Not, And, Or, Always, Eventually, Until, Reach, Escape,
    BooleanEval,
)
from strel.parametric import (
    ParametricAtomicPredicate, ParametricAlways, ParametricFormula,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, N, F, T = 2, 3, 6, 8


def make_signal(val: float = 0.0) -> torch.Tensor:
    s = torch.full((B, N, F, T), val)
    # pos_x = feature 2, pos_y = feature 3: spread agents spatially
    s[:, :, 2, :] = torch.linspace(0, 2, N).view(1, N, 1)
    s[:, :, 3, :] = 0.0
    return s


def make_lab() -> torch.Tensor:
    return torch.zeros(B, N, T, dtype=torch.long)


# ---------------------------------------------------------------------------
# Quantitative semantics — exact mode (differentiable=False, default)
# ---------------------------------------------------------------------------

class TestQuantitativeExact:

    def test_atomic_positive(self):
        """Signal above threshold → positive robustness."""
        phi = AtomicPredicate(var_ind=0, threshold=0.5)
        signal = make_signal(1.0)   # feature 0 = 1.0 > 0.5
        lab = make_lab()
        rho = phi.evaluate(signal, lab)   # default: differentiable=False
        assert (rho > 0).all(), f"Expected all positive, got min={rho.min()}"

    def test_atomic_negative(self):
        """Signal below threshold → negative robustness."""
        phi = AtomicPredicate(var_ind=0, threshold=0.5)
        signal = make_signal(0.0)   # feature 0 = 0.0 < 0.5
        lab = make_lab()
        rho = phi.evaluate(signal, lab)
        assert (rho < 0).all()

    def test_not_flips_sign(self):
        phi = AtomicPredicate(var_ind=0, threshold=0.5)
        signal = make_signal(1.0)
        lab = make_lab()
        rho_phi = phi.evaluate(signal, lab)
        rho_not = Not(phi).evaluate(signal, lab)
        assert torch.allclose(rho_phi, -rho_not)

    def test_and_is_min(self):
        """And(True, phi) = phi  (exact: min(+inf, rho) = rho)."""
        phi = AtomicPredicate(var_ind=0, threshold=0.5)
        signal = make_signal(1.0)
        lab = make_lab()
        rho_phi = phi.evaluate(signal, lab)
        rho_and = And(TrueFormula(), phi).evaluate(signal, lab, differentiable=False)
        assert torch.allclose(rho_phi, rho_and)

    def test_or_is_max(self):
        """Or(False, phi) = phi  (exact: max(-inf, rho) = rho)."""
        phi = AtomicPredicate(var_ind=0, threshold=0.5)
        signal = make_signal(1.0)
        lab = make_lab()
        rho_phi = phi.evaluate(signal, lab)
        rho_or  = Or(FalseFormula(), phi).evaluate(signal, lab, differentiable=False)
        assert torch.allclose(rho_phi, rho_or)

    def test_always_shape(self):
        phi = Always(AtomicPredicate(0, 0.0))
        rho = phi.evaluate(make_signal(1.0), make_lab())
        assert rho.shape == (B, N, T)

    def test_eventually_shape(self):
        phi = Eventually(AtomicPredicate(0, 0.0))
        rho = phi.evaluate(make_signal(1.0), make_lab())
        assert rho.shape == (B, N, T)

    def test_until_shape(self):
        phi = Until(AtomicPredicate(0, 0.0), AtomicPredicate(1, 0.0))
        rho = phi.evaluate(make_signal(1.0), make_lab())
        assert rho.shape == (B, N, T)

    def test_reach_shape(self):
        phi = Reach(TrueFormula(), AtomicPredicate(0, 0.0), d1=0.0, d2=2.0)
        rho = phi.evaluate(make_signal(1.0), make_lab())
        assert rho.shape == (B, N, T)

    def test_escape_shape(self):
        phi = Escape(AtomicPredicate(0, 0.0), d1=0.0, d2=2.0)
        rho = phi.evaluate(make_signal(1.0), make_lab())
        assert rho.shape == (B, N, T)


# ---------------------------------------------------------------------------
# Quantitative semantics — smooth mode (differentiable=True)
# ---------------------------------------------------------------------------

class TestQuantitativeSmooth:

    def test_smooth_is_lower_bound_of_exact_always(self):
        """smooth_rho <= exact_rho for Always (conservative LB property)."""
        phi = Always(AtomicPredicate(0, 0.5), beta=20.0)
        signal = make_signal(1.0)
        lab    = make_lab()
        rho_exact  = phi.evaluate(signal, lab, differentiable=False)
        rho_smooth = phi.evaluate(signal, lab, differentiable=True)
        assert (rho_smooth <= rho_exact + 1e-5).all(), (
            f"smooth robustness exceeded exact by {(rho_smooth - rho_exact).max().item():.4f}"
        )

    def test_smooth_is_lower_bound_of_exact_eventually(self):
        """smooth_rho <= exact_rho for Eventually."""
        phi = Eventually(AtomicPredicate(0, 0.5), beta=20.0)
        signal = make_signal(1.0)
        lab    = make_lab()
        rho_exact  = phi.evaluate(signal, lab, differentiable=False)
        rho_smooth = phi.evaluate(signal, lab, differentiable=True)
        assert (rho_smooth <= rho_exact + 1e-5).all()

    def test_smooth_is_lower_bound_of_exact_and(self):
        """smooth_rho <= exact_rho for And."""
        phi = And(AtomicPredicate(0, 0.0), AtomicPredicate(1, 0.0))
        signal = make_signal(1.0)
        lab    = make_lab()
        rho_exact  = phi.evaluate(signal, lab, differentiable=False)
        rho_smooth = phi.evaluate(signal, lab, differentiable=True)
        assert (rho_smooth <= rho_exact + 1e-5).all()

    def test_smooth_and_exact_same_sign(self):
        """Smooth and exact robustness should agree on satisfaction (sign)."""
        phi = Always(And(AtomicPredicate(0, 0.0), AtomicPredicate(1, 0.0)))
        signal = make_signal(1.0)
        lab    = make_lab()
        rho_exact  = phi.evaluate(signal, lab, differentiable=False)
        rho_smooth = phi.evaluate(signal, lab, differentiable=True)
        # When exact is positive (satisfied), smooth must also be positive
        # (it's a lower bound — if smooth > 0 then exact >= smooth > 0)
        mask_satisfied = rho_exact > 0
        if mask_satisfied.any():
            # smooth can be negative even if exact is positive (conservative),
            # but exact can't be negative when smooth is positive
            assert (rho_exact[rho_smooth > 0] > 0).all(), \
                "exact is negative but smooth was positive — violates LB property"

    def test_gradient_flows_smooth(self):
        """Loss gradient must reach the signal tensor in smooth mode."""
        signal = make_signal(1.0).requires_grad_(True)
        lab = make_lab()
        phi = Always(And(AtomicPredicate(0, 0.0), AtomicPredicate(1, 0.0)))
        rho = phi.evaluate(signal, lab, differentiable=True)
        (-rho.mean()).backward()
        assert signal.grad is not None
        assert not signal.grad.isnan().any()

    def test_gradient_absent_exact(self):
        """Exact mode uses amin/amax — gradients from exact min may be None or zero."""
        # This is a documentation test: exact semantics may or may not produce gradients
        # depending on PyTorch version, but the smooth mode is the reliable path.
        signal = make_signal(1.0).requires_grad_(True)
        lab = make_lab()
        phi = AtomicPredicate(0, 0.0)   # linear — always differentiable
        rho = phi.evaluate(signal, lab, differentiable=False)
        (-rho.mean()).backward()
        assert signal.grad is not None   # atomic is linear, always differentiable

    def test_smooth_shapes_match_exact(self):
        """Output shape is the same regardless of mode."""
        phi = Always(And(AtomicPredicate(0, 0.0), AtomicPredicate(1, 0.0)))
        signal = make_signal(1.0)
        lab    = make_lab()
        rho_e = phi.evaluate(signal, lab, differentiable=False)
        rho_s = phi.evaluate(signal, lab, differentiable=True)
        assert rho_e.shape == rho_s.shape == (B, N, T)


# ---------------------------------------------------------------------------
# Boolean semantics
# ---------------------------------------------------------------------------

class TestBoolean:

    def test_satisfied_when_positive(self):
        phi = AtomicPredicate(var_ind=0, threshold=0.5)
        signal = make_signal(1.0)
        lab = make_lab()
        bool_phi = BooleanEval(phi)
        assert bool_phi.is_satisfied(signal, lab).all()

    def test_violated_when_negative(self):
        phi = AtomicPredicate(var_ind=0, threshold=0.5)
        signal = make_signal(0.0)
        lab = make_lab()
        bool_phi = BooleanEval(phi)
        assert not bool_phi.is_satisfied(signal, lab).any()

    def test_satisfaction_rate_one(self):
        phi = TrueFormula()
        bool_phi = BooleanEval(phi)
        rate = bool_phi.satisfaction_rate(make_signal(), make_lab(), reduce="all")
        assert rate.item() == pytest.approx(1.0)

    def test_satisfaction_rate_zero(self):
        phi = FalseFormula()
        bool_phi = BooleanEval(phi)
        rate = bool_phi.satisfaction_rate(make_signal(), make_lab(), reduce="all")
        assert rate.item() == pytest.approx(0.0)

    def test_boolean_uses_exact_semantics(self):
        """BooleanEval always runs with differentiable=False (exact)."""
        phi = Always(AtomicPredicate(0, 0.5))
        signal = make_signal(1.0)
        lab = make_lab()
        bool_phi = BooleanEval(phi)
        # Check: result equals sign of exact robustness
        rho_exact = phi.evaluate(signal, lab, differentiable=False)
        bool_result = bool_phi.evaluate(signal, lab)
        expected = torch.where(rho_exact > 0.0, torch.ones_like(rho_exact), -torch.ones_like(rho_exact))
        assert torch.allclose(bool_result, expected)


# ---------------------------------------------------------------------------
# Parametric semantics
# ---------------------------------------------------------------------------

class TestParametric:

    def test_parametric_atom_gradient(self):
        """Gradient must flow into the learned threshold."""
        atom = ParametricAtomicPredicate(var_ind=0, init_threshold=0.0)
        signal = make_signal(1.0)
        lab = make_lab()
        rho = atom.evaluate(signal, lab, differentiable=True)
        (-rho.mean()).backward()
        assert atom.threshold.grad is not None

    def test_parametric_formula_parameters(self):
        """ParametricFormula must expose its leaf parameters."""
        phi = ParametricFormula(
            ParametricAlways(
                ParametricAtomicPredicate(var_ind=0, init_threshold=0.5),
                init_beta=10.0,
            )
        )
        params = list(phi.parameters())
        assert len(params) >= 2   # threshold + log_beta

    def test_parametric_threshold_updates(self):
        """Optimiser step changes the threshold."""
        atom = ParametricAtomicPredicate(var_ind=0, init_threshold=0.0)
        signal = make_signal(0.0)
        lab = make_lab()
        opt = torch.optim.SGD([atom.threshold], lr=0.1)
        before = atom.threshold.item()
        rho = atom.evaluate(signal, lab, differentiable=True)
        (-rho.mean()).backward()
        opt.step()
        after = atom.threshold.item()
        assert before != after

    def test_parametric_exact_mode(self):
        """Parametric formulas also support differentiable=False for reporting."""
        phi = ParametricFormula(
            ParametricAlways(
                ParametricAtomicPredicate(var_ind=0, init_threshold=0.5),
            )
        )
        signal = make_signal(1.0)
        lab = make_lab()
        # differentiable=True: smooth (default for parametric)
        rho_smooth = phi(signal, lab, differentiable=True)
        # differentiable=False: exact (for reporting after training)
        with torch.no_grad():
            rho_exact = phi(signal, lab, differentiable=False)
        assert (rho_smooth.detach() <= rho_exact + 1e-5).all()

    def test_forward_differentiable_true(self):
        """ParametricFormula.forward defaults to differentiable=True."""
        phi = ParametricFormula(
            ParametricAtomicPredicate(var_ind=0, init_threshold=0.0)
        )
        signal = make_signal(1.0).requires_grad_(True)
        lab = make_lab()
        rho = phi(signal, lab)   # calls forward → differentiable=True
        (-rho.mean()).backward()
        assert signal.grad is not None


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    test_classes = [TestQuantitativeExact, TestQuantitativeSmooth, TestBoolean, TestParametric]
    results = []
    for cls in test_classes:
        obj = cls()
        for name in [m for m in dir(obj) if m.startswith("test_")]:
            try:
                getattr(obj, name)()
                results.append(f"  PASS  {cls.__name__}.{name}")
            except Exception as e:
                results.append(f"  FAIL  {cls.__name__}.{name}  →  {e}")

    for r in results:
        print(r)

    n_fail = sum(1 for r in results if "FAIL" in r)
    print(f"\n{len(results) - n_fail}/{len(results)} tests passed.")
    sys.exit(n_fail)
