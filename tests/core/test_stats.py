import pytest

from omakase_bench.core.stats import mcnemar_exact, wilson


def test_wilson_known_values():
    lo, hi = wilson(20, 20)
    assert lo == pytest.approx(0.8389, abs=1e-3) and hi == 1.0
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)


def test_wilson_empty():
    assert wilson(0, 0) == (0.0, 1.0)


def test_mcnemar():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(6, 0) == pytest.approx(0.03125)
