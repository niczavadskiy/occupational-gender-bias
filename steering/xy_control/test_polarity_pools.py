"""CPU tests for polarity pool builder (no Stage C pack required for unit bits)."""

from __future__ import annotations

import math

from steering.xy_control.analyze_polarity_pools import bonferroni, stratified_bootstrap_p


def test_bonferroni_m2() -> None:
    assert bonferroni([0.01, 0.04]) == [0.02, 0.08]
    assert bonferroni([0.6, 0.01]) == [1.0, 0.02]


def test_stratified_bootstrap_rejects_large_effect() -> None:
    by = {
        "a": [2.0, 2.5, 1.5, 2.2],
        "b": [1.8, 2.1, 1.9, 2.0],
    }
    mean, lo, hi, p = stratified_bootstrap_p(by, n_boot=2000, seed=1)
    assert mean > 1.5
    assert lo > 0
    assert p < 0.05


def test_stratified_bootstrap_nullish() -> None:
    by = {
        "a": [0.1, -0.1, 0.05, -0.05],
        "b": [0.2, -0.2, 0.0, 0.0],
    }
    mean, lo, hi, p = stratified_bootstrap_p(by, n_boot=2000, seed=2)
    assert abs(mean) < 0.15
    assert p > 0.05 or (lo <= 0 <= hi)


def main() -> int:
    test_bonferroni_m2()
    test_stratified_bootstrap_rejects_large_effect()
    test_stratified_bootstrap_nullish()
    print("ok", math.isfinite(1.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
