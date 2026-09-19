"""Synthetic tests for stratified contrast PCA (no GPU)."""

from __future__ import annotations

import sys

import numpy as np

from steering.contrastive.rank import (
    build_steering_basis,
    cosine,
    interpret,
    layer_report,
    mean_diff,
    n_above_null,
    stratum_diffs,
    svd_spectrum,
)


def _make_panel(
    *,
    directions: list[np.ndarray],
    n_per: int,
    rows_per: int,
    noise: float,
    seed: int,
    mag: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    d = int(directions[0].size)
    H, y, soc = [], [], []
    for s, v in enumerate(directions):
        v = v / np.linalg.norm(v)
        for _rep in range(n_per):
            name = f"soc_{s:02d}"
            for i in range(rows_per):
                yi = 1 if i < rows_per // 2 else 0
                h = (2 * yi - 1) * mag * v + noise * rng.normal(size=d)
                H.append(h)
                y.append(yi)
                soc.append(name)
    return np.stack(H), np.array(y, dtype=int), np.array(soc)


def test_stratum_diffs_drops_unanimous() -> None:
    H = np.zeros((6, 4))
    H[:3, 0] = 1.0
    y = np.array([1, 1, 1, 0, 0, 1])
    labels = np.array(["A", "A", "A", "B", "B", "B"])
    mask = np.ones(6, dtype=bool)
    diff = stratum_diffs(H, y, labels, mask, min_per_class=1)
    assert diff.n_strata == 1, diff.dropped
    assert diff.names == ["B"]
    assert diff.n_dropped == 1


def test_one_direction_spectrum() -> None:
    d = 48
    v = np.zeros(d)
    v[0] = 1.0
    dirs = [v] * 12
    H, y, soc = _make_panel(directions=dirs, n_per=1, rows_per=20, noise=0.03, seed=0)
    train = np.ones(len(y), dtype=bool)
    diff = stratum_diffs(H, y, soc, train, min_per_class=2)
    assert diff.n_strata == 12
    vg = mean_diff(H, y)
    un = svd_spectrum(diff.D, center=False, row_unit=False)
    assert abs(cosine(un.V[:, 0], vg)) > 0.98
    assert un.explained[0] > 0.85
    dummy = np.array(["mf"] * len(y))
    rep = layer_report(
        H,
        y,
        train,
        soc=soc,
        context_order=dummy,
        position_variant=dummy,
        v_g=vg,
        n_null=16,
        seed=0,
        min_per_class=2,
        k_max=4,
    )
    assert "error" not in rep
    assert abs(rep["cos_uncent_pc1_vg"]) > 0.98
    assert rep["n_above_null"] == 0, rep["n_above_null"]
    assert rep["interpret"]["verdict"] == "one_direction"
    W = rep["W"]
    assert cosine(W[:, 0], vg) > 0.99


def test_two_direction_spectrum() -> None:
    d = 48
    v0, v1 = np.zeros(d), np.zeros(d)
    v0[0] = 1.0
    v1[1] = 1.0
    dirs = [v0 if i % 2 == 0 else v1 for i in range(12)]
    H, y, soc = _make_panel(directions=dirs, n_per=1, rows_per=24, noise=0.03, seed=1)
    train = np.ones(len(y), dtype=bool)
    vg = mean_diff(H, y)
    dummy = np.array(["mf"] * len(y))
    rep = layer_report(
        H,
        y,
        train,
        soc=soc,
        context_order=dummy,
        position_variant=dummy,
        v_g=vg,
        n_null=16,
        seed=1,
        min_per_class=2,
        k_max=4,
    )
    assert "error" not in rep
    assert rep["uncentered"]["explained"][0] < 0.75
    assert rep["n_above_null"] >= 1, (rep["n_above_null"], rep["unit_centered"]["explained"][:4])
    assert rep["interpret"]["verdict"] == "low_rank_subspace"
    W = build_steering_basis(vg, np.asarray(rep["W"])[:, 1:], k_max=2)
    assert abs(cosine(W[:, 0], vg)) > 0.99
    # extra column should not be v_G
    assert abs(cosine(W[:, 1], vg)) < 0.25


def test_n_above_null_is_prefix() -> None:
    obs = np.array([0.4, 0.2, 0.05, 0.3])
    p95 = [0.2, 0.15, 0.12, 0.1]
    assert n_above_null(obs, p95) == 2


def test_interpret_underpowered() -> None:
    out = interpret(
        cos_uncent_pc1_vg=0.99,
        uncent_pc1_share=0.9,
        n_above_unit_centered=0,
        unit_pr=3.0,
        unit_pr_null_p95=4.0,
        n_strata=3,
    )
    assert out["verdict"] == "underpowered"


def main() -> int:
    tests = [
        test_stratum_diffs_drops_unanimous,
        test_one_direction_spectrum,
        test_two_direction_spectrum,
        test_n_above_null_is_prefix,
        test_interpret_underpowered,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"OK  {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
