"""Pure-numpy helpers for paired XY-control vectors (no torch)."""

from __future__ import annotations

import numpy as np


def family_split3(
    family_ids: list[int],
    *,
    seed: int,
    train_frac: float,
    val_frac: float,
) -> tuple[set[int], set[int], set[int]]:
    uniq = sorted(set(int(x) for x in family_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    if n == 1:
        return set(uniq), set(), set()
    if n == 2:
        return {uniq[0]}, {uniq[1]}, set()
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_train = min(max(1, n_train), n - 2)
    n_val = min(max(1, n_val), n - n_train - 1)
    train = set(uniq[:n_train])
    val = set(uniq[n_train : n_train + n_val])
    test = set(uniq[n_train + n_val :])
    return train, val, test


def family_split3_by_soc(
    items: list[dict],
    *,
    seed: int,
    train_frac: float,
    val_frac: float,
    soc_key: str = "soc_major_title",
    id_key: str = "scenario_family_id",
) -> tuple[set[int], set[int], set[int]]:
    """70/15/15 independently inside each SOC, then union the ids.

    Stops one large domain from flooding a global shuffle. Same seed as
    per-SOC so a given SOC's family ids match across pipelines.
    """
    by_soc: dict[str, list[int]] = {}
    for it in items:
        title = str(it.get(soc_key) or "UNKNOWN")
        by_soc.setdefault(title, []).append(int(it[id_key]))
    train: set[int] = set()
    val: set[int] = set()
    test: set[int] = set()
    for title in sorted(by_soc):
        t, v, te = family_split3(
            by_soc[title], seed=seed, train_frac=train_frac, val_frac=val_frac
        )
        train |= t
        val |= v
        test |= te
    if train & val or train & test or val & test:
        raise ValueError("stratified split produced overlapping family ids")
    return train, val, test


def fit_v_raw(D: np.ndarray) -> tuple[np.ndarray, float]:
    """v_raw = mean(d_i), then unit-norm. Returns (v_hat, ||v_raw||)."""
    if D.ndim != 2 or D.shape[0] == 0:
        raise ValueError(f"D must be N×d, got {D.shape}")
    v = D.mean(axis=0).astype(np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("v_raw is ~0 — paired contrast collapsed")
    return (v / n).astype(np.float64), n
