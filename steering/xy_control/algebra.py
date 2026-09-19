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


def fit_v_raw(D: np.ndarray) -> tuple[np.ndarray, float]:
    """v_raw = mean(d_i), then unit-norm. Returns (v_hat, ||v_raw||)."""
    if D.ndim != 2 or D.shape[0] == 0:
        raise ValueError(f"D must be N×d, got {D.shape}")
    v = D.mean(axis=0).astype(np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("v_raw is ~0 — paired contrast collapsed")
    return (v / n).astype(np.float64), n
