"""Benjamini–Hochberg FDR correction."""

from __future__ import annotations

import numpy as np

from src.metrics.stats_tests import TestResult


def benjamini_hochberg(results: list[TestResult], q: float = 0.05) -> None:
    """Attach q_value and rejected_fdr in place."""
    indexed = [(i, r) for i, r in enumerate(results) if r.p_raw is not None and np.isfinite(r.p_raw)]
    if not indexed:
        return
    m = len(indexed)
    order = sorted(indexed, key=lambda t: results[t[0]].p_raw)  # type: ignore[arg-type]
    p_sorted = [results[i].p_raw for i, _ in order]

    # q-values (storey-style step-up)
    qvals = [1.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        p = p_sorted[rank - 1]
        val = min(prev, p * m / rank)
        qvals[rank - 1] = min(val, 1.0)
        prev = val

    # rejection
    k = 0
    for rank in range(1, m + 1):
        if p_sorted[rank - 1] <= (rank / m) * q:
            k = rank
    for j, (idx, _) in enumerate(order):
        results[idx].q_value = qvals[j]
        results[idx].rejected_fdr = j < k


def apply_fdr_families(all_results: list[TestResult], families: dict[str, list[str]], q: float) -> None:
    by_id = {r.test_id: r for r in all_results}
    for family_name, test_ids in families.items():
        group = [by_id[tid] for tid in test_ids if tid in by_id and by_id[tid].p_raw is not None]
        for r in group:
            r.family_name = family_name
        benjamini_hochberg(group, q=q)
