"""Benjamini–Hochberg FDR correction."""

from __future__ import annotations

import numpy as np

from src.metrics.stats_tests import TestResult


def p_for_bh(r: TestResult) -> float | None:
    """
    p-value for BH within a family.

    H1 choice row tests carry cluster t-test p in extra.cluster_choice; prob and
    cluster-level tests already store cluster p in p_raw.
    """
    cc = (r.extra or {}).get("cluster_choice") or {}
    cp = cc.get("cluster_p")
    if cp is not None:
        try:
            p = float(cp)
            if np.isfinite(p):
                return p
        except (TypeError, ValueError):
            pass
    return r.p_raw


def benjamini_hochberg(results: list[TestResult], q: float = 0.05) -> None:
    """Attach q_value and rejected_fdr in place (cluster-aware for H1 choice)."""
    indexed: list[tuple[int, float]] = []
    for i, r in enumerate(results):
        p = p_for_bh(r)
        if p is not None and np.isfinite(p):
            indexed.append((i, p))
            r.extra["p_fdr"] = p
    if not indexed:
        return
    m = len(indexed)
    order = sorted(indexed, key=lambda t: t[1])
    p_sorted = [p for _, p in order]

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


def apply_fdr_families(
    all_results: list[TestResult],
    families: dict[str, list[str]],
    q: float,
    *,
    exclude_test_ids: set[str] | None = None,
) -> None:
    by_id = {r.test_id: r for r in all_results}
    skip = exclude_test_ids or set()
    for family_name, test_ids in families.items():
        group = [
            by_id[tid]
            for tid in test_ids
            if tid in by_id
            and tid not in skip
            and p_for_bh(by_id[tid]) is not None
        ]
        for r in group:
            r.family_name = family_name
        benjamini_hochberg(group, q=q)
