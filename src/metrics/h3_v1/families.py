"""FDR families for H3 v1."""

from __future__ import annotations


def h3_v1_fdr_families(_df, test_ids: list[str], results_by_id: dict | None = None) -> dict[str, list[str]]:
    """Group test_ids by family_name registered on TestResult."""
    if results_by_id is None:
        return {}
    ids = set(test_ids)
    out: dict[str, list[str]] = {}
    for tid in test_ids:
        if tid not in ids:
            continue
        r = results_by_id.get(tid)
        if r is None or not r.family_name:
            continue
        out.setdefault(r.family_name, []).append(tid)
    return out
