"""FDR families for H1 v1 strata."""

from __future__ import annotations

H1V1_PRIMARY_TEST_PREFIX = "h1v1_primary_"


def is_h1v1_primary_test(test_id: str) -> bool:
    """Pre-specified confirmatory tests — outside BH families."""
    return test_id.startswith(H1V1_PRIMARY_TEST_PREFIX)


def h1_v1_fdr_families(df, test_ids: list[str]) -> dict[str, list[str]]:
    """Group registered test_ids into BH-FDR families (primary excluded)."""
    ids = {tid for tid in test_ids if not is_h1v1_primary_test(tid)}
    test_ids = [tid for tid in test_ids if tid in ids]
    out: dict[str, list[str]] = {}

    layout_mw = [
        tid
        for tid in test_ids
        if tid.startswith("h1v1_layout_") and tid.endswith("_man_vs_woman")
    ]
    if layout_mw:
        out["h1v1_strata_layout_mw"] = [tid for tid in layout_mw if tid in ids]

    layout_prob = [tid for tid in test_ids if tid.startswith("h1v1_layout_") and tid.endswith("_margin_prob")]
    if layout_prob:
        out["h1v1_strata_layout_margin_prob"] = [tid for tid in layout_prob if tid in ids]

    pos_mcn = [
        tid for tid in test_ids
        if tid.startswith("h1v1_mcnemar_pos_p0_p1") and not tid.endswith("_prob")
    ]
    if pos_mcn:
        out["h1v1_strata_position_mcnemar"] = [tid for tid in pos_mcn if tid in ids]

    pos_mcn_prob = [tid for tid in test_ids if tid.startswith("h1v1_mcnemar_pos_p0_p1") and tid.endswith("_prob")]
    if pos_mcn_prob:
        out["h1v1_strata_position_mcnemar_prob"] = [tid for tid in pos_mcn_prob if tid in ids]

    ctx_mcn = [
        tid for tid in test_ids
        if tid.startswith("h1v1_mcnemar_ctx_") and not tid.endswith("_prob")
    ]
    if ctx_mcn:
        out["h1v1_strata_context_mcnemar"] = [tid for tid in ctx_mcn if tid in ids]

    ctx_mcn_prob = [
        tid for tid in test_ids
        if tid.startswith("h1v1_mcnemar_ctx_") and tid.endswith("_prob")
    ]
    if ctx_mcn_prob:
        out["h1v1_strata_context_mcnemar_prob"] = [tid for tid in ctx_mcn_prob if tid in ids]

    abst = [tid for tid in test_ids if tid.startswith("h1v1_abst_") and tid.endswith("_man_vs_woman")]
    if abst:
        out["h1v1_strata_abstain"] = [tid for tid in abst if tid in ids]

    abst_prob = [
        tid for tid in test_ids
        if tid.startswith("h1v1_abst_") and tid.endswith("_margin_prob")
    ]
    if abst_prob:
        out["h1v1_strata_abstain_prob"] = [tid for tid in abst_prob if tid in ids]

    soc = [tid for tid in test_ids if tid.startswith("h1v1_soc_") and tid.endswith("_man_vs_woman")]
    if soc:
        out["h1v1_strata_soc_major"] = [tid for tid in soc if tid in ids]

    soc_prob = [
        tid for tid in test_ids
        if tid.startswith("h1v1_soc_") and tid.endswith("_margin_prob")
    ]
    if soc_prob:
        out["h1v1_strata_soc_major_prob"] = [tid for tid in soc_prob if tid in ids]

    onet = [tid for tid in test_ids if tid.startswith("h1v1_onet_") and tid.endswith("_man_vs_woman")]
    if onet:
        out["h1v1_strata_onet_action"] = [tid for tid in onet if tid in ids]

    onet_prob = [
        tid for tid in test_ids
        if tid.startswith("h1v1_onet_") and tid.endswith("_margin_prob")
    ]
    if onet_prob:
        out["h1v1_strata_onet_action_prob"] = [tid for tid in onet_prob if tid in ids]

    return out
