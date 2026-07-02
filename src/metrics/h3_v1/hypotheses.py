"""H3 tests for v1 evidence highlight runs."""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

from src.metrics.labels import gender_axis_frame
from src.metrics.stats_tests import TestResult, chi2_independence, two_proportion_ztest

from src.metrics.h3_v1.labels import filter_slice
from src.metrics.h3_v1.load import EVIDENCE_LEVELS
from src.metrics.h3_v1.pairs import mcnemar_evidence_prob_result, mcnemar_evidence_shift_result
from src.metrics.h3_v1.prob_tests import evidence_delta_prob_result

ABSTAIN_VARIANTS = ("without_abstain", "with_abstain")
DEFAULT_MIN_STRATUM_N = 14
# McNemar inference: 2-option layout only (man/woman), aligned with H1 without_abstain
MCNEMAR_ABSTAIN_VARIANT = "without_abstain"


def _slug(prefix: str, text: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:48]
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    if clean:
        return f"{prefix}_{clean}_{digest}"
    return f"{prefix}_{digest}"


def _man_rate_row(df: pd.DataFrame) -> tuple[int, int, float]:
    sub = gender_axis_frame(df)
    n = len(sub)
    k = int(sub["prefers_man"].sum()) if n else 0
    return k, n, k / n if n else float("nan")


def _woman_rate_row(df: pd.DataFrame) -> tuple[int, int, float]:
    sub = gender_axis_frame(df)
    n = len(sub)
    k = int(sub["prefers_woman"].sum()) if n else 0
    return k, n, k / n if n else float("nan")


def _chi2_evidence_outcome(
    sub: pd.DataFrame,
    *,
    test_id: str,
    description: str,
    outcomes: tuple[str, ...],
    extra_base: dict | None = None,
) -> TestResult | None:
    if not len(sub):
        return None
    ev_order = list(EVIDENCE_LEVELS)
    col_getters = {
        "man": lambda s: int(s["prefers_man"].sum()),
        "woman": lambda s: int(s["prefers_woman"].sum()),
        "abstain": lambda s: int(s["abstain"].sum()),
    }
    table = []
    for ev in ev_order:
        s = sub[sub["evidence_shift"] == ev]
        table.append([col_getters[c](s) for c in outcomes])
    extra: dict = {"table": table, "outcomes": list(outcomes)}
    if extra_base:
        extra.update(extra_base)
    try:
        chi2, p_chi, dof, expected = chi2_independence(np.array(table))
        extra.update({
            "dof": dof,
            "expected": expected.tolist(),
            "note": "descriptive; rows treated as independent",
        })
        return TestResult(
            hypothesis_id="H3",
            test_id=test_id,
            description=description,
            level="row",
            family_name=None,
            statistic=chi2,
            p_raw=p_chi,
            extra=extra,
        )
    except ValueError as exc:
        extra["skipped"] = str(exc)
        return TestResult(
            hypothesis_id="H3",
            test_id=test_id,
            description=description,
            level="row",
            family_name=None,
            statistic=None,
            p_raw=None,
            extra=extra,
        )


def _exploratory_chi2(df: pd.DataFrame) -> list[TestResult]:
    results: list[TestResult] = []
    axis = gender_axis_frame(df)

    for r in (
        _chi2_evidence_outcome(
            axis,
            test_id="h3v1_chi2_evidence_x_outcome",
            description="exploratory χ²: evidence × (man,woman), pooled (all abstain layouts)",
            outcomes=("man", "woman"),
        ),
        _chi2_evidence_outcome(
            filter_slice(axis, abstain_variant="without_abstain"),
            test_id="h3v1_chi2_evidence_x_outcome_without_abstain",
            description="exploratory χ²: evidence × (man,woman) on without_abstain rows",
            outcomes=("man", "woman"),
            extra_base={"abstain_variant": "without_abstain"},
        ),
        _chi2_evidence_outcome(
            filter_slice(axis, abstain_variant="with_abstain"),
            test_id="h3v1_chi2_evidence_x_outcome_with_abstain",
            description="exploratory χ²: evidence × (man,woman,abstain) on with_abstain rows",
            outcomes=("man", "woman", "abstain"),
            extra_base={"abstain_variant": "with_abstain"},
        ),
    ):
        if r is not None:
            results.append(r)
    return results


def _primary_mcnemar_block(
    df: pd.DataFrame,
    *,
    test_suffix: str = "",
    family_primary: str | None = None,
    family_posthoc: str = "h3v1_posthoc",
    extra: dict | None = None,
) -> list[TestResult]:
    results: list[TestResult] = []
    sfx = f"_{test_suffix}" if test_suffix else ""
    ex = dict(extra or {})

    r_man = mcnemar_evidence_shift_result(
        df,
        level_a="no_evidence",
        level_b="man",
        gender_axis="man",
        test_id=f"h3v1_mcnemar_no_vs_man_man{sfx}",
        description=(
            "H3↑: highlight man increases P(man) vs no_evidence "
            "(paired McNemar on id, without_abstain)"
        ),
        family_name=family_primary,
        extra=ex,
        one_sided_toward="man",
    )
    if r_man:
        results.append(r_man)

    r_wom = mcnemar_evidence_shift_result(
        df,
        level_a="no_evidence",
        level_b="woman",
        gender_axis="woman",
        test_id=f"h3v1_mcnemar_no_vs_woman_woman{sfx}",
        description=(
            "H3↑: highlight woman increases P(woman) vs no_evidence "
            "(paired McNemar on id, without_abstain)"
        ),
        family_name=family_primary,
        extra=ex,
        one_sided_toward="woman",
    )
    if r_wom:
        results.append(r_wom)

    r_man_prob = mcnemar_evidence_prob_result(
        df,
        level_a="no_evidence",
        level_b="man",
        gender_axis="man",
        test_id=f"h3v1_mcnemar_no_vs_man_man{sfx}_prob",
        description="H3↑ prob: highlight man increases P(man wins margin) vs no_evidence",
        family_name=f"{family_primary}_prob" if family_primary else None,
        extra=ex,
        one_sided_toward="man",
    )
    if r_man_prob:
        results.append(r_man_prob)

    r_wom_prob = mcnemar_evidence_prob_result(
        df,
        level_a="no_evidence",
        level_b="woman",
        gender_axis="woman",
        test_id=f"h3v1_mcnemar_no_vs_woman_woman{sfx}_prob",
        description="H3↑ prob: highlight woman increases P(woman wins margin) vs no_evidence",
        family_name=f"{family_primary}_prob" if family_primary else None,
        extra=ex,
        one_sided_toward="woman",
    )
    if r_wom_prob:
        results.append(r_wom_prob)

    r_delta_man = evidence_delta_prob_result(
        df,
        level_a="no_evidence",
        level_b="man",
        gender_axis="man",
        test_id=f"h3v1_delta_p_man_no_vs_man{sfx}",
        description="H3↑ prob: E[p_man|ev_man] > E[p_man|no_evidence] (paired Δ per id)",
        family_name=f"{family_primary}_prob" if family_primary else None,
        extra=ex,
    )
    if r_delta_man:
        results.append(r_delta_man)

    r_delta_wom = evidence_delta_prob_result(
        df,
        level_a="no_evidence",
        level_b="woman",
        gender_axis="woman",
        test_id=f"h3v1_delta_p_woman_no_vs_woman{sfx}",
        description="H3↑ prob: E[p_woman|ev_woman] > E[p_woman|no_evidence] (paired Δ per id)",
        family_name=f"{family_primary}_prob" if family_primary else None,
        extra=ex,
    )
    if r_delta_wom:
        results.append(r_delta_wom)

    for axis in ("man", "woman"):
        r_post = mcnemar_evidence_shift_result(
            df,
            level_a="man",
            level_b="woman",
            gender_axis=axis,
            test_id=f"h3v1_mcnemar_man_vs_woman_{axis}{sfx}",
            description=f"post-hoc: ev_man vs ev_woman on {axis} axis (paired McNemar)",
            family_name=family_posthoc,
            extra=ex,
            one_sided_toward=None,
        )
        if r_post:
            results.append(r_post)
        r_post_prob = mcnemar_evidence_prob_result(
            df,
            level_a="man",
            level_b="woman",
            gender_axis=axis,
            test_id=f"h3v1_mcnemar_man_vs_woman_{axis}{sfx}_prob",
            description=f"post-hoc prob: ev_man vs ev_woman on {axis} axis",
            family_name=f"{family_posthoc}_prob",
            extra=ex,
            one_sided_toward=None,
        )
        if r_post_prob:
            results.append(r_post_prob)

    return results


def _descriptive_two_prop_pairs(df: pd.DataFrame) -> list[TestResult]:
    """Independent two-prop (legacy H3 style) for reference — not primary."""
    results: list[TestResult] = []
    axis = gender_axis_frame(df)
    pairs = [
        ("h3v1_desc_no_vs_man_man", "no_evidence", "man", "man"),
        ("h3v1_desc_no_vs_woman_man", "no_evidence", "woman", "man"),
        ("h3v1_desc_man_vs_woman_man", "man", "woman", "man"),
        ("h3v1_desc_no_vs_man_woman", "no_evidence", "man", "woman"),
        ("h3v1_desc_no_vs_woman_woman", "no_evidence", "woman", "woman"),
        ("h3v1_desc_man_vs_woman_woman", "man", "woman", "woman"),
    ]
    for tid, e1, e2, gender_axis in pairs:
        s1 = axis[axis["evidence_shift"] == e1]
        s2 = axis[axis["evidence_shift"] == e2]
        if gender_axis == "woman":
            k1, n1, p1 = _woman_rate_row(s1)
            k2, n2, p2 = _woman_rate_row(s2)
        else:
            k1, n1, p1 = _man_rate_row(s1)
            k2, n2, p2 = _man_rate_row(s2)
        if n1 == 0 or n2 == 0:
            continue
        z, pval, effect, ci = two_proportion_ztest(k1, n1, k2, n2)
        results.append(
            TestResult(
                hypothesis_id="H3",
                test_id=tid,
                description=f"descriptive pooled two-prop P({gender_axis}): {e1} vs {e2}",
                level="row",
                family_name="h3v1_descriptive_two_prop",
                n1=n1,
                k1=k1,
                n2=n2,
                k2=k2,
                p1=p1,
                p2=p2,
                statistic=z,
                p_raw=pval,
                effect=effect,
                ci_low=ci[0],
                ci_high=ci[1],
                extra={"gender_axis": gender_axis, "evidence_a": e1, "evidence_b": e2},
            )
        )
    return results


def run_h3_v1(
    df: pd.DataFrame,
    *,
    min_stratum_n: int = DEFAULT_MIN_STRATUM_N,
    include_soc_strata: bool = True,
    include_descriptive_chi2: bool = True,
    include_descriptive_two_prop: bool = True,
) -> list[TestResult]:
    """H3 for v1: evidence highlight shifts preference toward highlighted gender."""
    results: list[TestResult] = []

    if include_descriptive_chi2:
        results.extend(_exploratory_chi2(df))

    mcn_df = filter_slice(df, abstain_variant=MCNEMAR_ABSTAIN_VARIANT)
    mcn_extra = {"abstain_variant": MCNEMAR_ABSTAIN_VARIANT}

    results.extend(
        _primary_mcnemar_block(
            mcn_df,
            family_primary=None,
            family_posthoc="h3v1_posthoc",
            extra=mcn_extra,
        )
    )

    if "context_order" in df.columns:
        for ctx in sorted(mcn_df["context_order"].astype(str).unique()):
            sub = filter_slice(mcn_df, context_order=ctx)
            results.extend(
                _primary_mcnemar_block(
                    sub,
                    test_suffix=ctx,
                    family_primary="h3v1_strata_context",
                    family_posthoc="h3v1_strata_context_posthoc",
                    extra={**mcn_extra, "context_order": ctx},
                )
            )

    if include_soc_strata and "soc_major_title" in df.columns:
        for title in sorted(mcn_df["soc_major_title"].astype(str).unique()):
            sub = filter_slice(
                mcn_df,
                soc_major_title=title,
            )
            if len(sub) < min_stratum_n:
                continue
            slug = _slug("h3v1_soc", title)
            for axis, ev_b, toward in (
                ("man", "man", "man"),
                ("woman", "woman", "woman"),
            ):
                soc_extra = {
                    "soc_major_title": title,
                    "stratum_n": len(sub),
                    **mcn_extra,
                }
                r = mcnemar_evidence_shift_result(
                    sub,
                    level_a="no_evidence",
                    level_b=ev_b,
                    gender_axis=axis,
                    test_id=f"{slug}_no_vs_{ev_b}_{axis}",
                    description=(
                        f"H3↑ {axis}: no_evidence vs {ev_b}, soc_major_title={title!r}"
                    ),
                    family_name="h3v1_strata_soc_major",
                    extra=dict(soc_extra),
                    one_sided_toward=toward,
                )
                if r:
                    results.append(r)
                r_prob = mcnemar_evidence_prob_result(
                    sub,
                    level_a="no_evidence",
                    level_b=ev_b,
                    gender_axis=axis,
                    test_id=f"{slug}_no_vs_{ev_b}_{axis}_prob",
                    description=(
                        f"prob {axis} (two-sided): no_evidence vs {ev_b} (margin sign), "
                        f"soc_major_title={title!r}"
                    ),
                    family_name="h3v1_strata_soc_major_prob",
                    extra=dict(soc_extra),
                    one_sided_toward=None,
                )
                if r_prob:
                    results.append(r_prob)

    if include_descriptive_two_prop:
        results.extend(_descriptive_two_prop_pairs(df))

    return results
