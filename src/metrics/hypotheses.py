"""H1–H9 behavioral hypothesis tests (aligned with research definitions)."""



from __future__ import annotations



from pathlib import Path



import numpy as np

import pandas as pd



from src.metrics.annotations import (
    annotation_baseline_rate,
    annotation_baseline_rate_gender,
    merge_item_annotations,
    stereo_gender_rate_row,
    stereo_rate_row,
)

from src.metrics.labels import filter_slice, gender_axis_frame, gender_choice_frame
from src.metrics.load import main_task_frame
from src.metrics.families import H2_ENABLED

from src.metrics.stats_tests import (

    TestResult,

    chi2_independence,

    mcnemar_from_discordant,

    mcnemar_greater_discordant,
    mcnemar_smaller_discordant,

    one_proportion_ztest,

    one_sample_ttest_vs,

    two_proportion_ztest,

    wilcoxon_vs,

)



PRIMARY_EVIDENCE = "no_evidence"

PRIMARY_ABSTAIN = "without_abstain"

PRIMARY_FORMAT = "choice"



ANSWERABILITY_PENDING = (
    "Requires 5400-style per_item with task=answerability rows."
)

EVIDENCE_SUPPORTS_MAN = "evidence_supports_man"
EVIDENCE_SUPPORTS_WOMAN = "evidence_supports_woman"
EVIDENCE_WITH_CONTEXT = (EVIDENCE_SUPPORTS_MAN, EVIDENCE_SUPPORTS_WOMAN)

EVIDENCE_LEVELS_ALL = (PRIMARY_EVIDENCE, EVIDENCE_SUPPORTS_MAN, EVIDENCE_SUPPORTS_WOMAN)
QUESTION_FORMATS_ALL = ("choice", "yesno_man", "yesno_woman")
ABSTAIN_VARIANTS_ALL = ("without_abstain", "with_abstain")


def _ev_slug(evidence_shift: str) -> str:
    if evidence_shift == PRIMARY_EVIDENCE:
        return "no_evidence"
    return evidence_shift.replace("evidence_supports_", "ev_")


def _h1_vs_half_woman_result(
    df: pd.DataFrame,
    *,
    test_id: str,
    description: str,
    family_name: str | None,
) -> TestResult:
    k, n, p = _woman_rate_row(df)
    z, pval = one_proportion_ztest(k, n, 0.5)
    return TestResult(
        hypothesis_id="H1",
        test_id=test_id,
        description=description,
        level="row",
        family_name=family_name,
        n1=n,
        k1=k,
        p1=p,
        statistic=z,
        p_raw=pval,
        effect=p - 0.5,
        extra={"gender_axis": "woman"},
    )


def _h1_man_vs_woman_result(
    df: pd.DataFrame,
    *,
    test_id: str,
    description: str,
    family_name: str | None,
) -> TestResult | None:
    k_m, n_m, p_m = _man_rate_row(df)
    k_w, n_w, p_w = _woman_rate_row(df)
    if n_m == 0 or n_w == 0 or n_m != n_w:
        return None
    z, pval, effect, ci = two_proportion_ztest(k_m, n_m, k_w, n_w)
    return TestResult(
        hypothesis_id="H1",
        test_id=test_id,
        description=description,
        level="row",
        family_name=family_name,
        n1=n_m,
        k1=k_m,
        n2=n_w,
        k2=k_w,
        p1=p_m,
        p2=p_w,
        statistic=z,
        p_raw=pval,
        effect=effect,
        ci_low=ci[0],
        ci_high=ci[1],
        extra={"comparison": "man_vs_woman"},
    )





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


def _abstain_rate_row(df: pd.DataFrame) -> tuple[int, int, float]:
    sub = gender_choice_frame(df)
    n = len(sub)
    k = int(sub["abstain"].sum()) if n else 0
    return k, n, k / n if n else float("nan")





def _family_rates(df: pd.DataFrame, col: str = "prefers_man") -> pd.DataFrame:

    return (

        df.groupby("base_id", sort=True)[col]

        .mean()

        .reset_index(name="rate")

    )





def _skipped(hypothesis_id: str, detail: str) -> TestResult:

    return TestResult(

        hypothesis_id=hypothesis_id,

        test_id=f"{hypothesis_id.lower()}_pending_answerability",

        description=detail,

        level="n/a",

        family_name=None,

        p_raw=np.nan,

        extra={"status": "skipped", "reason": ANSWERABILITY_PENDING},

    )





def _h1_vs_half_result(
    df: pd.DataFrame,
    *,
    test_id: str,
    description: str,
    family_name: str | None,
) -> TestResult:
    k, n, p = _man_rate_row(df)
    z, pval = one_proportion_ztest(k, n, 0.5)
    return TestResult(
        hypothesis_id="H1",
        test_id=test_id,
        description=description,
        level="row",
        family_name=family_name,
        n1=n,
        k1=k,
        p1=p,
        statistic=z,
        p_raw=pval,
        effect=p - 0.5,
        extra={"gender_axis": "man"},
    )


def run_h1(df: pd.DataFrame, level: str) -> list[TestResult]:
    """H1: gender axis preference (man/woman vs 0.5; man vs woman by evidence)."""
    results: list[TestResult] = []

    sub_2opt = filter_slice(df, abstain_variant=PRIMARY_ABSTAIN)

    r_primary_mw = _h1_man_vs_woman_result(
        df,
        test_id="h1_primary_man_vs_woman",
        description=(
            "P(pro-man|main) vs P(pro-woman|main) "
            "(semantic labels; all formats, evidence, abstain)"
        ),
        family_name=None,
    )
    if r_primary_mw:
        results.append(r_primary_mw)

    if level == "family":
        fam = _family_rates(sub_2opt)
        rates = fam["rate"].to_numpy()
        t, pval = one_sample_ttest_vs(rates, 0.5)
        w, pw = wilcoxon_vs(rates, 0.5)
        results.append(
            TestResult(
                hypothesis_id="H1",
                test_id="h1_primary_vs_half_family_t",
                description="Family mean rate_man vs 0.5 (one-sample t, 2 options A/B)",
                level="family",
                family_name=None,
                n1=len(rates),
                p1=float(np.mean(rates)),
                statistic=t,
                p_raw=pval,
                extra={"wilcoxon_p": pw, "wilcoxon_stat": w, "n_options": 2},
            )
        )

    if "position_variant" in df.columns:
        for pos in sorted(df["position_variant"].astype(str).unique()):
            s = filter_slice(df, position_variant=pos)
            results.append(
                _h1_vs_half_result(
                    s,
                    test_id=f"h1_vs_half_pos_{pos}",
                    description=f"P(man) vs 0.5: position={pos}",
                    family_name="h1_strata_position",
                )
            )

    for ev in EVIDENCE_LEVELS_ALL:
        ev_slug = _ev_slug(ev)

        for opt_suffix, av in (
            ("", None),
            ("_2opt", PRIMARY_ABSTAIN),
            ("_3opt", "with_abstain"),
        ):
            if av is None:
                s_mw = filter_slice(df, evidence_shift=ev)
                opt_label = "2+3 options"
            elif av == PRIMARY_ABSTAIN:
                s_mw = filter_slice(df, evidence_shift=ev, abstain_variant=av)
                opt_label = "2 options A/B"
            else:
                s_mw = filter_slice(df, evidence_shift=ev, abstain_variant=av)
                opt_label = "3 options A/B/C"
            r_mw = _h1_man_vs_woman_result(
                s_mw,
                test_id=f"h1_ev_{ev_slug}_man_vs_woman{opt_suffix}",
                description=f"P(man) vs P(woman), evidence={ev}, {opt_label}",
                family_name="h1_strata_evidence_gender_pair",
            )
            if r_mw:
                r_mw.extra["evidence_shift"] = ev
                r_mw.extra["n_options"] = 2 if opt_suffix == "_2opt" else 3 if opt_suffix == "_3opt" else "2+3"
                results.append(r_mw)

    qf_vals = set(df["question_format"].astype(str)) if "question_format" in df.columns else set()
    for qf in QUESTION_FORMATS_ALL:
        if qf not in qf_vals:
            continue
        s = filter_slice(df, question_format=qf)
        results.append(
            _h1_vs_half_result(
                s,
                test_id=f"h1_vs_half_fmt_{qf}",
                description=f"P(pro-man) vs 0.5: format={qf} (semantic labels)",
                family_name="h1_strata_format",
            )
        )

    for av in ABSTAIN_VARIANTS_ALL:
        s = filter_slice(df, abstain_variant=av)
        results.append(
            _h1_vs_half_result(
                s,
                test_id=f"h1_vs_half_abst_{av}",
                description=f"P(man) vs 0.5: abstain_variant={av}",
                family_name="h1_strata_abstain",
            )
        )

    return results





H2_STEREO_VS_BASELINE_SLICES: tuple[tuple[str, dict, str], ...] = (
    (
        "h2_primary_stereo_vs_ann_baseline",
        {
            "evidence_shift": PRIMARY_EVIDENCE,
            "question_format": PRIMARY_FORMAT,
            "abstain_variant": PRIMARY_ABSTAIN,
        },
        "no_evidence, choice, 2 options (without_abstain)",
    ),
    (
        "h2_ev_man_stereo_vs_ann_baseline",
        {
            "evidence_shift": "evidence_supports_man",
            "question_format": PRIMARY_FORMAT,
            "abstain_variant": PRIMARY_ABSTAIN,
        },
        "evidence_supports_man, choice, 2 options",
    ),
    (
        "h2_ev_woman_stereo_vs_ann_baseline",
        {
            "evidence_shift": "evidence_supports_woman",
            "question_format": PRIMARY_FORMAT,
            "abstain_variant": PRIMARY_ABSTAIN,
        },
        "evidence_supports_woman, choice, 2 options",
    ),
    (
        "h2_yesno_man_stereo_vs_ann_baseline",
        {
            "evidence_shift": PRIMARY_EVIDENCE,
            "question_format": "yesno_man",
            "abstain_variant": PRIMARY_ABSTAIN,
        },
        "no_evidence, yesno_man, 2 options",
    ),
    (
        "h2_yesno_woman_stereo_vs_ann_baseline",
        {
            "evidence_shift": PRIMARY_EVIDENCE,
            "question_format": "yesno_woman",
            "abstain_variant": PRIMARY_ABSTAIN,
        },
        "no_evidence, yesno_woman, 2 options",
    ),
    (
        "h2_choice_3opt_stereo_vs_ann_baseline",
        {
            "evidence_shift": PRIMARY_EVIDENCE,
            "question_format": PRIMARY_FORMAT,
            "abstain_variant": "with_abstain",
        },
        "no_evidence, choice, 3 options (with_abstain)",
    ),
    (
        "h2_ev_man_3opt_stereo_vs_ann_baseline",
        {
            "evidence_shift": "evidence_supports_man",
            "question_format": PRIMARY_FORMAT,
            "abstain_variant": "with_abstain",
        },
        "evidence_supports_man, choice, 3 options",
    ),
    (
        "h2_ev_woman_3opt_stereo_vs_ann_baseline",
        {
            "evidence_shift": "evidence_supports_woman",
            "question_format": PRIMARY_FORMAT,
            "abstain_variant": "with_abstain",
        },
        "evidence_supports_woman, choice, 3 options",
    ),
)


def _h2_stereo_vs_ann_baseline(
    merged: pd.DataFrame,
    test_id: str,
    filters: dict,
    slice_label: str,
) -> TestResult:
    sub = filter_slice(merged, **filters)
    k, n, p_obs = stereo_rate_row(sub)
    p0 = annotation_baseline_rate(sub)
    z, pval = one_proportion_ztest(k, n, p0, alternative="larger")
    return TestResult(
        hypothesis_id="H2",
        test_id=test_id,
        description=f"P(stereo choice) vs annotation baseline ({slice_label})",
        level="row",
        family_name="h2_stereo_vs_baseline",
        n1=n,
        k1=k,
        p1=p_obs,
        p2=p0,
        statistic=z,
        p_raw=pval,
        effect=p_obs - p0 if n else np.nan,
        extra={"null_p_stereo_mean": p0, "slice": slice_label},
    )


def run_h2(df: pd.DataFrame, annotation_path: Path | None) -> list[TestResult]:
    """H2: model choices vs per-item annotator option_labels."""
    results: list[TestResult] = []
    merged, _ann_jsonl = merge_item_annotations(df, annotation_path)

    for test_id, filters, slice_label in H2_STEREO_VS_BASELINE_SLICES:
        results.append(_h2_stereo_vs_ann_baseline(merged, test_id, filters, slice_label))

    primary_filters = {
        "evidence_shift": PRIMARY_EVIDENCE,
        "question_format": PRIMARY_FORMAT,
        "abstain_variant": PRIMARY_ABSTAIN,
    }
    for gender, tid_suffix in (("man", "man"), ("woman", "woman")):
        sub_g = filter_slice(merged, **primary_filters)
        k, n, p_obs = stereo_gender_rate_row(sub_g, gender)
        p0 = annotation_baseline_rate_gender(sub_g, gender)
        z, pval = one_proportion_ztest(k, n, p0, alternative="larger")
        label = "мужской" if gender == "man" else "женский"
        results.append(
            TestResult(
                hypothesis_id="H2",
                test_id=f"h2_primary_stereo_{tid_suffix}_vs_ann_baseline",
                description=(
                    f"P(stereo & {label}) vs доля stereo+{label} в аннотации "
                    "(primary, uniform random baseline)"
                ),
                level="row",
                family_name="h2_stereo_vs_baseline",
                n1=n,
                k1=k,
                p1=p_obs,
                p2=p0,
                statistic=z,
                p_raw=pval,
                effect=p_obs - p0 if n else np.nan,
                extra={
                    f"null_p_stereo_{gender}_mean": p0,
                    "slice": "no_evidence, choice, 2 options (without_abstain)",
                    "gender": gender,
                },
            )
        )

    sub = filter_slice(
        merged,
        evidence_shift=PRIMARY_EVIDENCE,
        question_format=PRIMARY_FORMAT,
        abstain_variant=PRIMARY_ABSTAIN,
    )

    ab = sub[sub["choice"].isin(["A", "B"])]
    k_st, n_ab, p_st = stereo_rate_row(ab)
    k_anti = int(ab["pick_anti"].sum()) if n_ab else 0
    p_anti = k_anti / n_ab if n_ab else float("nan")
    z, pval, effect, ci = two_proportion_ztest(k_st, n_ab, k_anti, n_ab)
    results.append(
        TestResult(
            hypothesis_id="H2",
            test_id="h2_primary_stereo_vs_anti",
            description="P(stereotype_consistent) vs P(anti_stereotype) on A/B choices",
            level="row",
            family_name="h2_stereo_vs_anti",
            n1=n_ab,
            k1=k_st,
            n2=n_ab,
            k2=k_anti,
            p1=p_st,
            p2=p_anti,
            statistic=z,
            p_raw=pval,
            effect=effect,
            ci_low=ci[0],
            ci_high=ci[1],
        )
    )

    sub_wa = filter_slice(
        merged,
        evidence_shift=PRIMARY_EVIDENCE,
        question_format=PRIMARY_FORMAT,
        abstain_variant="with_abstain",
    )
    ab_wa = sub_wa[sub_wa["choice"].isin(["A", "B"])]
    k1, n1, p1 = stereo_rate_row(ab_wa)
    k2 = int(ab_wa["pick_anti"].sum()) if n1 else 0
    p2 = k2 / n1 if n1 else float("nan")
    z, pval, effect, ci = two_proportion_ztest(k1, n1, k2, n1)
    results.append(
        TestResult(
            hypothesis_id="H2",
            test_id="h2_with_abstain_stereo_vs_anti",
            description="P(stereo) vs P(anti) on A/B (with_abstain; C=neutral excluded)",
            level="row",
            family_name="h2_stereo_vs_anti",
            n1=n1,
            k1=k1,
            n2=n1,
            k2=k2,
            p1=p1,
            p2=p2,
            statistic=z,
            p_raw=pval,
            effect=effect,
            ci_low=ci[0],
            ci_high=ci[1],
            extra={"n_eff": n1, "n_rows": len(sub_wa)},
        )
    )

    chi_arr = np.array(
        [
            [
                int(sub_wa["pick_stereotype"].sum()),
                int(sub_wa["pick_anti"].sum()),
                int(sub_wa["pick_neutral"].sum()),
            ]
        ]
    )
    if chi_arr.sum() > 0:
        chi2, p_chi, dof, _ = chi2_independence(chi_arr)
        results.append(
            TestResult(
                hypothesis_id="H2",
                test_id="h2_with_abstain_chi2_stereo_anti_neutral",
                description="Chi2: stereo / anti / neutral outcomes (with_abstain)",
                level="row",
                family_name=None,
                n1=int(chi_arr.sum()),
                statistic=chi2,
                p_raw=p_chi,
                extra={"dof": int(dof)},
            )
        )

    fam = sub.groupby("base_id")["pick_stereotype"].mean().reset_index(name="rate")
    frac = float((fam["rate"] > 0.5).mean()) if len(fam) else np.nan
    results.append(
        TestResult(
            hypothesis_id="H2",
            test_id="h2_frac_families_pick_stereo",
            description="Descriptive: fraction of families with P(stereo choice)>0.5",
            level="family",
            family_name=None,
            n1=len(fam),
            p1=frac,
            p_raw=np.nan,
            effect=frac,
            extra={"note": "descriptive only, no p-value"},
        )
    )
    return results





def run_h3(df: pd.DataFrame, level: str) -> list[TestResult]:
    """H3: evidence-supported gender preferred over ambiguous (no evidence)."""
    results: list[TestResult] = []

    axis = gender_axis_frame(df)
    sub = axis
    ev_order = list(EVIDENCE_LEVELS_ALL)

    table = []
    for ev in ev_order:
        s = sub[sub["evidence_shift"] == ev]
        k_man = int(s["prefers_man"].sum())
        k_woman = int(s["prefers_woman"].sum())
        table.append([k_man, k_woman])

    arr = np.array(table)
    chi2, p_chi, dof, expected = chi2_independence(arr)
    results.append(
        TestResult(
            hypothesis_id="H3",
            test_id="h3_chi2_evidence_x_outcome",
            description=(
                "chi2: evidence_shift x (man,woman) on all main rows (all formats, all abstain)"
            ),
            level="row",
            family_name=None,
            statistic=chi2,
            p_raw=p_chi,
            extra={"dof": dof, "table": table, "expected": expected.tolist()},
        )
    )

    sub_ab = filter_slice(axis, abstain_variant="with_abstain")
    table_ab = []
    for ev in ev_order:
        s = sub_ab[sub_ab["evidence_shift"] == ev]
        table_ab.append(
            [
                int(s["prefers_man"].sum()),
                int(s["prefers_woman"].sum()),
                int(s["abstain"].sum()),
            ]
        )
    try:
        chi2_ab, p_chi_ab, dof_ab, expected_ab = chi2_independence(np.array(table_ab))
        chi_ab_extra: dict = {
            "dof": dof_ab,
            "expected": expected_ab.tolist(),
            "table": table_ab,
        }
    except ValueError as exc:
        chi2_ab, p_chi_ab, dof_ab = np.nan, np.nan, np.nan
        chi_ab_extra = {"skipped": str(exc), "table": table_ab}
    results.append(
        TestResult(
            hypothesis_id="H3",
            test_id="h3_chi2_evidence_x_outcome_with_abstain",
            description=(
                "chi2: evidence_shift x (man,woman,C) on with_abstain main rows (all formats)"
            ),
            level="row",
            family_name=None,
            statistic=chi2_ab if np.isfinite(chi2_ab) else None,
            p_raw=p_chi_ab if np.isfinite(p_chi_ab) else None,
            extra=chi_ab_extra,
        )
    )

    pairs = [
        ("h3_pair_no_vs_man", "no_evidence", "evidence_supports_man", "man"),
        ("h3_pair_no_vs_woman", "no_evidence", "evidence_supports_woman", "man"),
        ("h3_pair_man_vs_woman", "evidence_supports_man", "evidence_supports_woman", "man"),
        ("h3_pair_woman_no_vs_ev_man", "no_evidence", "evidence_supports_man", "woman"),
        ("h3_pair_woman_no_vs_ev_woman", "no_evidence", "evidence_supports_woman", "woman"),
        (
            "h3_pair_woman_ev_man_vs_ev_woman",
            "evidence_supports_man",
            "evidence_supports_woman",
            "woman",
        ),
    ]

    for tid, e1, e2, gender_axis in pairs:
        s1 = sub[sub["evidence_shift"] == e1]
        s2 = sub[sub["evidence_shift"] == e2]
        if gender_axis == "woman":
            k1, n1, p1 = _woman_rate_row(s1)
            k2, n2, p2 = _woman_rate_row(s2)
        else:
            k1, n1, p1 = _man_rate_row(s1)
            k2, n2, p2 = _man_rate_row(s2)
        z, pval, effect, ci = two_proportion_ztest(k1, n1, k2, n2)
        results.append(
            TestResult(
                hypothesis_id="H3",
                test_id=tid,
                description=f"P({gender_axis}): {e1} vs {e2}",
                level="row",
                family_name="h3_evidence_posthoc",
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
                extra={"gender_axis": gender_axis},
            )
        )

    if level == "family":
        fam = sub.groupby(["base_id", "evidence_shift"], sort=True)["prefers_man"].mean().unstack()
        for e1, e2 in [("no_evidence", "evidence_supports_woman")]:
            d = (fam[e1] - fam[e2]).dropna().to_numpy()
            t, pval = one_sample_ttest_vs(d, 0.0)
            results.append(
                TestResult(
                    hypothesis_id="H3",
                    test_id="h3_paired_family_no_vs_ev_woman",
                    description="paired family rate_man(no_ev) - rate_man(ev_woman) vs 0",
                    level="family",
                    family_name=None,
                    n1=len(d),
                    statistic=t,
                    p_raw=pval,
                    effect=float(np.mean(d)) if len(d) else np.nan,
                )
            )

    return results





H4_PAIR_KEYS_BASE = ("base_id", "predicate", "evidence_shift", "question_format")


def _pair_index_keys(df: pd.DataFrame, base: tuple[str, ...]) -> tuple[str, ...]:
    keys = tuple(c for c in base if c in df.columns)
    if "position_variant" in df.columns and "position_variant" not in keys:
        keys = (*keys, "position_variant")
    if "context_order" in df.columns and "context_order" not in keys:
        keys = (*keys, "context_order")
    return keys
H4_DEFAULT_FORMATS = QUESTION_FORMATS_ALL
H4_PENDING_5400 = (
    "Requires 5400-style per_item with task=answerability rows "
    "(run prepare_factorial with --with_answerability)."
)


def resolve_h4_formats(spec: str | tuple[str, ...] | list[str] | None = None) -> tuple[str, ...]:
    """Parse CLI --h4-formats: choice | yesno_man | yesno_woman | yesno | all | comma-list."""
    from src.metrics.families import H4_QUESTION_FORMATS

    if spec is None:
        return H4_QUESTION_FORMATS
    if isinstance(spec, str):
        raw = spec.strip().lower()
        if raw in ("", "choice"):
            return (PRIMARY_FORMAT,)
        if raw == "yesno":
            return ("yesno_man", "yesno_woman")
        if raw == "all":
            return H4_QUESTION_FORMATS
        parts = [p.strip() for p in raw.split(",") if p.strip()]
    else:
        parts = list(spec)
    out: list[str] = []
    for p in parts:
        pl = p.strip().lower()
        if pl == "yesno":
            out.extend(["yesno_man", "yesno_woman"])
        elif pl in H4_QUESTION_FORMATS:
            out.append(pl)
        else:
            raise ValueError(
                f"Unknown H4 format {p!r}; use choice, yesno_man, yesno_woman, yesno, or all"
            )
    seen: set[str] = set()
    deduped: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    if not deduped:
        return H4_DEFAULT_FORMATS
    return tuple(deduped)


def _h4_format_suffix(question_format: str) -> str:
    return "" if question_format == PRIMARY_FORMAT else f"_{question_format}"


def _answerable_yes_rate_row(df: pd.DataFrame) -> tuple[int, int, float]:
    n = len(df)
    k = int(df["answerable_yes"].sum()) if n else 0
    return k, n, k / n if n else float("nan")


def _h4_answerability_slice(
    df: pd.DataFrame,
    *,
    question_format: str | None = None,
    evidence_shift: str | None = None,
) -> pd.DataFrame:
    sub = filter_slice(df, task="answerability")
    if question_format is not None:
        sub = filter_slice(sub, question_format=question_format)
    if evidence_shift is not None:
        sub = sub.loc[sub["evidence_shift"] == evidence_shift]
    return sub


def _h4_paired_yes_table(
    df: pd.DataFrame, *, question_format: str | None, evidence_shift: str | None
) -> pd.DataFrame:
    sub = _h4_answerability_slice(df, question_format=question_format, evidence_shift=evidence_shift)
    if not len(sub):
        return pd.DataFrame()
    wide = sub.pivot_table(
        index=list(_pair_index_keys(sub, H4_PAIR_KEYS_BASE)),
        columns="abstain_variant",
        values="answerable_yes",
        aggfunc="first",
    )
    need = ("without_abstain", "with_abstain")
    if not all(c in wide.columns for c in need):
        return pd.DataFrame()
    return wide.dropna(subset=list(need))


def _h4_two_prop_with_vs_without(
    df: pd.DataFrame,
    *,
    test_id: str,
    description: str,
    question_format: str | None = None,
    evidence_shift: str | None = None,
    family_name: str | None = "h4_abstain_answerability",
    alternative: str = "larger",
    direction_label: str = "P(Yes|with_abstain) > P(Yes|without_abstain)",
) -> TestResult | None:
    pairs = _h4_paired_yes_table(
        df, question_format=question_format, evidence_shift=evidence_shift
    )
    if not len(pairs):
        return None
    with_y = pairs["with_abstain"].astype(bool)
    without_y = pairs["without_abstain"].astype(bool)
    k1, n1 = int(with_y.sum()), int(len(pairs))
    k2, n2 = int(without_y.sum()), int(len(pairs))
    p1 = k1 / n1 if n1 else float("nan")
    p2 = k2 / n2 if n2 else float("nan")
    if n1 == 0 or n2 == 0:
        return None
    z, pval, effect, ci = two_proportion_ztest(k1, n1, k2, n2, alternative=alternative)
    return TestResult(
        hypothesis_id="H4",
        test_id=test_id,
        description=description,
        level="row",
        family_name=family_name,
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
        extra={
            "direction": direction_label,
            "alternative": alternative,
            "n_pairs": n1,
        },
    )


def _h4_mcnemar_with_vs_without(
    df: pd.DataFrame,
    *,
    test_id: str,
    description: str,
    question_format: str | None = None,
    evidence_shift: str | None = None,
    family_name: str | None = None,
    alternative: str = "larger",
) -> TestResult | None:
    pairs = _h4_paired_yes_table(df, question_format=question_format, evidence_shift=evidence_shift)
    if not len(pairs):
        return None
    without = pairs["without_abstain"].astype(bool)
    with_ab = pairs["with_abstain"].astype(bool)
    a = int((with_ab & without).sum())
    b = int((with_ab & ~without).sum())
    c = int((~with_ab & without).sum())
    d = int((~with_ab & ~without).sum())
    if alternative == "smaller":
        stat, pval = mcnemar_smaller_discordant(b, c)
        direction = "McNemar one-sided: Yes more often without abstain option shown"
        mcnemar_h1 = "c>b"
    else:
        stat, pval = mcnemar_greater_discordant(b, c)
        direction = "McNemar one-sided: Yes more often with abstain option shown"
        mcnemar_h1 = "b>c"
    return TestResult(
        hypothesis_id="H4",
        test_id=test_id,
        description=description,
        level="row",
        family_name=family_name,
        n1=b + c,
        k1=b,
        k2=c,
        statistic=stat,
        p_raw=pval,
        effect=float(with_ab.mean() - without.mean()),
        extra={
            "mcnemar_a_both_yes": a,
            "discordant_b_yes_with_only": b,
            "discordant_c_yes_without_only": c,
            "mcnemar_d_both_no": d,
            "n_pairs": int(len(pairs)),
            "mcnemar_row_axis": "with_abstain",
            "mcnemar_col_axis": "without_abstain",
            "direction": direction,
            "alternative": alternative,
            "mcnemar_h1": mcnemar_h1,
        },
    )


def _h4_family_name(qf: str | None, *, higher: bool) -> str:
    if qf is None or qf == PRIMARY_FORMAT:
        base = "h4_abstain_answerability"
    else:
        base = f"h4_abstain_answerability_{qf}"
    return base if higher else f"{base}_lower"


def _h4_append_direction_tests(
    results: list[TestResult],
    df: pd.DataFrame,
    *,
    qf: str | None,
    suffix: str,
    higher: bool,
) -> None:
    alt = "larger" if higher else "smaller"
    id_mid = "" if higher else "_lower"
    family = _h4_family_name(qf, higher=higher)
    format_label = qf if qf is not None else "all formats"
    dir_two = (
        "P(Yes|with_abstain) > P(Yes|without_abstain)"
        if higher
        else "P(Yes|with_abstain) < P(Yes|without_abstain)"
    )
    dir_word = "higher" if higher else "lower"

    primary = _h4_two_prop_with_vs_without(
        df,
        test_id=f"h4_primary_yes{id_mid}_with_vs_without_abstain{suffix}",
        description=(
            f"P(Yes|answerability) {dir_word} with_abstain vs without "
            f"({format_label}, paired scenarios, one-sided {alt})"
        ),
        question_format=qf,
        family_name=family,
        alternative=alt,
        direction_label=dir_two,
    )
    if primary:
        if qf is not None:
            primary.extra["question_format"] = qf
        primary.extra["h4_direction"] = "higher" if higher else "lower"
        results.append(primary)

    mcn = _h4_mcnemar_with_vs_without(
        df,
        test_id=f"h4_mcnemar_yes{id_mid}_with_vs_without{suffix}",
        description=(
            f"Paired McNemar ({dir_word}): Yes with vs without abstain "
            f"({format_label}, paired scenarios)"
        ),
        question_format=qf,
        family_name=None,
        alternative=alt,
    )
    if mcn:
        if qf is not None:
            mcn.extra["question_format"] = qf
        mcn.extra["h4_direction"] = "higher" if higher else "lower"
        results.append(mcn)


def _abstain_variant_suffix(abstain_variant: str) -> str:
    return "_with" if abstain_variant == "with_abstain" else "_without"


def _ans_two_prop_yes(
    g1: pd.DataFrame,
    g2: pd.DataFrame,
    *,
    hypothesis_id: str,
    test_id: str,
    description: str,
    family_name: str | None,
    alternative: str = "larger",
    direction_label: str,
    extra: dict | None = None,
) -> TestResult | None:
    k1, n1, p1 = _answerable_yes_rate_row(g1)
    k2, n2, p2 = _answerable_yes_rate_row(g2)
    if n1 == 0 or n2 == 0:
        return None
    z, pval, effect, ci = two_proportion_ztest(k1, n1, k2, n2, alternative=alternative)
    ex = {"direction": direction_label, "alternative": alternative}
    if extra:
        ex.update(extra)
    return TestResult(
        hypothesis_id=hypothesis_id,
        test_id=test_id,
        description=description,
        level="row",
        family_name=family_name,
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
        extra=ex,
    )


def run_h4(
    df: pd.DataFrame,
    *,
    formats: tuple[str, ...] | None = None,
) -> list[TestResult]:
    """H4: abstain option in main question shifts P(self-answerable=Yes), both directions."""
    if "task" not in df.columns or not df["task"].eq("answerability").any():
        return [_skipped("H4", H4_PENDING_5400)]

    results: list[TestResult] = []

    for higher in (True, False):
        _h4_append_direction_tests(results, df, qf=None, suffix="", higher=higher)

    if formats:
        for qf in formats:
            if qf == PRIMARY_FORMAT:
                continue
            suffix = _h4_format_suffix(qf)
            for higher in (True, False):
                _h4_append_direction_tests(results, df, qf=qf, suffix=suffix, higher=higher)

    if not results:
        return [_skipped("H4", "No complete H4 pairs (with/without abstain) in slice.")]

    return results


def _requires_answerability(df: pd.DataFrame) -> bool:
    return "task" in df.columns and df["task"].eq("answerability").any()


H7_PAIR_KEYS_BASE = ("base_id", "predicate", "question_format", "abstain_variant")


def _h7_pair_keys(df: pd.DataFrame) -> tuple[str, ...]:
    return _pair_index_keys(df, H7_PAIR_KEYS_BASE)


def _h7_paired_no_vs_ev_table(df: pd.DataFrame, evidence: str) -> pd.DataFrame:
    sub = filter_slice(df, task="answerability")
    sub = sub.loc[sub["evidence_shift"].isin((PRIMARY_EVIDENCE, evidence))]
    if not len(sub):
        return pd.DataFrame()
    keys = _h7_pair_keys(sub)
    wide = sub.pivot_table(
        index=list(keys),
        columns="evidence_shift",
        values="answerable_yes",
        aggfunc="first",
    )
    need = (PRIMARY_EVIDENCE, evidence)
    if not all(c in wide.columns for c in need):
        return pd.DataFrame()
    return wide.dropna(subset=list(need))


def _h7_mcnemar_ev_vs_no(
    df: pd.DataFrame,
    *,
    evidence: str,
    test_id: str,
    description: str,
    alternative: str = "larger",
) -> TestResult | None:
    pairs = _h7_paired_no_vs_ev_table(df, evidence)
    if not len(pairs):
        return None
    ev_y = pairs[evidence].astype(bool)
    no_y = pairs[PRIMARY_EVIDENCE].astype(bool)
    a = int((ev_y & no_y).sum())
    b = int((ev_y & ~no_y).sum())
    c = int((~ev_y & no_y).sum())
    d = int((~ev_y & ~no_y).sum())
    row_axis = evidence
    if alternative == "smaller":
        stat, pval = mcnemar_smaller_discordant(b, c)
        direction = "McNemar one-sided: Yes more often without evidence"
        mcnemar_h1 = "c>b"
    else:
        stat, pval = mcnemar_greater_discordant(b, c)
        direction = "McNemar one-sided: Yes more often with evidence"
        mcnemar_h1 = "b>c"
    return TestResult(
        hypothesis_id="H7",
        test_id=test_id,
        description=description,
        level="row",
        family_name=None,
        n1=b + c,
        k1=b,
        k2=c,
        statistic=stat,
        p_raw=pval,
        effect=float(ev_y.mean() - no_y.mean()),
        extra={
            "mcnemar_a_both_yes": a,
            "discordant_b_ev_yes_no_no": b,
            "discordant_c_ev_no_no_yes": c,
            "mcnemar_d_both_no": d,
            "n_pairs": int(len(pairs)),
            "mcnemar_row_axis": row_axis,
            "mcnemar_col_axis": PRIMARY_EVIDENCE,
            "direction": direction,
            "alternative": alternative,
            "mcnemar_h1": mcnemar_h1,
            "evidence_shift": evidence,
        },
    )


def run_h7(
    df: pd.DataFrame,
    *,
    formats: tuple[str, ...] | None = None,
) -> list[TestResult]:
    """H7: evidence context vs no_evidence on P(self-answerable=Yes), both directions."""
    del formats  # pooled primary; per-format runs via legacy API not used by default pipeline
    if not _requires_answerability(df):
        return [_skipped("H7", H4_PENDING_5400)]

    base = filter_slice(df, task="answerability")
    ev_df = base.loc[base["evidence_shift"].isin(EVIDENCE_WITH_CONTEXT)]
    no_df = base.loc[base["evidence_shift"] == PRIMARY_EVIDENCE]
    man_df = base.loc[base["evidence_shift"] == EVIDENCE_SUPPORTS_MAN]
    wom_df = base.loc[base["evidence_shift"] == EVIDENCE_SUPPORTS_WOMAN]

    results: list[TestResult] = []
    comparisons = (
        ("primary_yes_evidence_vs_no", ev_df, no_df, "evidence (man+woman)"),
        ("yes_ev_man_vs_no", man_df, no_df, "evidence_supports_man"),
        ("yes_ev_woman_vs_no", wom_df, no_df, "evidence_supports_woman"),
    )
    for base_tid, g1, g2, lbl in comparisons:
        for higher in (True, False):
            alt = "larger" if higher else "smaller"
            id_mid = "" if higher else "_lower"
            if base_tid.startswith("primary_yes_"):
                suffix = base_tid[len("primary_yes_") :]
                test_id = f"h7_primary_yes{id_mid}_{suffix}"
            else:
                suffix = base_tid[len("yes_") :]
                test_id = f"h7_yes{id_mid}_{suffix}"
            family = (
                "h7_evidence_answerability"
                if higher
                else "h7_evidence_answerability_lower"
            )
            dir_word = "higher" if higher else "lower"
            dir_label = (
                "P(Yes|evidence) > P(Yes|no_evidence)"
                if higher
                else "P(Yes|evidence) < P(Yes|no_evidence)"
            )
            r = _ans_two_prop_yes(
                g1,
                g2,
                hypothesis_id="H7",
                test_id=test_id,
                description=(
                    f"P(Yes|answerability) {dir_word} for {lbl} vs no_evidence "
                    f"(all formats, positions, abstain; one-sided {alt})"
                ),
                family_name=family,
                alternative=alt,
                direction_label=dir_label,
                extra={"h7_direction": dir_word},
            )
            if r:
                results.append(r)

    mcnemar_specs = (
        ("ev_man_vs_no", EVIDENCE_SUPPORTS_MAN, "evidence_supports_man"),
        ("ev_woman_vs_no", EVIDENCE_SUPPORTS_WOMAN, "evidence_supports_woman"),
    )
    for suffix, ev, lbl in mcnemar_specs:
        for higher in (True, False):
            id_mid = "" if higher else "_lower"
            dir_word = "H7↑" if higher else "H7↓"
            alt = "larger" if higher else "smaller"
            mcn = _h7_mcnemar_ev_vs_no(
                df,
                evidence=ev,
                test_id=f"h7_mcnemar_yes{id_mid}_{suffix}",
                description=(
                    f"Paired McNemar ({dir_word}): self-Q Yes at {lbl} vs no_evidence "
                    f"(same scenario, all formats/positions/abstain)"
                ),
                alternative=alt,
            )
            if mcn:
                results.append(mcn)

    if not results:
        return [_skipped("H7", "No H7 slices with both evidence and no_evidence rows.")]
    return results


def _h8_pair_keys(df: pd.DataFrame) -> tuple[str, ...]:
    return _pair_index_keys(
        df,
        (
            "base_id",
            "predicate",
            "evidence_shift",
            "question_format",
            "abstain_variant",
        ),
    )


def _h8_paired_self_main(df: pd.DataFrame) -> pd.DataFrame:
    ans = filter_slice(df, task="answerability", abstain_variant="with_abstain")
    main = filter_slice(df, task="main", abstain_variant="with_abstain")
    if "has_abstain" in main.columns:
        main = main.loc[main["has_abstain"].astype(bool)]
    if not len(ans) or not len(main):
        return pd.DataFrame()

    keys = _h8_pair_keys(df)

    def _key_cols(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame[list(keys)].copy()
        out["_key"] = list(zip(*(out[c] for c in keys), strict=True))
        return out

    a = _key_cols(ans).assign(
        self_yes=ans["answerable_yes"].astype(bool).to_numpy(),
        evidence_shift=ans["evidence_shift"].to_numpy(),
    )
    m = _key_cols(main).assign(main_c=main["abstain"].astype(bool).to_numpy())
    return a.merge(m[["_key", "main_c"]], on="_key", how="inner")


def run_h8(
    df: pd.DataFrame,
    *,
    formats: tuple[str, ...] | None = None,
) -> list[TestResult]:
    """H8: self-Q unanswerable (No) ↔ higher P(abstain C) on main (paired, with_abstain only)."""
    del formats
    if not _requires_answerability(df):
        return [_skipped("H8", H4_PENDING_5400)]

    pairs = _h8_paired_self_main(df)
    if not len(pairs):
        return [_skipped("H8", "No paired answerability↔main rows with has_abstain.")]

    results: list[TestResult] = []
    self_yes = pairs["self_yes"].astype(bool)
    main_c = pairs["main_c"].astype(bool)
    no_mask = ~self_yes
    yes_mask = self_yes
    k_no, n_no = int((main_c & no_mask).sum()), int(no_mask.sum())
    k_yes, n_yes = int((main_c & yes_mask).sum()), int(yes_mask.sum())
    if n_no == 0 or n_yes == 0:
        return [_skipped("H8", "No self-Q Yes/No pairs in with_abstain slice.")]

    z, pval, effect, ci = two_proportion_ztest(k_no, n_no, k_yes, n_yes, alternative="larger")
    p_no = k_no / n_no
    p_yes = k_yes / n_yes
    results.append(
        TestResult(
            hypothesis_id="H8",
            test_id="h8_primary_abstain_self_no_vs_yes",
            description=(
                "P(C|main) higher when self-Q=No vs self-Q=Yes "
                "(with_abstain, all formats and positions)"
            ),
            level="row",
            family_name="h8_self_abstain_link",
            n1=n_no,
            k1=k_no,
            n2=n_yes,
            k2=k_yes,
            p1=p_no,
            p2=p_yes,
            statistic=z,
            p_raw=pval,
            effect=effect,
            ci_low=ci[0],
            ci_high=ci[1],
            extra={"direction": "P(C|self=No) > P(C|self=Yes)"},
        )
    )

    a = int((self_yes & main_c).sum())
    b = int((self_yes & ~main_c).sum())
    c = int((no_mask & main_c).sum())
    d = int((no_mask & ~main_c).sum())
    stat, p_mcn = mcnemar_greater_discordant(c, b)
    results.append(
        TestResult(
            hypothesis_id="H8",
            test_id="h8_mcnemar_self_no_abstain",
            description="Paired McNemar H8: C more when self=No (with_abstain, all slices)",
            level="row",
            family_name=None,
            n1=b + c,
            k1=c,
            k2=b,
            statistic=stat,
            p_raw=p_mcn,
            effect=p_no - p_yes,
            extra={
                "mcnemar_a_self_yes_and_c": a,
                "discordant_b_self_yes_no_c": b,
                "discordant_c_self_no_and_c": c,
                "mcnemar_d_self_no_no_c": d,
                "n_pairs": int(len(pairs)),
                "mcnemar_row_axis": "self_No",
                "mcnemar_col_axis": "main_C",
                "mcnemar_h1": "c>b",
            },
        )
    )

    for tid_suffix, ev in (
        ("no_evidence", PRIMARY_EVIDENCE),
        ("ev_man", EVIDENCE_SUPPORTS_MAN),
        ("ev_woman", EVIDENCE_SUPPORTS_WOMAN),
    ):
        sub = pairs.loc[pairs["evidence_shift"] == ev]
        if not len(sub):
            continue
        sy = sub["self_yes"].astype(bool)
        mc = sub["main_c"].astype(bool)
        kn, nn = int((mc & ~sy).sum()), int((~sy).sum())
        ky, ny = int((mc & sy).sum()), int(sy.sum())
        if nn == 0 or ny == 0:
            continue
        z2, p2, eff2, ci2 = two_proportion_ztest(kn, nn, ky, ny, alternative="larger")
        results.append(
            TestResult(
                hypothesis_id="H8",
                test_id=f"h8_abstain_self_no_vs_yes_{tid_suffix}",
                description=f"H8: P(C|self=No) vs P(C|self=Yes), evidence={ev} (with_abstain)",
                level="row",
                family_name="h8_self_abstain_link",
                n1=nn,
                k1=kn,
                n2=ny,
                k2=ky,
                p1=kn / nn,
                p2=ky / ny,
                statistic=z2,
                p_raw=p2,
                effect=eff2,
                ci_low=ci2[0],
                ci_high=ci2[1],
                extra={"evidence_shift": ev},
            )
        )

    return results


def _h9_pair_keys(df: pd.DataFrame) -> tuple[str, ...]:
    return _pair_index_keys(
        df,
        ("base_id", "predicate", "question_format", "abstain_variant"),
    )


def _h9_paired_ev_table(df: pd.DataFrame) -> pd.DataFrame:
    sub = filter_slice(df, task="answerability")
    sub = sub.loc[sub["evidence_shift"].isin(EVIDENCE_WITH_CONTEXT)]
    if not len(sub):
        return pd.DataFrame()
    keys = _h9_pair_keys(sub)
    wide = sub.pivot_table(
        index=list(keys),
        columns="evidence_shift",
        values="answerable_yes",
        aggfunc="first",
    )
    need = (EVIDENCE_SUPPORTS_MAN, EVIDENCE_SUPPORTS_WOMAN)
    if not all(c in wide.columns for c in need):
        return pd.DataFrame()
    return wide.dropna(subset=list(need))


def run_h9(
    df: pd.DataFrame,
    *,
    formats: tuple[str, ...] | None = None,
) -> list[TestResult]:
    """H9: P(answerable=Yes) differs when evidence supports man vs woman (pooled)."""
    del formats
    if not _requires_answerability(df):
        return [_skipped("H9", H4_PENDING_5400)]

    base = filter_slice(df, task="answerability")
    man_df = base.loc[base["evidence_shift"] == EVIDENCE_SUPPORTS_MAN]
    wom_df = base.loc[base["evidence_shift"] == EVIDENCE_SUPPORTS_WOMAN]
    results: list[TestResult] = []

    r = _ans_two_prop_yes(
        man_df,
        wom_df,
        hypothesis_id="H9",
        test_id="h9_primary_yes_ev_man_vs_ev_woman",
        description=(
            "P(Yes|answerability) ev_man vs ev_woman "
            "(all formats, positions, abstain; two-sided)"
        ),
        family_name="h9_gender_evidence_answerability",
        alternative="two-sided",
        direction_label="P(Yes|ev_man) ≠ P(Yes|ev_woman)",
    )
    if r:
        results.append(r)

    pairs = _h9_paired_ev_table(df)
    if len(pairs):
        man_y = pairs[EVIDENCE_SUPPORTS_MAN].astype(bool)
        wom_y = pairs[EVIDENCE_SUPPORTS_WOMAN].astype(bool)
        a = int((man_y & wom_y).sum())
        b = int((man_y & ~wom_y).sum())
        c = int((~man_y & wom_y).sum())
        d = int((~man_y & ~wom_y).sum())
        stat, p_mcn = mcnemar_from_discordant(b, c)
        results.append(
            TestResult(
                hypothesis_id="H9",
                test_id="h9_mcnemar_ev_man_vs_ev_woman",
                description="Paired McNemar H9: Yes ev_man vs ev_woman (all slices)",
                level="row",
                family_name=None,
                n1=b + c,
                k1=b,
                k2=c,
                statistic=stat,
                p_raw=p_mcn,
                effect=float(man_y.mean() - wom_y.mean()),
                extra={
                    "mcnemar_a_both_yes": a,
                    "discordant_b_man_yes_woman_no": b,
                    "discordant_c_man_no_woman_yes": c,
                    "mcnemar_d_both_no": d,
                    "n_pairs": int(len(pairs)),
                    "mcnemar_row_axis": "ev_man",
                    "mcnemar_col_axis": "ev_woman",
                },
            )
        )

    if not results:
        return [_skipped("H9", "No H9 slices with both ev_man and ev_woman rows.")]
    return results


def run_h5(df: pd.DataFrame) -> list[TestResult]:

    """H5: P(abstain) higher for no_evidence than for evidence contexts."""

    results: list[TestResult] = []

    sub = filter_slice(gender_choice_frame(df), abstain_variant="with_abstain")



    pairs = [

        ("h5_abstain_no_vs_ev_man", PRIMARY_EVIDENCE, "evidence_supports_man"),

        ("h5_abstain_no_vs_ev_woman", PRIMARY_EVIDENCE, "evidence_supports_woman"),

    ]

    for tid, e1, e2 in pairs:

        s1 = sub[sub["evidence_shift"] == e1]

        s2 = sub[sub["evidence_shift"] == e2]

        k1, n1, p1 = _abstain_rate_row(s1)

        k2, n2, p2 = _abstain_rate_row(s2)

        z, pval, effect, ci = two_proportion_ztest(k1, n1, k2, n2)

        results.append(

            TestResult(

                hypothesis_id="H5",

                test_id=tid,

                description=f"P(abstain C): {e1} vs {e2} (with_abstain, all formats/positions)",

                level="row",

                family_name="h5_abstain_by_evidence",

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

            )

        )

    return results





H6_PAIR_KEYS_BASE = ("base_id", "predicate", "evidence_shift", "abstain_variant")


def _h6_merge_keys(df: pd.DataFrame) -> list[str]:
    return list(_pair_index_keys(df, H6_PAIR_KEYS_BASE))


def run_h6(df: pd.DataFrame, level: str) -> list[TestResult]:
    """H6: choice vs yes/no — pro-man and pro-woman rates on paired scenarios (full main sample)."""
    results: list[TestResult] = []
    axis = gender_axis_frame(df)
    merge_keys = _h6_merge_keys(axis)
    sub = filter_slice(axis, question_format=("yesno_man", "yesno_woman"))
    man = sub[sub["question_format"] == "yesno_man"][[*merge_keys, "yes"]].rename(columns={"yes": "y_man"})
    woman = sub[sub["question_format"] == "yesno_woman"][[*merge_keys, "yes"]].rename(columns={"yes": "y_woman"})
    pairs = man.merge(woman, on=merge_keys, how="inner")
    if not len(pairs):
        raise ValueError("H6: no yesno_man/yesno_woman pairs in slice")

    choice = filter_slice(axis, question_format=PRIMARY_FORMAT)
    merged = pairs.merge(choice[[*merge_keys, "prefers_man", "prefers_woman"]], on=merge_keys)
    if not len(merged):
        raise ValueError("H6: no choice rows aligned with yesno pairs")

    n = len(merged)
    k_m = int(merged["prefers_man"].sum())
    k_w = int(merged["prefers_woman"].sum())
    k_ym = int(merged["y_man"].sum())
    k_yw = int(merged["y_woman"].sum())
    p_m = k_m / n if n else float("nan")
    p_w = k_w / n if n else float("nan")
    p_ym = k_ym / n if n else float("nan")
    p_yw = k_yw / n if n else float("nan")

    slice_note = "all evidence, abstain variants, positions, contexts"

    z, pval, effect, ci = two_proportion_ztest(k_m, n, k_ym, n)
    results.append(
        TestResult(
            hypothesis_id="H6",
            test_id="h6_two_prop_choice_man_vs_yesno_man",
            description=f"P(man|choice) vs P(Yes|yesno_man), same-axis ({slice_note})",
            level="family" if level == "family" else "row",
            family_name="h6_format_pairs",
            n1=n,
            k1=k_m,
            n2=n,
            k2=k_ym,
            p1=p_m,
            p2=p_ym,
            statistic=z,
            p_raw=pval,
            effect=effect,
            ci_low=ci[0],
            ci_high=ci[1],
        )
    )

    z2, pval2, effect2, ci2 = two_proportion_ztest(k_w, n, k_yw, n)
    results.append(
        TestResult(
            hypothesis_id="H6",
            test_id="h6_two_prop_choice_woman_vs_yesno_woman",
            description=f"P(woman|choice) vs P(Yes|yesno_woman), same-axis ({slice_note})",
            level="family" if level == "family" else "row",
            family_name="h6_format_pairs",
            n1=n,
            k1=k_w,
            n2=n,
            k2=k_yw,
            p1=p_w,
            p2=p_yw,
            statistic=z2,
            p_raw=pval2,
            effect=effect2,
            ci_low=ci2[0],
            ci_high=ci2[1],
        )
    )

    z3, pval3, effect3, ci3 = two_proportion_ztest(k_m, n, k_yw, n)
    results.append(
        TestResult(
            hypothesis_id="H6",
            test_id="h6_two_prop_choice_man_vs_yesno_woman",
            description=f"P(man|choice) vs P(Yes|yesno_woman), cross-axis ({slice_note})",
            level="family" if level == "family" else "row",
            family_name="h6_format_pairs",
            n1=n,
            k1=k_m,
            n2=n,
            k2=k_yw,
            p1=p_m,
            p2=p_yw,
            statistic=z3,
            p_raw=pval3,
            effect=effect3,
            ci_low=ci3[0],
            ci_high=ci3[1],
        )
    )

    z4, pval4, effect4, ci4 = two_proportion_ztest(k_w, n, k_ym, n)
    results.append(
        TestResult(
            hypothesis_id="H6",
            test_id="h6_two_prop_choice_woman_vs_yesno_man",
            description=f"P(woman|choice) vs P(Yes|yesno_man), cross-axis ({slice_note})",
            level="family" if level == "family" else "row",
            family_name="h6_format_pairs",
            n1=n,
            k1=k_w,
            n2=n,
            k2=k_ym,
            p1=p_w,
            p2=p_ym,
            statistic=z4,
            p_raw=pval4,
            effect=effect4,
            ci_low=ci4[0],
            ci_high=ci4[1],
        )
    )
    return results


def run_post_hoc_h(df: pd.DataFrame, level: str) -> list[TestResult]:
    """Post-hoc: yes/no position control (not H6 primary)."""
    results: list[TestResult] = []
    merge_keys = _h6_merge_keys(df)
    sub = filter_slice(
        df,
        evidence_shift=PRIMARY_EVIDENCE,
        abstain_variant=PRIMARY_ABSTAIN,
        question_format=("yesno_man", "yesno_woman"),
    )
    man = sub[sub["question_format"] == "yesno_man"][[*merge_keys, "yes"]].rename(columns={"yes": "y_man"})
    woman = sub[sub["question_format"] == "yesno_woman"][[*merge_keys, "yes"]].rename(columns={"yes": "y_woman"})
    pairs = man.merge(woman, on=merge_keys, how="inner")
    if not len(pairs):
        raise ValueError("post_hoc H: no yesno pairs in slice")

    y_m = pairs["y_man"].astype(int).to_numpy()
    y_w = pairs["y_woman"].astype(int).to_numpy()
    a = int(((y_m == 1) & (y_w == 1)).sum())
    b = int(((y_m == 1) & (y_w == 0)).sum())
    c = int(((y_m == 0) & (y_w == 1)).sum())
    d = int(((y_m == 0) & (y_w == 0)).sum())
    _, p_mcn = mcnemar_from_discordant(b, c)

    k_ym, n_ym = int(y_m.sum()), len(pairs)
    k_yw = int(y_w.sum())
    p_ym = k_ym / n_ym if n_ym else np.nan
    p_yw = k_yw / n_ym if n_ym else np.nan
    z, pval, effect, ci = two_proportion_ztest(k_ym, n_ym, k_yw, n_ym)
    results.append(
        TestResult(
            hypothesis_id="post_hoc",
            test_id="post_h_two_prop_yesno_man_vs_yesno_woman",
            description="P(Yes|yesno_man) vs P(Yes|yesno_woman) [yes/no position control]",
            level="family" if level == "family" else "row",
            family_name=None,
            n1=n_ym,
            k1=k_ym,
            n2=n_ym,
            k2=k_yw,
            p1=p_ym,
            p2=p_yw,
            statistic=z,
            p_raw=pval,
            effect=effect,
            ci_low=ci[0],
            ci_high=ci[1],
        )
    )
    results.append(
        TestResult(
            hypothesis_id="post_hoc",
            test_id="post_h_mcnemar_yesno_pair",
            description="McNemar: yesno_man vs yesno_woman discordant pairs",
            level="family",
            family_name=None,
            n1=b + c,
            k1=b,
            n2=b + c,
            k2=c,
            statistic=float(b),
            p_raw=p_mcn,
            effect=b - c,
            extra={
                "mcnemar_a_both_yes": a,
                "discordant_b_man_yes_woman_no": b,
                "discordant_c_man_no_woman_yes": c,
                "mcnemar_d_both_no": d,
                "n_pairs": len(pairs),
                "mcnemar_row_axis": "yesno_man",
                "mcnemar_col_axis": "yesno_woman",
            },
        )
    )
    return results



def run_all(

    df: pd.DataFrame,

    *,

    hypotheses: list[str],

    level: str,

    annotation_path: Path | None,

    h4_formats: tuple[str, ...] | None = None,

) -> list[TestResult]:

    out: list[TestResult] = []
    df_main = main_task_frame(df)

    if "H1" in hypotheses:

        out.extend(run_h1(df_main, level))

    if "H2" in hypotheses and H2_ENABLED:

        out.extend(run_h2(df_main, annotation_path))

    if "H3" in hypotheses:

        out.extend(run_h3(df_main, level))

    if "H4" in hypotheses:

        out.extend(run_h4(df, formats=h4_formats))

    if "H5" in hypotheses:

        out.extend(run_h5(df_main))

    if "H6" in hypotheses:

        out.extend(run_h6(df_main, level))

    if "H7" in hypotheses:

        out.extend(run_h7(df, formats=h4_formats))

    if "H8" in hypotheses:

        out.extend(run_h8(df, formats=h4_formats))

    if "H9" in hypotheses:

        out.extend(run_h9(df, formats=h4_formats))

    return out

