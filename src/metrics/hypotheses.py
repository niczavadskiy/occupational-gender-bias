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

from src.metrics.labels import filter_slice

from src.metrics.stats_tests import (

    TestResult,

    chi2_independence,

    mcnemar_from_discordant,

    one_proportion_ztest,

    one_sample_ttest_vs,

    two_proportion_ztest,

    wilcoxon_vs,

)



PRIMARY_EVIDENCE = "no_evidence"

PRIMARY_ABSTAIN = "without_abstain"

PRIMARY_FORMAT = "choice"



ANSWERABILITY_PENDING = (

    "Requires answerability / unanswerability labels (not in per_item.jsonl yet)."

)





def _man_rate_row(df: pd.DataFrame) -> tuple[int, int, float]:

    n = len(df)

    k = int(df["prefers_man"].sum()) if n else 0

    return k, n, k / n if n else float("nan")


def _woman_rate_row(df: pd.DataFrame) -> tuple[int, int, float]:

    n = len(df)

    k = int(df["prefers_woman"].sum()) if n else 0

    return k, n, k / n if n else float("nan")





def _abstain_rate_row(df: pd.DataFrame) -> tuple[int, int, float]:

    n = len(df)

    k = int(df["abstain"].sum()) if n else 0

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





def run_h1(df: pd.DataFrame, level: str) -> list[TestResult]:

    """H1: preference on gender (social axis)."""

    results: list[TestResult] = []

    sub = filter_slice(

        df,

        evidence_shift=PRIMARY_EVIDENCE,

        question_format=PRIMARY_FORMAT,

        abstain_variant=PRIMARY_ABSTAIN,

    )

    k, n, p = _man_rate_row(sub)

    z, pval = one_proportion_ztest(k, n, 0.5)

    results.append(

        TestResult(

            hypothesis_id="H1",

            test_id="h1_primary_vs_half",

            description="P(man|choice,no_evidence,without_abstain) vs 0.5",

            level="row",

            family_name=None,

            n1=n,

            k1=k,

            p1=p,

            statistic=z,

            p_raw=pval,

            effect=p - 0.5,

        )

    )



    if level == "family":

        fam = _family_rates(sub)

        rates = fam["rate"].to_numpy()

        t, pval = one_sample_ttest_vs(rates, 0.5)

        w, pw = wilcoxon_vs(rates, 0.5)

        results.append(

            TestResult(

                hypothesis_id="H1",

                test_id="h1_primary_vs_half_family_t",

                description="Family mean rate_man vs 0.5 (one-sample t)",

                level="family",

                family_name=None,

                n1=len(rates),

                p1=float(np.mean(rates)),

                statistic=t,

                p_raw=pval,

                extra={"wilcoxon_p": pw, "wilcoxon_stat": w},

            )

        )



    strata = [

        ("h1_vs_half_no_evidence_choice_without", PRIMARY_EVIDENCE),

        ("h1_vs_half_evidence_man_choice_without", "evidence_supports_man"),

        ("h1_vs_half_evidence_woman_choice_without", "evidence_supports_woman"),

    ]

    for tid, ev in strata:

        s = filter_slice(

            df,

            evidence_shift=ev,

            question_format=PRIMARY_FORMAT,

            abstain_variant=PRIMARY_ABSTAIN,

        )

        k, n, p = _man_rate_row(s)

        z, pval = one_proportion_ztest(k, n, 0.5)

        results.append(

            TestResult(

                hypothesis_id="H1",

                test_id=tid,

                description=f"P(man) vs 0.5: evidence={ev}, without_abstain",

                level="row",

                family_name="h1_strata_vs_half",

                n1=n,

                k1=k,

                p1=p,

                statistic=z,

                p_raw=pval,

                effect=p - 0.5,

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

    sub = filter_slice(

        df,

        question_format=PRIMARY_FORMAT,

        abstain_variant=PRIMARY_ABSTAIN,

    )

    ev_order = ["no_evidence", "evidence_supports_man", "evidence_supports_woman"]

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

            description="chi2: evidence_shift x (man,woman) on choice",

            level="row",

            family_name=None,

            statistic=chi2,

            p_raw=p_chi,

            extra={"dof": dof, "table": table, "expected": expected.tolist()},

        )

    )

    sub_ab = filter_slice(df, question_format=PRIMARY_FORMAT, abstain_variant="with_abstain")
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
            description="chi2: evidence_shift x (man,woman,C) on choice, with_abstain",
            level="row",
            family_name=None,
            statistic=chi2_ab if np.isfinite(chi2_ab) else None,
            p_raw=p_chi_ab if np.isfinite(p_chi_ab) else None,
            extra=chi_ab_extra,
        )
    )

    pairs = [

        ("h3_pair_no_vs_man", "no_evidence", "evidence_supports_man"),

        ("h3_pair_no_vs_woman", "no_evidence", "evidence_supports_woman"),

        ("h3_pair_man_vs_woman", "evidence_supports_man", "evidence_supports_woman"),

    ]

    for tid, e1, e2 in pairs:

        s1 = sub[sub["evidence_shift"] == e1]

        s2 = sub[sub["evidence_shift"] == e2]

        k1, n1, p1 = _man_rate_row(s1)

        k2, n2, p2 = _man_rate_row(s2)

        z, pval, effect, ci = two_proportion_ztest(k1, n1, k2, n2)

        results.append(

            TestResult(

                hypothesis_id="H3",

                test_id=tid,

                description=f"P(man): {e1} vs {e2}",

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



    evw = filter_slice(

        df,

        evidence_shift="evidence_supports_woman",

        question_format=PRIMARY_FORMAT,

        abstain_variant=PRIMARY_ABSTAIN,

    )

    k, n, p = _man_rate_row(evw)

    z, pval = one_proportion_ztest(k, n, 0.5)

    results.append(

        TestResult(

            hypothesis_id="H3",

            test_id="h3_ev_woman_rate_man_vs_half",

            description="P(man) on evidence_supports_woman vs 0.5 (evidence override check)",

            level="row",

            family_name=None,

            n1=n,

            k1=k,

            p1=p,

            statistic=z,

            p_raw=pval,

            effect=p - 0.5,

        )

    )

    return results





def run_h4() -> list[TestResult]:

    """H4: abstain option increases answerability (needs answerability labels)."""

    return [_skipped("H4", ANSWERABILITY_PENDING)]





def run_h5(df: pd.DataFrame) -> list[TestResult]:

    """H5: P(abstain) higher for no_evidence than for evidence contexts."""

    results: list[TestResult] = []

    sub = filter_slice(df, question_format=PRIMARY_FORMAT, abstain_variant="with_abstain")



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

                description=f"P(abstain C): {e1} vs {e2} (with_abstain, choice)",

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





def run_h6(df: pd.DataFrame, level: str) -> list[TestResult]:
    """H6: choice vs yes/no — pro-man and pro-woman rates on paired base_ids."""
    results: list[TestResult] = []
    sub = filter_slice(
        df,
        evidence_shift=PRIMARY_EVIDENCE,
        abstain_variant=PRIMARY_ABSTAIN,
        question_format=("yesno_man", "yesno_woman"),
    )
    man = sub[sub["question_format"] == "yesno_man"][["base_id", "yes"]].rename(columns={"yes": "y_man"})
    woman = sub[sub["question_format"] == "yesno_woman"][["base_id", "yes"]].rename(columns={"yes": "y_woman"})
    pairs = man.merge(woman, on="base_id", how="inner")
    if len(pairs) != 75:
        raise ValueError(f"H6 expected 75 yesno pairs, got {len(pairs)}")

    choice = filter_slice(
        df,
        evidence_shift=PRIMARY_EVIDENCE,
        abstain_variant=PRIMARY_ABSTAIN,
        question_format=PRIMARY_FORMAT,
    )
    merged = pairs.merge(choice[["base_id", "prefers_man", "prefers_woman"]], on="base_id")

    k_m, n_m, p_m = _man_rate_row(choice)
    k_ym = int(merged["y_man"].sum())
    z, pval, effect, ci = two_proportion_ztest(k_m, n_m, k_ym, len(merged))
    results.append(
        TestResult(
            hypothesis_id="H6",
            test_id="h6_two_prop_choice_man_vs_yesno_man",
            description="P(man|choice) vs P(Yes|yesno_man), same-axis",
            level="family" if level == "family" else "row",
            family_name="h6_format_pairs",
            n1=n_m,
            k1=k_m,
            n2=len(merged),
            k2=k_ym,
            p1=p_m,
            p2=float(merged["y_man"].mean()),
            statistic=z,
            p_raw=pval,
            effect=effect,
            ci_low=ci[0],
            ci_high=ci[1],
        )
    )

    k_w, n_w, p_w = _woman_rate_row(choice)
    k_yw = int(merged["y_woman"].sum())
    z2, pval2, effect2, ci2 = two_proportion_ztest(k_w, n_w, k_yw, len(merged))
    results.append(
        TestResult(
            hypothesis_id="H6",
            test_id="h6_two_prop_choice_woman_vs_yesno_woman",
            description="P(woman|choice) vs P(Yes|yesno_woman), same-axis",
            level="family" if level == "family" else "row",
            family_name="h6_format_pairs",
            n1=n_w,
            k1=k_w,
            n2=len(merged),
            k2=k_yw,
            p1=p_w,
            p2=float(merged["y_woman"].mean()),
            statistic=z2,
            p_raw=pval2,
            effect=effect2,
            ci_low=ci2[0],
            ci_high=ci2[1],
        )
    )

    z3, pval3, effect3, ci3 = two_proportion_ztest(k_m, n_m, k_yw, len(merged))
    results.append(
        TestResult(
            hypothesis_id="H6",
            test_id="h6_two_prop_choice_man_vs_yesno_woman",
            description="P(man|choice) vs P(Yes|yesno_woman), cross-axis",
            level="family" if level == "family" else "row",
            family_name="h6_format_pairs",
            n1=n_m,
            k1=k_m,
            n2=len(merged),
            k2=k_yw,
            p1=p_m,
            p2=float(merged["y_woman"].mean()),
            statistic=z3,
            p_raw=pval3,
            effect=effect3,
            ci_low=ci3[0],
            ci_high=ci3[1],
        )
    )

    z4, pval4, effect4, ci4 = two_proportion_ztest(k_w, n_w, k_ym, len(merged))
    results.append(
        TestResult(
            hypothesis_id="H6",
            test_id="h6_two_prop_choice_woman_vs_yesno_man",
            description="P(woman|choice) vs P(Yes|yesno_man), cross-axis",
            level="family" if level == "family" else "row",
            family_name="h6_format_pairs",
            n1=n_w,
            k1=k_w,
            n2=len(merged),
            k2=k_ym,
            p1=p_w,
            p2=float(merged["y_man"].mean()),
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
    sub = filter_slice(
        df,
        evidence_shift=PRIMARY_EVIDENCE,
        abstain_variant=PRIMARY_ABSTAIN,
        question_format=("yesno_man", "yesno_woman"),
    )
    man = sub[sub["question_format"] == "yesno_man"][["base_id", "yes"]].rename(columns={"yes": "y_man"})
    woman = sub[sub["question_format"] == "yesno_woman"][["base_id", "yes"]].rename(columns={"yes": "y_woman"})
    pairs = man.merge(woman, on="base_id", how="inner")
    if len(pairs) != 75:
        raise ValueError(f"post_hoc H expected 75 yesno pairs, got {len(pairs)}")

    y_m = pairs["y_man"].astype(int).to_numpy()
    y_w = pairs["y_woman"].astype(int).to_numpy()
    b = int(((y_m == 1) & (y_w == 0)).sum())
    c = int(((y_m == 0) & (y_w == 1)).sum())
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
            extra={"discordant_b_man_yes_woman_no": b, "discordant_c_man_no_woman_yes": c},
        )
    )
    return results



def run_h7() -> list[TestResult]:

    return [_skipped("H7", ANSWERABILITY_PENDING)]





def run_h8() -> list[TestResult]:

    return [_skipped("H8", ANSWERABILITY_PENDING)]





def run_h9() -> list[TestResult]:

    return [_skipped("H9", ANSWERABILITY_PENDING)]





def run_all(

    df: pd.DataFrame,

    *,

    hypotheses: list[str],

    level: str,

    annotation_path: Path | None,

) -> list[TestResult]:

    out: list[TestResult] = []

    if "H1" in hypotheses:

        out.extend(run_h1(df, level))

    if "H2" in hypotheses:

        out.extend(run_h2(df, annotation_path))

    if "H3" in hypotheses:

        out.extend(run_h3(df, level))

    if "H4" in hypotheses:

        out.extend(run_h4())

    if "H5" in hypotheses:

        out.extend(run_h5(df))

    if "H6" in hypotheses:

        out.extend(run_h6(df, level))

    if "H7" in hypotheses:

        out.extend(run_h7())

    if "H8" in hypotheses:

        out.extend(run_h8())

    if "H9" in hypotheses:

        out.extend(run_h9())

    return out

