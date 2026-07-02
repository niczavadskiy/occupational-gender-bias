"""Prob-constrained outcome tests for H1 v1 (margin / mean p_man vs p_woman)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.metrics.labels import gender_axis_frame
from src.metrics.stats_tests import (
    TestResult,
    cluster_one_sample_margin_stats,
    one_sample_ttest_vs,
    shapiro_wilk_normality,
    wilcoxon_vs,
)

from src.metrics.h1_v1.flip_rates import (
    enrich_context_mcnemar_prob,
    enrich_position_mcnemar_prob,
)
from src.metrics.h1_v1.labels import filter_slice
from src.metrics.h1_v1.pairs import V1_SCENARIO_KEYS, mcnemar_gender_swap_result, scenario_index_cols

ABSTAIN_VARIANTS = ("without_abstain", "with_abstain")
TWO_OPT_POSITIONS = ("p0", "p1")
PROB_TARGET = "prob_constrained"
MAN_COL = "prob_prefers_man"
WOMAN_COL = "prob_prefers_woman"


def _margin_ci(margin: np.ndarray, *, se: float | None = None) -> tuple[float, float]:
    margin = margin[np.isfinite(margin)]
    n = len(margin)
    if n == 0:
        return (float("nan"), float("nan"))
    mean = float(np.mean(margin))
    if n < 2:
        return (mean, mean)
    if se is None or not math.isfinite(se):
        se = float(np.std(margin, ddof=1)) / math.sqrt(n)
    return (mean - 1.96 * se, mean + 1.96 * se)


def _cluster_col(sub: pd.DataFrame) -> str:
    if "scenario_family_id" in sub.columns:
        return "scenario_family_id"
    keys = [c for c in V1_SCENARIO_KEYS if c in sub.columns]
    if keys:
        sub["_cluster_id"] = sub[keys].astype(str).agg("|".join, axis=1)
        return "_cluster_id"
    sub["_cluster_id"] = sub.index.astype(str)
    return "_cluster_id"


def _prob_layout_contrasts(sub: pd.DataFrame, cluster_col: str) -> dict[str, dict[str, float]]:
    """Paired swap contrasts aggregated to base-item level (when 2×2 layout is complete)."""
    need = {"position_variant", "context_order", "gender_margin", cluster_col}
    if not need.issubset(sub.columns):
        return {}
    pos_levels = {"p0", "p1"}
    ctx_levels = {"woman_first", "man_first"}
    if not pos_levels.issubset(set(sub["position_variant"].astype(str))):
        return {}
    if not ctx_levels.issubset(set(sub["context_order"].astype(str))):
        return {}

    wide = sub.pivot_table(
        index=cluster_col,
        columns=["position_variant", "context_order"],
        values="gender_margin",
        aggfunc="first",
    )
    expected_cols = [
        ("p0", "woman_first"),
        ("p1", "woman_first"),
        ("p0", "man_first"),
        ("p1", "man_first"),
    ]
    if not all(c in wide.columns for c in expected_cols):
        return {}
    wide = wide.dropna(subset=expected_cols)
    if len(wide) < 2:
        return {}

    p0 = (wide[("p0", "woman_first")] + wide[("p0", "man_first")]) / 2.0
    p1 = (wide[("p1", "woman_first")] + wide[("p1", "man_first")]) / 2.0
    wf = (wide[("p0", "woman_first")] + wide[("p1", "woman_first")]) / 2.0
    mf = (wide[("p0", "man_first")] + wide[("p1", "man_first")]) / 2.0
    contrasts = {
        "gender_mean": wide[expected_cols].mean(axis=1).to_numpy(dtype=float),
        "position_effect": (p0 - p1).to_numpy(dtype=float),
        "context_order_effect": (wf - mf).to_numpy(dtype=float),
        "position_x_context": (
            (wide[("p0", "woman_first")] - wide[("p1", "woman_first")])
            - (wide[("p0", "man_first")] - wide[("p1", "man_first")])
        ).to_numpy(dtype=float),
    }
    out: dict[str, dict[str, float]] = {}
    for name, delta in contrasts.items():
        stat, pval = one_sample_ttest_vs(delta, 0.0)
        out[name] = {
            "effect": float(np.mean(delta)),
            "t": stat,
            "p": pval,
            "n_clusters": int(len(delta)),
        }
    return out


def man_vs_woman_prob_result(
    df: pd.DataFrame,
    *,
    test_id: str,
    description: str,
    family_name: str | None,
    extra: dict | None = None,
) -> TestResult | None:
    """
    Test H0: E[p_man - p_woman] = 0 on gender-axis rows.

    Primary inference uses mean margin per base item (scenario_family_id) to avoid
    treating swap variants as independent observations. Naive row-level stats are
    retained in extra for design-effect diagnostics.
    """
    sub = gender_axis_frame(df)
    if "gender_margin" not in sub.columns:
        return None
    sub = sub.loc[sub["gender_margin"].notna()].copy()
    if sub.empty:
        return None

    cluster_col = _cluster_col(sub)
    margin = sub["gender_margin"].astype(float).to_numpy()
    groups = sub[cluster_col].astype(str).to_numpy()
    mean_m = float(np.mean(sub["p_man"]))
    mean_w = float(np.mean(sub["p_woman"]))
    mean_margin = float(np.mean(margin))

    cluster = cluster_one_sample_margin_stats(margin, groups, mu=0.0)
    cluster_means = (
        sub.groupby(cluster_col, sort=False)["gender_margin"]
        .mean()
        .astype(float)
        .to_numpy()
    )
    cluster_mean_margin = float(np.mean(cluster_means)) if len(cluster_means) else float("nan")
    w_stat, w_pval = wilcoxon_vs(margin, 0.0)
    ci_low, ci_high = _margin_ci(
        cluster_means,
        se=cluster.get("cluster_se"),
    )

    merged = {
        "outcome_target": PROB_TARGET,
        "comparison": "mean_margin_man_minus_woman",
        "cluster_col": cluster_col,
        "mean_p_man": mean_m,
        "mean_p_woman": mean_w,
        "sum_p_man": float(sub["p_man"].sum()),
        "sum_p_woman": float(sub["p_woman"].sum()),
        **shapiro_wilk_normality(margin),
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_pval,
        "cluster_mean_margin": cluster_mean_margin,
        **cluster,
        "layout_contrasts": _prob_layout_contrasts(sub, cluster_col),
    }
    if extra:
        merged.update(extra)
    return TestResult(
        hypothesis_id="H1",
        test_id=test_id,
        description=description,
        level="cluster",
        family_name=family_name,
        n1=cluster["n_clusters"],
        k1=None,
        n2=None,
        k2=None,
        p1=mean_m,
        p2=mean_w,
        statistic=cluster["cluster_t"],
        p_raw=cluster["cluster_p"],
        effect=mean_margin,
        ci_low=ci_low,
        ci_high=ci_high,
        extra=merged,
    )


def run_prob_core_tests(df: pd.DataFrame) -> list[TestResult]:
    """Primary + layout + McNemar (position, context p0/p1) on prob axis."""
    results: list[TestResult] = []

    r0 = man_vs_woman_prob_result(
        df,
        test_id="h1v1_primary_man_vs_woman_prob",
        description=(
            "E[p_man - p_woman]=0 (position-aware prob_constrained; all layouts)"
        ),
        family_name=None,
    )
    if r0:
        results.append(r0)

    if "position_variant" not in df.columns:
        return results

    context_orders = (
        sorted(df["context_order"].astype(str).unique())
        if "context_order" in df.columns
        else [None]
    )
    positions = sorted(df["position_variant"].astype(str).unique())

    for ctx in context_orders:
        for pos in positions:
            for av in ABSTAIN_VARIANTS:
                if av not in set(df.get("abstain_variant", pd.Series(dtype=str)).astype(str)):
                    continue
                if av == "without_abstain" and pos not in TWO_OPT_POSITIONS:
                    continue
                if ctx is not None:
                    s = filter_slice(
                        df,
                        position_variant=pos,
                        context_order=ctx,
                        abstain_variant=av,
                    )
                    tid = f"h1v1_layout_{pos}_{ctx}_{av}_margin_prob"
                    extra = {
                        "position_variant": pos,
                        "context_order": ctx,
                        "abstain_variant": av,
                    }
                else:
                    s = filter_slice(df, position_variant=pos, abstain_variant=av)
                    tid = f"h1v1_layout_{pos}_{av}_margin_prob"
                    extra = {"position_variant": pos, "abstain_variant": av}
                r = man_vs_woman_prob_result(
                    s,
                    test_id=tid,
                    description=(
                        f"E[p_man - p_woman]=0: position={pos}, "
                        f"context_order={ctx}, abstain_variant={av}"
                    ),
                    family_name="h1v1_strata_layout_margin_prob",
                    extra=extra,
                )
                if r:
                    results.append(r)

    for ctx in context_orders:
        if ctx is None:
            sub = filter_slice(df, abstain_variant="without_abstain")
            tid = "h1v1_mcnemar_pos_p0_p1_prob"
        else:
            sub = filter_slice(
                df,
                abstain_variant="without_abstain",
                context_order=ctx,
            )
            tid = f"h1v1_mcnemar_pos_p0_p1_{ctx}_prob"
        idx = scenario_index_cols(sub)
        r = mcnemar_gender_swap_result(
            sub,
            index_cols=idx,
            pivot_col="position_variant",
            level_a="p0",
            level_b="p1",
            test_id=tid,
            description=(
                "McNemar p0 vs p1 (prob argmax): man@p0 & woman@p1 vs woman@p0 & man@p1 "
                f"(position bias; context_order={ctx})"
            ),
            family_name="h1v1_strata_position_mcnemar_prob",
            extra={"context_order": ctx, "abstain_variant": "without_abstain"},
            man_col=MAN_COL,
            woman_col=WOMAN_COL,
            outcome_target=PROB_TARGET,
        )
        if r:
            r.extra = enrich_position_mcnemar_prob(r.extra, sub)
            results.append(r)

    for pos in positions:
        for av in ABSTAIN_VARIANTS:
            if av not in set(df.get("abstain_variant", pd.Series(dtype=str)).astype(str)):
                continue
            if av == "without_abstain" and pos not in TWO_OPT_POSITIONS:
                continue
            sub = filter_slice(df, position_variant=pos, abstain_variant=av)
            if "context_order" not in sub.columns:
                continue
            ctx_vals = set(sub["context_order"].astype(str).unique())
            if not {"man_first", "woman_first"}.issubset(ctx_vals):
                continue
            r = mcnemar_gender_swap_result(
                sub,
                index_cols=scenario_index_cols(sub, "position_variant", "abstain_variant"),
                pivot_col="context_order",
                level_a="man_first",
                level_b="woman_first",
                test_id=f"h1v1_mcnemar_ctx_{pos}_{av}_prob",
                description=(
                    "McNemar mf vs wf (prob argmax): man@mf & woman@wf vs woman@mf & man@wf "
                    f"(narrative-order bias; position={pos}, abstain={av})"
                ),
                family_name="h1v1_strata_context_mcnemar_prob",
                extra={
                    "position_variant": pos,
                    "abstain_variant": av,
                },
                man_col=MAN_COL,
                woman_col=WOMAN_COL,
                outcome_target=PROB_TARGET,
            )
            if r:
                if av == "without_abstain" and pos in TWO_OPT_POSITIONS:
                    r.extra = enrich_context_mcnemar_prob(r.extra, sub)
                results.append(r)

    return results
