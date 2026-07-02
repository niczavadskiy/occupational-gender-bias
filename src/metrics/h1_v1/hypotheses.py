"""H1 tests for v1 occupation-based runs."""

from __future__ import annotations

import hashlib
import re

import pandas as pd

from src.metrics.labels import gender_axis_frame
from src.metrics.stats_tests import TestResult, cluster_choice_preference_stats, trinomial_choice_analysis

from src.metrics.h1_v1.choice_cluster import man_vs_woman_choice_cluster_result

from src.metrics.h1_v1.labels import filter_slice
from src.metrics.h1_v1.flip_rates import enrich_context_mcnemar, enrich_position_mcnemar
from src.metrics.h1_v1.pairs import mcnemar_gender_swap_result, scenario_index_cols
from src.metrics.h1_v1.prob_tests import man_vs_woman_prob_result, run_prob_core_tests

ABSTAIN_VARIANTS = ("without_abstain", "with_abstain")
TWO_OPT_POSITIONS = ("p0", "p1")
DEFAULT_MIN_STRATUM_N = 14  # one context_order block per scenario (28 when merged)
DEFAULT_MIN_SOC_G = 15  # min base items (scenario_family_id) per soc_major_title stratum
SOC_STRATUM_ABSTAIN = "without_abstain"  # 2-option layout only (man/woman, p0/p1)


def soc_base_item_count(df: pd.DataFrame) -> int:
    """Unique scenario_family_id on gender-axis rows in a slice."""
    if "scenario_family_id" not in df.columns:
        return 0
    sub = gender_axis_frame(df)
    if sub.empty:
        return 0
    return int(sub["scenario_family_id"].astype(str).nunique())


def _slug(prefix: str, text: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:48]
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    if clean:
        return f"{prefix}_{clean}_{digest}"
    return f"{prefix}_{digest}"


def _choice_counts(df: pd.DataFrame) -> tuple[int, int, int, int]:
    sub = gender_axis_frame(df)
    n = len(sub)
    k_m = int(sub["prefers_man"].sum()) if n else 0
    k_w = int(sub["prefers_woman"].sum()) if n else 0
    k_a = int(sub["abstain"].sum()) if n and "abstain" in sub.columns else max(0, n - k_m - k_w)
    return k_m, k_w, k_a, n


def _man_rate_row(df: pd.DataFrame) -> tuple[int, int, float]:
    k_m, _, _, n = _choice_counts(df)
    return k_m, n, k_m / n if n else float("nan")


def _woman_rate_row(df: pd.DataFrame) -> tuple[int, int, float]:
    _, k_w, _, n = _choice_counts(df)
    return k_w, n, k_w / n if n else float("nan")


def _man_vs_woman_result(
    df: pd.DataFrame,
    *,
    test_id: str,
    description: str,
    family_name: str | None,
    extra: dict | None = None,
) -> TestResult | None:
    k_m, k_w, k_a, n_total = _choice_counts(df)
    if n_total == 0:
        return None
    n_gender = k_m + k_w
    if n_gender == 0:
        return None

    tri = trinomial_choice_analysis(k_m, k_w, k_a)
    pref = tri["preference_among_non_abstain"]
    if pref is None:
        return None

    p_m_raw = k_m / n_total
    p_w_raw = k_w / n_total
    ci = pref["ci_theta_wald_95"]

    sub = gender_axis_frame(df)
    cluster_col = "scenario_family_id" if "scenario_family_id" in sub.columns else None
    cluster_stats: dict = {}
    if not sub.empty and cluster_col:
        cluster_stats = cluster_choice_preference_stats(
            sub["prefers_man"].to_numpy(),
            sub["prefers_woman"].to_numpy(),
            sub[cluster_col].astype(str).to_numpy(),
        )

    merged_extra = {
        "comparison": "man_vs_woman",
        "outcome_target": "choice",
        "test_kind": "conditional_binomial",
        "h0": "P(man | man or woman) = 0.5",
        "non_abstain": n_gender,
        "k_abstain": k_a,
        "abstain_rate": tri["abstain_rate"],
        "theta_hat": pref["theta_hat"],
        "conditional_gap": pref["conditional_gap"],
        "raw_gap": tri["raw_gap"],
        "p_norm": pref["p_norm"],
        "p_exact": pref["p_exact"],
        "p_man_raw": p_m_raw,
        "p_woman_raw": p_w_raw,
        "cluster_choice": cluster_stats,
    }
    if extra:
        merged_extra.update(extra)

    return TestResult(
        hypothesis_id="H1",
        test_id=test_id,
        description=description,
        level="row",
        family_name=family_name,
        n1=n_gender,
        k1=k_m,
        n2=n_gender,
        k2=k_w,
        p1=p_m_raw,
        p2=p_w_raw,
        statistic=pref["z"],
        p_raw=pref["p_exact"],
        effect=pref["conditional_gap"],
        ci_low=ci[0],
        ci_high=ci[1],
        extra=merged_extra,
    )


def _position_layout_tests(df: pd.DataFrame) -> list[TestResult]:
    """P(man) vs P(woman) and McNemar diagnostics per layout slice."""
    results: list[TestResult] = []
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
                    tid = f"h1v1_layout_{pos}_{ctx}_{av}_man_vs_woman"
                    extra = {
                        "position_variant": pos,
                        "context_order": ctx,
                        "abstain_variant": av,
                    }
                else:
                    s = filter_slice(df, position_variant=pos, abstain_variant=av)
                    tid = f"h1v1_layout_{pos}_{av}_man_vs_woman"
                    extra = {"position_variant": pos, "abstain_variant": av}
                r = _man_vs_woman_result(
                    s,
                    test_id=tid,
                    description=(
                        f"P(man) vs P(woman): position={pos}, "
                        f"context_order={ctx}, abstain_variant={av}"
                    ),
                    family_name="h1v1_strata_layout_mw",
                    extra=extra,
                )
                if r:
                    results.append(r)

    # McNemar p0 vs p1 (2-opt), per context_order
    for ctx in context_orders:
        if ctx is None:
            sub = filter_slice(df, abstain_variant="without_abstain")
            tid = "h1v1_mcnemar_pos_p0_p1"
        else:
            sub = filter_slice(
                df,
                abstain_variant="without_abstain",
                context_order=ctx,
            )
            tid = f"h1v1_mcnemar_pos_p0_p1_{ctx}"
        idx = scenario_index_cols(sub)
        r = mcnemar_gender_swap_result(
            sub,
            index_cols=idx,
            pivot_col="position_variant",
            level_a="p0",
            level_b="p1",
            test_id=tid,
            description=(
                "McNemar p0 vs p1 (2-opt): man@p0 & woman@p1 vs woman@p0 & man@p1 "
                f"(position bias; context_order={ctx})"
            ),
            family_name="h1v1_strata_position_mcnemar",
            extra={"context_order": ctx, "abstain_variant": "without_abstain"},
        )
        if r:
            r.extra = enrich_position_mcnemar(r.extra, sub)
            results.append(r)

    # McNemar man_first vs woman_first, per position × abstain
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
                test_id=f"h1v1_mcnemar_ctx_{pos}_{av}",
                description=(
                    "McNemar man_first vs woman_first: man@mf & woman@wf vs woman@mf & man@wf "
                    f"(narrative-order bias; position={pos}, abstain={av})"
                ),
                family_name="h1v1_strata_context_mcnemar",
                extra={
                    "position_variant": pos,
                    "abstain_variant": av,
                },
            )
            if r:
                if av == "without_abstain" and pos in TWO_OPT_POSITIONS:
                    r.extra = enrich_context_mcnemar(r.extra, sub)
                results.append(r)

    return results


def run_h1_v1(
    df: pd.DataFrame,
    *,
    min_stratum_n: int = DEFAULT_MIN_STRATUM_N,
    min_soc_g: int = DEFAULT_MIN_SOC_G,
    include_soc_strata: bool = True,
    include_onet_strata: bool = True,
) -> list[TestResult]:
    """H1 for v1: primary + layout/position + occupation strata."""
    results: list[TestResult] = []

    r_primary = _man_vs_woman_result(
        df,
        test_id="h1v1_primary_man_vs_woman",
        description="P(pro-man|main) vs P(pro-woman|main) (choice; all positions, abstain, context)",
        family_name=None,
    )
    if r_primary:
        results.append(r_primary)
    r_primary_cluster = man_vs_woman_choice_cluster_result(
        df,
        test_id="h1v1_primary_man_vs_woman_cluster",
        description=(
            "E[theta_i]=0.5 (choice; theta_i=M_i/(M_i+W_i) per scenario_family_id; all layouts)"
        ),
        family_name=None,
    )
    if r_primary_cluster:
        results.append(r_primary_cluster)

    results.extend(_position_layout_tests(df))
    results.extend(run_prob_core_tests(df))

    for av in ABSTAIN_VARIANTS:
        if "abstain_variant" not in df.columns:
            break
        if av not in set(df["abstain_variant"].astype(str)):
            continue
        s = filter_slice(df, abstain_variant=av)
        r = _man_vs_woman_result(
            s,
            test_id=f"h1v1_abst_{av}_man_vs_woman",
            description=f"P(man) vs P(woman): abstain_variant={av} (pooled layout)",
            family_name="h1v1_strata_abstain",
            extra={"abstain_variant": av},
        )
        if r:
            results.append(r)
        r_prob = man_vs_woman_prob_result(
            s,
            test_id=f"h1v1_abst_{av}_margin_prob",
            description=f"E[p_man - p_woman]=0: abstain_variant={av} (pooled layout)",
            family_name="h1v1_strata_abstain_prob",
            extra={"abstain_variant": av},
        )
        if r_prob:
            results.append(r_prob)

    if include_soc_strata and "soc_major_title" in df.columns:
        for title in sorted(df["soc_major_title"].astype(str).unique()):
            if not title or title.lower() in ("nan", "none", ""):
                continue
            s = filter_slice(
                df,
                soc_major_title=title,
                abstain_variant=SOC_STRATUM_ABSTAIN,
            )
            if len(s) < min_stratum_n:
                continue
            stratum_g = soc_base_item_count(s)
            if stratum_g < min_soc_g:
                continue
            slug = _slug("h1v1_soc", title)
            soc_extra = {
                "soc_major_title": title,
                "stratum_n": len(s),
                "stratum_g": stratum_g,
                "abstain_variant": SOC_STRATUM_ABSTAIN,
            }
            r = _man_vs_woman_result(
                s,
                test_id=f"{slug}_man_vs_woman",
                description=(
                    f"P(man) vs P(woman): soc_major_title={title!r}, "
                    f"abstain_variant={SOC_STRATUM_ABSTAIN!r}"
                ),
                family_name="h1v1_strata_soc_major",
                extra=soc_extra,
            )
            if r:
                results.append(r)
            r_prob = man_vs_woman_prob_result(
                s,
                test_id=f"{slug}_margin_prob",
                description=(
                    f"E[p_man - p_woman]=0: soc_major_title={title!r}, "
                    f"abstain_variant={SOC_STRATUM_ABSTAIN!r}"
                ),
                family_name="h1v1_strata_soc_major_prob",
                extra=soc_extra,
            )
            if r_prob:
                results.append(r_prob)

    if include_onet_strata and "onet_action" in df.columns and "soc" in df.columns:
        group_cols = ["soc", "profession", "onet_action"]
        if "soc_major_title" in df.columns:
            group_cols.insert(2, "soc_major_title")
        for keys, s in df.groupby(group_cols, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_map = dict(zip(group_cols, keys, strict=True))
            if len(s) < min_stratum_n:
                continue
            soc = str(key_map["soc"])
            action = str(key_map["onet_action"])
            profession = str(key_map.get("profession", ""))
            title = str(key_map.get("soc_major_title", ""))
            scenario_key = f"{soc}|{action}"
            slug = _slug("h1v1_onet", scenario_key)
            extra_onet = {
                "soc": soc,
                "profession": profession,
                "onet_action": action,
                "soc_major_title": title,
                "stratum_n": len(s),
            }
            r = _man_vs_woman_result(
                s,
                test_id=f"{slug}_man_vs_woman",
                description=(
                    f"P(man) vs P(woman): soc={soc!r}, profession={profession!r}, "
                    f"onet_action={action!r}"
                ),
                family_name="h1v1_strata_onet_action",
                extra=extra_onet,
            )
            if r:
                results.append(r)
            r_prob = man_vs_woman_prob_result(
                s,
                test_id=f"{slug}_margin_prob",
                description=(
                    f"E[p_man - p_woman]=0: soc={soc!r}, profession={profession!r}, "
                    f"onet_action={action!r}"
                ),
                family_name="h1v1_strata_onet_action_prob",
                extra=extra_onet,
            )
            if r_prob:
                results.append(r_prob)

    return results
