"""Cluster (item-level) choice preference tests — parallel to prob margin cluster t-test."""

from __future__ import annotations

import pandas as pd

from src.metrics.labels import gender_axis_frame
from src.metrics.stats_tests import TestResult, cluster_choice_preference_stats

from src.metrics.h1_v1.pairs import V1_SCENARIO_KEYS


def _cluster_col(sub: pd.DataFrame) -> str:
    if "scenario_family_id" in sub.columns:
        return "scenario_family_id"
    keys = [c for c in V1_SCENARIO_KEYS if c in sub.columns]
    if keys:
        sub["_cluster_id"] = sub[keys].astype(str).agg("|".join, axis=1)
        return "_cluster_id"
    sub["_cluster_id"] = sub.index.astype(str)
    return "_cluster_id"


def man_vs_woman_choice_cluster_result(
    df: pd.DataFrame,
    *,
    test_id: str,
    description: str,
    family_name: str | None,
    extra: dict | None = None,
) -> TestResult | None:
    """
    H0: E[theta_i] = 0.5 where theta_i = M_i/(M_i+W_i) per scenario_family_id.

    Parallel to man_vs_woman_prob_result (cluster t-test on item means).
    Ties within item (e.g. 2 man, 2 woman) -> theta_i = 0.5.
    """
    sub = gender_axis_frame(df)
    if sub.empty:
        return None
    if "prefers_man" not in sub.columns or "prefers_woman" not in sub.columns:
        return None

    sub = sub.copy()
    cluster_col = _cluster_col(sub)
    groups = sub[cluster_col].astype(str).to_numpy()

    cluster = cluster_choice_preference_stats(
        sub["prefers_man"].to_numpy(),
        sub["prefers_woman"].to_numpy(),
        groups,
    )
    if cluster["n_clusters_with_gender"] == 0:
        return None

    ci = cluster["ci_theta_cluster_wald_95"]
    merged = {
        "outcome_target": "choice",
        "test_kind": "cluster_theta",
        "comparison": "mean_theta_man_among_gender_per_item",
        "cluster_col": cluster_col,
        **cluster,
    }
    if extra:
        merged.update(extra)

    return TestResult(
        hypothesis_id="H1",
        test_id=test_id,
        description=description,
        level="cluster",
        family_name=family_name,
        n1=cluster["n_clusters_with_gender"],
        k1=cluster["sign_n_man_wins"],
        n2=cluster["n_clusters_with_gender"],
        k2=cluster["sign_n_woman_wins"],
        p1=cluster["mean_theta"],
        p2=1.0 - cluster["mean_theta"] if cluster["mean_theta"] == cluster["mean_theta"] else None,
        statistic=cluster["cluster_t"],
        p_raw=cluster["cluster_p"],
        effect=cluster["mean_gap"],
        ci_low=ci[0],
        ci_high=ci[1],
        extra=merged,
    )
