"""Paired prob-constrained tests for evidence shift (delta p_man / p_woman)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from src.metrics.labels import gender_axis_frame
from src.metrics.stats_tests import TestResult, one_sample_ttest_vs, shapiro_wilk_normality
from src.metrics.h3_v1.labels import PAIR_KEY

PROB_TARGET = "prob_constrained"


def _paired_delta(
    df: pd.DataFrame,
    *,
    level_a: str,
    level_b: str,
    value_col: str,
) -> np.ndarray | None:
    sub = gender_axis_frame(df)
    if not len(sub) or "evidence_shift" not in sub.columns:
        return None
    if value_col not in sub.columns:
        return None
    if not all(c in sub.columns for c in PAIR_KEY):
        return None

    wide = sub.pivot_table(
        index=list(PAIR_KEY),
        columns="evidence_shift",
        values=value_col,
        aggfunc="first",
    )
    if level_a not in wide.columns or level_b not in wide.columns:
        return None
    delta = (wide[level_b] - wide[level_a]).astype(float).to_numpy()
    return delta[np.isfinite(delta)]


def _margin_ci(margin: np.ndarray) -> tuple[float, float]:
    margin = margin[np.isfinite(margin)]
    n = len(margin)
    if n == 0:
        return (float("nan"), float("nan"))
    mean = float(np.mean(margin))
    if n < 2:
        return (mean, mean)
    se = float(np.std(margin, ddof=1)) / math.sqrt(n)
    return (mean - 1.96 * se, mean + 1.96 * se)


def _one_sample_greater(x: np.ndarray, mu: float = 0.0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    if len(x) == 1:
        if x[0] == mu:
            return (0.0, 1.0)
        return (np.inf if x[0] > mu else -np.inf, 0.0 if x[0] > mu else 1.0)
    t, p = stats.ttest_1samp(x, mu, alternative="greater")
    return (float(t), float(p))


def evidence_delta_prob_result(
    df: pd.DataFrame,
    *,
    level_a: str,
    level_b: str,
    gender_axis: str,
    test_id: str,
    description: str,
    family_name: str | None,
    extra: dict | None = None,
    one_sided: bool = True,
) -> TestResult | None:
    """Paired delta of p_man or p_woman: level_b minus level_a per id."""
    if gender_axis == "man":
        col = "p_man"
    elif gender_axis == "woman":
        col = "p_woman"
    else:
        raise ValueError(gender_axis)

    delta = _paired_delta(df, level_a=level_a, level_b=level_b, value_col=col)
    if delta is None or len(delta) == 0:
        return None

    mean_d = float(np.mean(delta))
    if one_sided:
        stat, pval = _one_sample_greater(delta, 0.0)
    else:
        stat, pval = one_sample_ttest_vs(delta, 0.0)

    ci_low, ci_high = _margin_ci(delta)
    merged = {
        "outcome_target": PROB_TARGET,
        "comparison": f"delta_{col}",
        "evidence_a": level_a,
        "evidence_b": level_b,
        "gender_axis": gender_axis,
        "mean_delta": mean_d,
        "pair_key": list(PAIR_KEY),
        **shapiro_wilk_normality(delta),
    }
    if extra:
        merged.update(extra)

    return TestResult(
        hypothesis_id="H3",
        test_id=test_id,
        description=description,
        level="row",
        family_name=family_name,
        n1=len(delta),
        statistic=stat,
        p_raw=pval,
        effect=mean_d,
        ci_low=ci_low,
        ci_high=ci_high,
        extra=merged,
    )
