"""Paired tables for v1 position / context McNemar tests."""

from __future__ import annotations

import pandas as pd

from src.metrics.labels import gender_axis_frame
from src.metrics.stats_tests import TestResult, mcnemar_from_discordant

V1_SCENARIO_KEYS = ("soc", "profession", "onet_action")


def scenario_index_cols(df: pd.DataFrame, *fixed: str) -> tuple[str, ...]:
    """Pairing index: scenario + any fixed layout fields (not the swapped factor)."""
    cols = [c for c in V1_SCENARIO_KEYS if c in df.columns]
    for c in fixed:
        if c in df.columns and c not in cols:
            cols.append(c)
    return tuple(cols)


def _gender_mcnemar_counts(
    df: pd.DataFrame,
    *,
    index_cols: tuple[str, ...],
    pivot_col: str,
    level_a: str,
    level_b: str,
    man_col: str = "prefers_man",
    woman_col: str = "prefers_woman",
) -> tuple[int, int, int, int, int] | None:
    """Return a,b,c,d,n_pairs for gender discordance across two pivot levels."""
    sub = gender_axis_frame(df)
    if not len(sub) or pivot_col not in sub.columns:
        return None
    if not all(c in sub.columns for c in index_cols):
        return None
    if man_col not in sub.columns or woman_col not in sub.columns:
        return None

    wide_m = sub.pivot_table(
        index=list(index_cols),
        columns=pivot_col,
        values=man_col,
        aggfunc="first",
    )
    wide_w = sub.pivot_table(
        index=list(index_cols),
        columns=pivot_col,
        values=woman_col,
        aggfunc="first",
    )
    if level_a not in wide_m.columns or level_b not in wide_m.columns:
        return None

    m0 = wide_m[level_a].fillna(False).astype(bool)
    w0 = wide_w[level_a].fillna(False).astype(bool)
    m1 = wide_m[level_b].fillna(False).astype(bool)
    w1 = wide_w[level_b].fillna(False).astype(bool)

    valid = (m0 | w0) & (m1 | w1) & ~(m0 & w0) & ~(m1 & w1)
    if not valid.any():
        return None

    b = int((m0 & w1 & valid).sum())
    c = int((w0 & m1 & valid).sum())
    a = int((m0 & m1 & valid).sum())
    d = int((w0 & w1 & valid).sum())
    n = int(valid.sum())
    return a, b, c, d, n


def mcnemar_gender_swap_result(
    df: pd.DataFrame,
    *,
    index_cols: tuple[str, ...],
    pivot_col: str,
    level_a: str,
    level_b: str,
    test_id: str,
    description: str,
    family_name: str | None,
    extra: dict | None = None,
    man_col: str = "prefers_man",
    woman_col: str = "prefers_woman",
    outcome_target: str = "choice",
) -> TestResult | None:
    """
    McNemar on gender discordance across two layout levels.

    b: prefers_man at level_a & prefers_woman at level_b
    c: prefers_woman at level_a & prefers_man at level_b
    """
    counts = _gender_mcnemar_counts(
        df,
        index_cols=index_cols,
        pivot_col=pivot_col,
        level_a=level_a,
        level_b=level_b,
        man_col=man_col,
        woman_col=woman_col,
    )
    if counts is None:
        return None
    a, b, c, d, n_pairs = counts
    if b + c == 0:
        stat, pval = 0.0, 1.0
    else:
        stat, pval = mcnemar_from_discordant(b, c)
    merged = {
        "mcnemar_a_both_man": a,
        "discordant_b_man_a_woman_b": b,
        "discordant_c_woman_a_man_b": c,
        "mcnemar_d_both_woman": d,
        "n_pairs": n_pairs,
        "mcnemar_row_axis": level_a,
        "mcnemar_col_axis": level_b,
        "mcnemar_h1": "b>c detects preference for level_a slot/content",
        "outcome_target": outcome_target,
    }
    if extra:
        merged.update(extra)
    return TestResult(
        hypothesis_id="H1",
        test_id=test_id,
        description=description,
        level="row",
        family_name=family_name,
        n1=b + c,
        k1=b,
        k2=c,
        statistic=stat,
        p_raw=pval,
        effect=(b - c) / n_pairs if n_pairs else float("nan"),
        extra=merged,
    )
