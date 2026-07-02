"""Paired McNemar tests across evidence_shift levels (pair key: id)."""

from __future__ import annotations

import pandas as pd

from src.metrics.labels import gender_axis_frame
from src.metrics.stats_tests import (
    TestResult,
    mcnemar_from_discordant,
    mcnemar_greater_discordant,
)
from src.metrics.h3_v1.labels import PAIR_KEY

PROB_TARGET = "prob_constrained"
MAN_COL = "prob_prefers_man"
WOMAN_COL = "prob_prefers_woman"


def _binary_pivot_counts(
    df: pd.DataFrame,
    *,
    index_cols: tuple[str, ...],
    pivot_col: str,
    level_a: str,
    level_b: str,
    success_col: str,
    failure_col: str,
) -> tuple[int, int, int, int, int] | None:
    """
    McNemar discordance on a binary success column across pivot levels.

    b: failure@a & success@b  (TOWARD success as we go a -> b)
    c: success@a & failure@b  (AWAY from success as we go a -> b)
  Pairs valid only when each side picked man or woman (not abstain / tie on prob).
    """
    sub = gender_axis_frame(df)
    if not len(sub) or pivot_col not in sub.columns:
        return None
    if not all(c in sub.columns for c in index_cols):
        return None
    if success_col not in sub.columns or failure_col not in sub.columns:
        return None

    wide_s = sub.pivot_table(
        index=list(index_cols),
        columns=pivot_col,
        values=success_col,
        aggfunc="first",
    )
    wide_f = sub.pivot_table(
        index=list(index_cols),
        columns=pivot_col,
        values=failure_col,
        aggfunc="first",
    )
    if level_a not in wide_s.columns or level_b not in wide_s.columns:
        return None

    s0 = wide_s[level_a].astype("boolean").fillna(False).to_numpy(dtype=bool)
    f0 = wide_f[level_a].astype("boolean").fillna(False).to_numpy(dtype=bool)
    s1 = wide_s[level_b].astype("boolean").fillna(False).to_numpy(dtype=bool)
    f1 = wide_f[level_b].astype("boolean").fillna(False).to_numpy(dtype=bool)

    valid = (s0 | f0) & (s1 | f1) & ~(s0 & f0) & ~(s1 & f1)
    if not valid.any():
        return None

    # b = TOWARD success (failure@a & success@b); c = AWAY (success@a & failure@b).
    # This matches mcnemar_b_meaning, the one-sided test (b>c => shift toward axis),
    # and effect=(b-c)/n (positive => toward highlighted gender).
    b = int((f0 & s1 & valid).sum())
    c = int((s0 & f1 & valid).sum())
    a = int((s0 & s1 & valid).sum())
    d = int((f0 & f1 & valid).sum())
    n = int(valid.sum())
    return a, b, c, d, n


def mcnemar_evidence_shift_result(
    df: pd.DataFrame,
    *,
    level_a: str,
    level_b: str,
    gender_axis: str,
    test_id: str,
    description: str,
    family_name: str | None,
    extra: dict | None = None,
    one_sided_toward: str | None = None,
    man_col: str = "prefers_man",
    woman_col: str = "prefers_woman",
    outcome_target: str = "choice",
) -> TestResult | None:
    """
    Paired McNemar on evidence_shift pivot (index=id).

    one_sided_toward: "man" or "woman" — evidence at level_b should increase that gender.
    None → two-sided discordance (man vs woman evidence post-hoc).
    """
    if gender_axis == "man":
        success_col, failure_col = man_col, woman_col
        toward_b = "failure@a & success@b"
        toward_c = "success@a & failure@b"
    elif gender_axis == "woman":
        success_col, failure_col = woman_col, man_col
        toward_b = "failure@a & success@b"
        toward_c = "success@a & failure@b"
    else:
        raise ValueError(f"gender_axis must be man|woman, got {gender_axis!r}")

    counts = _binary_pivot_counts(
        df,
        index_cols=PAIR_KEY,
        pivot_col="evidence_shift",
        level_a=level_a,
        level_b=level_b,
        success_col=success_col,
        failure_col=failure_col,
    )
    if counts is None:
        return None
    a, b, c, d, n_pairs = counts

    if one_sided_toward == gender_axis:
        if b + c == 0:
            stat, pval = 0.0, 1.0
        else:
            stat, pval = mcnemar_greater_discordant(b, c)
        h1 = f"b>c: shift toward {gender_axis} from {level_a} to {level_b}"
    elif one_sided_toward is None:
        if b + c == 0:
            stat, pval = 0.0, 1.0
        else:
            stat, pval = mcnemar_from_discordant(b, c)
        h1 = f"two-sided discordance {level_a} vs {level_b} on {gender_axis} axis"
    else:
        raise ValueError(f"one_sided_toward={one_sided_toward!r} mismatches gender_axis")

    merged = {
        "mcnemar_a_both_success": a,
        "discordant_b": b,
        "discordant_c": c,
        "mcnemar_d_both_failure": d,
        "n_pairs": n_pairs,
        "evidence_a": level_a,
        "evidence_b": level_b,
        "gender_axis": gender_axis,
        "mcnemar_h1": h1,
        "mcnemar_b_meaning": toward_b,
        "mcnemar_c_meaning": toward_c,
        "outcome_target": outcome_target,
        "pair_key": list(PAIR_KEY),
    }
    if extra:
        merged.update(extra)

    # marginal success rate at each level: a=both success, b=toward (success@b only),
    # c=away (success@a only) -> P(success@a)=(a+c)/n, P(success@b)=(a+b)/n
    rate_a = (a + c) / n_pairs if n_pairs else float("nan")
    rate_b_side = (a + b) / n_pairs if n_pairs else float("nan")
    return TestResult(
        hypothesis_id="H3",
        test_id=test_id,
        description=description,
        level="row",
        family_name=family_name,
        n1=n_pairs,
        k1=b,
        k2=c,
        p1=rate_a,
        p2=rate_b_side,
        statistic=stat,
        p_raw=pval,
        effect=(b - c) / n_pairs if n_pairs else float("nan"),
        extra=merged,
    )


def mcnemar_evidence_prob_result(
    df: pd.DataFrame,
    *,
    level_a: str,
    level_b: str,
    gender_axis: str,
    test_id: str,
    description: str,
    family_name: str | None,
    extra: dict | None = None,
    one_sided_toward: str | None = None,
) -> TestResult | None:
    man_col = MAN_COL
    woman_col = WOMAN_COL
    return mcnemar_evidence_shift_result(
        df,
        level_a=level_a,
        level_b=level_b,
        gender_axis=gender_axis,
        test_id=test_id,
        description=description,
        family_name=family_name,
        extra=extra,
        one_sided_toward=one_sided_toward,
        man_col=man_col,
        woman_col=woman_col,
        outcome_target=PROB_TARGET,
    )
