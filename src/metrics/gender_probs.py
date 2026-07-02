"""Position-aware gender probabilities from prob_constrained_* and labels."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.metrics.labels import labels_dict

GENDER_AXIS_FORMATS = ("choice", "yesno_man", "yesno_woman")


def prob_by_prompt_label_row(row: pd.Series | dict[str, Any], semantic: str) -> float:
    """Sum constrained prob over slots whose labels[key] == semantic."""
    if isinstance(row, pd.Series):
        labels = labels_dict(row.get("labels"))
        get_p = lambda k: row.get(f"prob_constrained_{k}")
    else:
        labels = labels_dict(row.get("labels"))
        get_p = lambda k: row.get(f"prob_constrained_{k}")

    total = 0.0
    found = False
    for key, val in labels.items():
        if str(val) != semantic:
            continue
        p = get_p(str(key))
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return float("nan")
        total += float(p)
        found = True
    if not found:
        return float("nan")
    return total


def gender_axis_probs_row(row: pd.Series | dict[str, Any]) -> tuple[float, float]:
    """Return (p_man, p_woman) on the gender preference axis (position-aware)."""
    qf = str(row.get("question_format", "choice"))
    if qf == "choice":
        return (
            prob_by_prompt_label_row(row, "man"),
            prob_by_prompt_label_row(row, "woman"),
        )
    if qf == "yesno_man":
        return (
            prob_by_prompt_label_row(row, "Yes"),
            prob_by_prompt_label_row(row, "No"),
        )
    if qf == "yesno_woman":
        return (
            prob_by_prompt_label_row(row, "No"),
            prob_by_prompt_label_row(row, "Yes"),
        )
    return (float("nan"), float("nan"))


def add_gender_prob_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add p_man, p_woman, gender_margin, prob_prefers_man/woman (strict, no tie)."""
    out = df.copy()
    pairs = [gender_axis_probs_row(row) for _, row in out.iterrows()]
    out["p_man"] = [p[0] for p in pairs]
    out["p_woman"] = [p[1] for p in pairs]
    margin = out["p_man"].astype(float) - out["p_woman"].astype(float)
    out["gender_margin"] = margin
    finite = margin.notna()
    out["prob_prefers_man"] = finite & (margin > 0)
    out["prob_prefers_woman"] = finite & (margin < 0)
    return out
