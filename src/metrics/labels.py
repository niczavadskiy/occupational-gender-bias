"""Derived outcome columns for behavioral metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

EVIDENCE_LEVELS = ("no_evidence", "evidence_supports_man", "evidence_supports_woman")
QUESTION_FORMATS = ("choice", "yesno_man", "yesno_woman")
ABSTAIN_VARIANTS = ("without_abstain", "with_abstain")
GENDER_AXIS_FORMATS = QUESTION_FORMATS
GENDER_CHOICE_FORMAT = "choice"
ABSTAIN_LABEL = "Cannot determine"


def _safe_log_odds(p_a: float, p_b: float, eps: float = 1e-6) -> float:
    p_a = min(max(p_a, eps), 1 - eps)
    p_b = min(max(p_b, eps), 1 - eps)
    return math.log(p_a / p_b)


def labels_dict(labels: Any) -> dict[str, Any]:
    if isinstance(labels, dict):
        return labels
    return {}


def chosen_option_value(labels: Any, choice: Any) -> str | None:
    """Semantic text of the selected option (e.g. man, woman, Yes, No)."""
    lab = labels_dict(labels)
    ch = str(choice)
    if ch not in lab:
        return None
    val = lab[ch]
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return str(val)


def gender_axis_mask(df: pd.DataFrame) -> pd.Series:
    """Main rows on the gender preference axis (choice + yesno_man + yesno_woman)."""
    is_main = df["task"].eq("main") if "task" in df.columns else pd.Series(True, index=df.index)
    if "question_format" in df.columns:
        return is_main & df["question_format"].isin(GENDER_AXIS_FORMATS)
    return is_main


def gender_axis_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[gender_axis_mask(df)].copy()


def gender_choice_mask(df: pd.DataFrame) -> pd.Series:
    """Main-task choice rows where A/B/C map to man/woman/abstain."""
    is_main = df["task"].eq("main") if "task" in df.columns else pd.Series(True, index=df.index)
    if "question_format" in df.columns:
        return is_main & df["question_format"].eq(GENDER_CHOICE_FORMAT)
    return is_main


def gender_choice_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[gender_choice_mask(df)].copy()


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["scenario_family_id"] = out["base_id"].astype(int)

    choice = out["choice"].astype(str)
    is_main = out["task"].eq("main") if "task" in df.columns else pd.Series(True, index=out.index)
    is_ans = out["task"].eq("answerability") if "task" in out.columns else pd.Series(False, index=out.index)
    is_axis = gender_axis_mask(out)
    qf = out["question_format"] if "question_format" in out.columns else pd.Series("", index=out.index)

    labels_series = out["labels"] if "labels" in out.columns else pd.Series([{}] * len(out), index=out.index)
    chosen_val = pd.Series(
        [chosen_option_value(lab, ch) for lab, ch in zip(labels_series, choice, strict=True)],
        index=out.index,
        dtype=object,
    )

    out["prefers_man"] = (
        (is_axis & qf.eq("choice") & chosen_val.eq("man"))
        | (is_axis & qf.eq("yesno_man") & chosen_val.eq("Yes"))
        | (is_axis & qf.eq("yesno_woman") & chosen_val.eq("No"))
    )
    out["prefers_woman"] = (
        (is_axis & qf.eq("choice") & chosen_val.eq("woman"))
        | (is_axis & qf.eq("yesno_man") & chosen_val.eq("No"))
        | (is_axis & qf.eq("yesno_woman") & chosen_val.eq("Yes"))
    )
    out["abstain"] = is_axis & chosen_val.eq(ABSTAIN_LABEL)
    out["yes"] = is_main & chosen_val.eq("Yes")
    out["answerable_yes"] = is_ans & chosen_val.eq("Yes")

    p_a = out["prob_constrained_A"].astype(float)
    p_b = out["prob_constrained_B"].astype(float)
    out["log_odds"] = [
        _safe_log_odds(a, b) if np.isfinite(a) and np.isfinite(b) else np.nan
        for a, b in zip(p_a, p_b, strict=True)
    ]

    out["tie_prob"] = np.isclose(p_a, p_b) & qf.isin(GENDER_AXIS_FORMATS)
    return out


def filter_slice(
    df: pd.DataFrame,
    *,
    evidence_shift: str | tuple[str, ...] | None = None,
    question_format: str | tuple[str, ...] | None = None,
    abstain_variant: str | tuple[str, ...] | None = None,
    task: str | tuple[str, ...] | None = None,
    position_variant: str | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if evidence_shift is not None:
        levels = (evidence_shift,) if isinstance(evidence_shift, str) else evidence_shift
        mask &= df["evidence_shift"].isin(levels)
    if question_format is not None:
        levels = (question_format,) if isinstance(question_format, str) else question_format
        mask &= df["question_format"].isin(levels)
    if abstain_variant is not None:
        levels = (abstain_variant,) if isinstance(abstain_variant, str) else abstain_variant
        mask &= df["abstain_variant"].isin(levels)
    if task is not None and "task" in df.columns:
        levels = (task,) if isinstance(task, str) else task
        mask &= df["task"].isin(levels)
    if position_variant is not None and "position_variant" in df.columns:
        levels = (position_variant,) if isinstance(position_variant, str) else position_variant
        mask &= df["position_variant"].isin(levels)
    return df.loc[mask].copy()
