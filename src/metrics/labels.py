"""Derived outcome columns for behavioral metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

EVIDENCE_LEVELS = ("no_evidence", "evidence_supports_man", "evidence_supports_woman")
QUESTION_FORMATS = ("choice", "yesno_man", "yesno_woman")
ABSTAIN_VARIANTS = ("without_abstain", "with_abstain")


def _safe_log_odds(p_a: float, p_b: float, eps: float = 1e-6) -> float:
    p_a = min(max(p_a, eps), 1 - eps)
    p_b = min(max(p_b, eps), 1 - eps)
    return math.log(p_a / p_b)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["scenario_family_id"] = out["base_id"].astype(int)

    choice = out["choice"].astype(str)
    out["prefers_man"] = choice == "A"
    out["prefers_woman"] = choice == "B"
    out["abstain"] = choice == "C"
    out["yes"] = choice == "A"  # Yes always on A for yesno_* formats

    p_a = out["prob_constrained_A"].astype(float)
    p_b = out["prob_constrained_B"].astype(float)
    out["log_odds"] = [
        _safe_log_odds(a, b) if np.isfinite(a) and np.isfinite(b) else np.nan
        for a, b in zip(p_a, p_b, strict=True)
    ]

    out["tie_prob"] = np.isclose(p_a, p_b) & out["question_format"].isin(
        ("choice", "yesno_man", "yesno_woman")
    )
    return out


def filter_slice(
    df: pd.DataFrame,
    *,
    evidence_shift: str | tuple[str, ...] | None = None,
    question_format: str | tuple[str, ...] | None = None,
    abstain_variant: str | tuple[str, ...] | None = None,
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
    return df.loc[mask].copy()
