"""Derived columns for v1 per_item rows (no base_id / evidence_shift)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.metrics.gender_probs import add_gender_prob_columns
from src.metrics.labels import ABSTAIN_LABEL, chosen_option_value, gender_axis_mask


def _safe_log_odds(p_a: float, p_b: float, eps: float = 1e-6) -> float:
    p_a = min(max(p_a, eps), 1 - eps)
    p_b = min(max(p_b, eps), 1 - eps)
    return math.log(p_a / p_b)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "base_id" in out.columns:
        out["scenario_family_id"] = out["base_id"].astype(int)
    elif "soc" in out.columns and "onet_action" in out.columns:
        out["scenario_family_id"] = (
            out["soc"].astype(str) + "|" + out["onet_action"].astype(str)
        )
    else:
        out["scenario_family_id"] = out["id"].astype(str)

    choice = out["choice"].astype(str)
    is_axis = gender_axis_mask(out)
    labels_series = out["labels"] if "labels" in out.columns else pd.Series([{}] * len(out), index=out.index)
    chosen_val = pd.Series(
        [chosen_option_value(lab, ch) for lab, ch in zip(labels_series, choice, strict=True)],
        index=out.index,
        dtype=object,
    )

    out["prefers_man"] = is_axis & chosen_val.eq("man")
    out["prefers_woman"] = is_axis & chosen_val.eq("woman")
    out["abstain"] = is_axis & chosen_val.eq(ABSTAIN_LABEL)

    p_a = out["prob_constrained_A"].astype(float)
    p_b = out["prob_constrained_B"].astype(float)
    out["log_odds"] = [
        _safe_log_odds(a, b) if np.isfinite(a) and np.isfinite(b) else np.nan
        for a, b in zip(p_a, p_b, strict=True)
    ]
    qf = out["question_format"] if "question_format" in out.columns else pd.Series("", index=out.index)
    out["tie_prob"] = np.isclose(p_a, p_b) & qf.eq("choice")
    out = add_gender_prob_columns(out)
    return out


def filter_slice(
    df: pd.DataFrame,
    *,
    question_format: str | tuple[str, ...] | None = None,
    abstain_variant: str | tuple[str, ...] | None = None,
    position_variant: str | tuple[str, ...] | None = None,
    context_order: str | None = None,
    soc_major_title: str | None = None,
    onet_action: str | None = None,
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if question_format is not None:
        levels = (question_format,) if isinstance(question_format, str) else question_format
        mask &= df["question_format"].isin(levels)
    if abstain_variant is not None:
        levels = (abstain_variant,) if isinstance(abstain_variant, str) else abstain_variant
        mask &= df["abstain_variant"].isin(levels)
    if position_variant is not None and "position_variant" in df.columns:
        levels = (position_variant,) if isinstance(position_variant, str) else position_variant
        mask &= df["position_variant"].isin(levels)
    if context_order is not None and "context_order" in df.columns:
        mask &= df["context_order"].astype(str) == str(context_order)
    if soc_major_title is not None:
        mask &= df["soc_major_title"].astype(str) == str(soc_major_title)
    if onet_action is not None:
        mask &= df["onet_action"].astype(str) == str(onet_action)
    return df.loc[mask].copy()
