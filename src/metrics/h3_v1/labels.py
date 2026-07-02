"""Derived columns and filters for h3_v1 (reuses h1_v1 outcome columns)."""

from __future__ import annotations

from src.metrics.h1_v1.labels import add_derived_columns as _add_h1_derived
from src.metrics.h3_v1.load import EVIDENCE_LEVELS

import pandas as pd


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    return _add_h1_derived(df)


def filter_slice(
    df: pd.DataFrame,
    *,
    evidence_shift: str | tuple[str, ...] | None = None,
    question_format: str | tuple[str, ...] | None = None,
    abstain_variant: str | tuple[str, ...] | None = None,
    position_variant: str | tuple[str, ...] | None = None,
    context_order: str | None = None,
    soc_major_title: str | None = None,
    onet_action: str | None = None,
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


PAIR_KEY = ("id",)
