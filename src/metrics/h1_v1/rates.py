"""Descriptive rate tables for v1 H1."""

from __future__ import annotations

import pandas as pd

from src.metrics.labels import gender_axis_frame
from src.metrics.stats_tests import proportion_summary

from src.metrics.h1_v1.labels import filter_slice


def _rate_row(df: pd.DataFrame, label: str, group_cols: dict) -> dict:
    n = len(df)
    k_man = int(df["prefers_man"].sum()) if n else 0
    k_woman = int(df["prefers_woman"].sum()) if n else 0
    k_abstain = int(df["abstain"].sum()) if n else 0
    sm = proportion_summary(k_man, n)
    return {
        "slice": label,
        **group_cols,
        "n": n,
        "k_man": k_man,
        "rate_man": sm["p"],
        "rate_man_ci_low": sm["ci_low"],
        "rate_man_ci_high": sm["ci_high"],
        "k_woman": k_woman,
        "rate_woman": k_woman / n if n else float("nan"),
        "k_abstain": k_abstain,
        "rate_abstain": k_abstain / n if n else float("nan"),
        "mean_log_odds": float(df["log_odds"].mean()) if n else float("nan"),
    }


def build_rates_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    base = gender_axis_frame(df)
    rows.append(_rate_row(base, "overall_gender_axis", {}))

    if "abstain_variant" in base.columns:
        for ab in sorted(base["abstain_variant"].astype(str).unique()):
            sub = filter_slice(base, abstain_variant=ab)
            rows.append(_rate_row(sub, "by_abstain", {"abstain_variant": ab}))

    if "soc_major_title" in base.columns:
        for title in sorted(base["soc_major_title"].astype(str).unique()):
            sub = filter_slice(base, soc_major_title=title)
            rows.append(_rate_row(sub, "by_soc_major", {"soc_major_title": title}))

    return pd.DataFrame(rows)


def build_rates_by_scenario(df: pd.DataFrame) -> pd.DataFrame:
    sub = gender_axis_frame(df)
    rows: list[dict] = []
    group_cols = ["soc", "profession", "soc_major_title", "onet_action", "abstain_variant"]
    present = [c for c in group_cols if c in sub.columns]
    for keys, g in sub.groupby(present, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(present, keys, strict=True))
        rec["k_man"] = int(g["prefers_man"].sum())
        rec["n"] = len(g)
        rec["rate_man"] = rec["k_man"] / rec["n"] if rec["n"] else float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)
