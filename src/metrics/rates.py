"""Rate tables (descriptive)."""

from __future__ import annotations

import pandas as pd

from src.metrics.labels import filter_slice
from src.metrics.stats_tests import proportion_summary


def _rate_row(df: pd.DataFrame, label: str, group_cols: dict) -> dict:
    n = len(df)
    k_man = int(df["prefers_man"].sum()) if n else 0
    k_woman = int(df["prefers_woman"].sum()) if n else 0
    k_abstain = int(df["abstain"].sum()) if n else 0
    sm = proportion_summary(k_man, n)
    row = {
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
    return row


def build_rates_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    rows.append(_rate_row(df, "overall", {}))

    for ev in sorted(df["evidence_shift"].unique()):
        sub = filter_slice(df, evidence_shift=ev)
        rows.append(_rate_row(sub, f"by_evidence", {"evidence_shift": ev}))

    for ab in sorted(df["abstain_variant"].unique()):
        sub = filter_slice(df, abstain_variant=ab)
        rows.append(_rate_row(sub, f"by_abstain", {"abstain_variant": ab}))

    for qf in sorted(df["question_format"].unique()):
        sub = filter_slice(df, question_format=qf)
        rows.append(_rate_row(sub, f"by_format", {"question_format": qf}))

    for ev in sorted(df["evidence_shift"].unique()):
        for ab in sorted(df["abstain_variant"].unique()):
            sub = filter_slice(df, evidence_shift=ev, abstain_variant=ab)
            rows.append(
                _rate_row(
                    sub,
                    "by_evidence_x_abstain",
                    {"evidence_shift": ev, "abstain_variant": ab},
                )
            )

    return pd.DataFrame(rows)


def build_rates_by_family(df: pd.DataFrame, *, question_format: str = "choice") -> pd.DataFrame:
    sub = filter_slice(df, question_format=question_format)
    rows: list[dict] = []
    group_cols = ["base_id", "predicate", "base_context", "evidence_shift", "abstain_variant"]
    for keys, g in sub.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(group_cols, keys, strict=True))
        rec["k_man"] = int(g["prefers_man"].sum())
        rec["n"] = len(g)
        rec["rate_man"] = rec["k_man"] / rec["n"] if rec["n"] else float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)
