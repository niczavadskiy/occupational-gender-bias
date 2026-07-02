"""Descriptive rate tables for H3 v1."""

from __future__ import annotations

import pandas as pd

from src.metrics.labels import gender_axis_frame
from src.metrics.stats_tests import proportion_summary

from src.metrics.h3_v1.labels import filter_slice
from src.metrics.h3_v1.load import EVIDENCE_LEVELS


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
        "mean_p_man": float(df["p_man"].mean()) if "p_man" in df.columns and n else float("nan"),
        "mean_p_woman": float(df["p_woman"].mean()) if "p_woman" in df.columns and n else float("nan"),
    }


def build_rates_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    base = gender_axis_frame(df)
    rows.append(_rate_row(base, "overall", {}))

    for ev in EVIDENCE_LEVELS:
        sub = filter_slice(base, evidence_shift=ev)
        rows.append(_rate_row(sub, "by_evidence", {"evidence_shift": ev}))

    if "context_order" in base.columns:
        for ctx in sorted(base["context_order"].astype(str).unique()):
            for ev in EVIDENCE_LEVELS:
                sub = filter_slice(base, context_order=ctx, evidence_shift=ev)
                rows.append(
                    _rate_row(
                        sub,
                        "by_context_evidence",
                        {"context_order": ctx, "evidence_shift": ev},
                    )
                )

    if "abstain_variant" in base.columns:
        for ab in sorted(base["abstain_variant"].astype(str).unique()):
            for ev in EVIDENCE_LEVELS:
                sub = filter_slice(base, abstain_variant=ab, evidence_shift=ev)
                rows.append(
                    _rate_row(
                        sub,
                        "by_abstain_evidence",
                        {"abstain_variant": ab, "evidence_shift": ev},
                    )
                )

    return pd.DataFrame(rows)


def build_evidence_flip_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Per-id choice flips between no_evidence and highlight conditions."""
    from src.metrics.h3_v1.labels import PAIR_KEY

    sub = gender_axis_frame(df)
    rows: list[dict] = []

    for ev_b, axis in (("man", "man"), ("woman", "woman")):
        wide_man = sub.pivot_table(
            index=list(PAIR_KEY),
            columns="evidence_shift",
            values="prefers_man",
            aggfunc="first",
        )
        wide_woman = sub.pivot_table(
            index=list(PAIR_KEY),
            columns="evidence_shift",
            values="prefers_woman",
            aggfunc="first",
        )
        if "no_evidence" not in wide_man.columns or ev_b not in wide_man.columns:
            continue

        m0 = wide_man["no_evidence"].astype("boolean").fillna(False).to_numpy(dtype=bool)
        w0 = wide_woman["no_evidence"].astype("boolean").fillna(False).to_numpy(dtype=bool)
        m1 = wide_man[ev_b].astype("boolean").fillna(False).to_numpy(dtype=bool)
        w1 = wide_woman[ev_b].astype("boolean").fillna(False).to_numpy(dtype=bool)
        valid = (m0 | w0) & (m1 | w1) & ~(m0 & w0) & ~(m1 & w1)
        n = int(valid.sum())
        if n == 0:
            continue

        if axis == "man":
            toward = int((~m0 & m1 & valid).sum())
            away = int((m0 & ~m1 & valid).sum())
        else:
            toward = int((~w0 & w1 & valid).sum())
            away = int((w0 & ~w1 & valid).sum())

        rows.append(
            {
                "comparison": f"no_evidence_vs_{ev_b}",
                "gender_axis": axis,
                "n_pairs": n,
                "shift_toward": toward,
                "shift_away": away,
                "rate_toward": toward / n,
                "rate_away": away / n,
                "net_rate": (toward - away) / n,
            }
        )

    return pd.DataFrame(rows)
