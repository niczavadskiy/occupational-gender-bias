"""Order-flip and first-shown pick rates (lechmazur/position_bias style)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.metrics.labels import gender_axis_frame

from src.metrics.h1_v1.labels import filter_slice

TWO_OPT_POSITIONS = ("p0", "p1")


def _rate(k: int, n: int) -> float:
    return k / n if n else float("nan")


def order_flip_from_mcnemar(a: int, b: int, c: int, d: int, n_pairs: int) -> dict[str, float | int]:
    """Canonical gender changes between paired views; stable = same gender both views."""
    disc = b + c
    return {
        "order_flip": _rate(disc, n_pairs),
        "stable_rate": _rate(a + d, n_pairs),
        "n_discordant": disc,
        "discordant_b_rate": _rate(b, disc),
        "discordant_c_rate": _rate(c, disc),
    }


def first_slot_pick_rate_prob(
    df: pd.DataFrame,
    *,
    context_order: str | None = None,
) -> dict[str, Any]:
    """Mean prob_constrained_A over p0+p1 (slot A displayed first)."""
    sub = filter_slice(df, abstain_variant="without_abstain", position_variant=TWO_OPT_POSITIONS)
    if context_order is not None:
        sub = filter_slice(sub, context_order=context_order)
    sub = gender_axis_frame(sub)
    if not len(sub) or "prob_constrained_A" not in sub.columns:
        return {
            "first_shown_pick": float("nan"),
            "first_shown_lift": float("nan"),
            "first_shown_n": 0,
        }
    p = sub["prob_constrained_A"].astype(float)
    p = p[p.notna()]
    n = len(p)
    pick = float(p.mean()) if n else float("nan")
    return {
        "first_shown_pick": pick,
        "first_shown_lift": pick - 0.5 if n else float("nan"),
        "first_shown_n": n,
    }


def first_narrative_pick_rate_prob(
    df: pd.DataFrame,
    *,
    position_variant: str,
    abstain_variant: str = "without_abstain",
) -> dict[str, Any]:
    """Mean prob mass on first-mentioned gender (p_man@mf, p_woman@wf), pooled."""
    sub = filter_slice(
        df,
        position_variant=position_variant,
        abstain_variant=abstain_variant,
    )
    sub = gender_axis_frame(sub)
    if not len(sub) or "context_order" not in sub.columns:
        return {
            "first_shown_pick": float("nan"),
            "first_shown_lift": float("nan"),
            "first_shown_n": 0,
            "first_shown_pick_mf": float("nan"),
            "first_shown_pick_wf": float("nan"),
        }
    mf = sub["context_order"].astype(str).eq("man_first")
    wf = sub["context_order"].astype(str).eq("woman_first")
    pm_mf = sub.loc[mf, "p_man"].astype(float)
    pw_wf = sub.loc[wf, "p_woman"].astype(float)
    pm_mf = pm_mf[pm_mf.notna()]
    pw_wf = pw_wf[pw_wf.notna()]
    pick_mf = float(pm_mf.mean()) if len(pm_mf) else float("nan")
    pick_wf = float(pw_wf.mean()) if len(pw_wf) else float("nan")
    pooled = pd.concat([pm_mf, pw_wf], ignore_index=True)
    n = len(pooled)
    pick = float(pooled.mean()) if n else float("nan")
    return {
        "first_shown_pick": pick,
        "first_shown_lift": pick - 0.5 if n else float("nan"),
        "first_shown_n": n,
        "first_shown_pick_mf": pick_mf,
        "first_shown_pick_wf": pick_wf,
    }


def first_slot_pick_rate(
    df: pd.DataFrame,
    *,
    context_order: str | None = None,
) -> dict[str, Any]:
    """
    P(choice=A) over p0+p1 prompts (slot A is always displayed first).

    Analogous to position_bias benchmark First-Shown Pick for option order.
    """
    sub = filter_slice(df, abstain_variant="without_abstain", position_variant=TWO_OPT_POSITIONS)
    if context_order is not None:
        sub = filter_slice(sub, context_order=context_order)
    sub = gender_axis_frame(sub)
    n = len(sub)
    if n == 0 or "choice" not in sub.columns:
        return {
            "first_shown_pick": float("nan"),
            "first_shown_lift": float("nan"),
            "first_shown_k": 0,
            "first_shown_n": 0,
        }
    k = int(sub["choice"].astype(str).eq("A").sum())
    pick = _rate(k, n)
    return {
        "first_shown_pick": pick,
        "first_shown_lift": pick - 0.5 if n else float("nan"),
        "first_shown_k": k,
        "first_shown_n": n,
    }


def first_narrative_pick_rate(
    df: pd.DataFrame,
    *,
    position_variant: str,
    abstain_variant: str = "without_abstain",
) -> dict[str, Any]:
    """
    P(choose first-mentioned gender in scenario text), pooled over mf and wf prompts.

    mf: first in narrative = man; wf: first in narrative = woman.
    """
    sub = filter_slice(
        df,
        position_variant=position_variant,
        abstain_variant=abstain_variant,
    )
    sub = gender_axis_frame(sub)
    if not len(sub) or "context_order" not in sub.columns:
        return {
            "first_shown_pick": float("nan"),
            "first_shown_lift": float("nan"),
            "first_shown_k": 0,
            "first_shown_n": 0,
            "first_shown_pick_mf": float("nan"),
            "first_shown_pick_wf": float("nan"),
        }
    mf = sub["context_order"].astype(str).eq("man_first")
    wf = sub["context_order"].astype(str).eq("woman_first")
    k_mf = int(sub.loc[mf, "prefers_man"].sum()) if mf.any() else 0
    n_mf = int(mf.sum())
    k_wf = int(sub.loc[wf, "prefers_woman"].sum()) if wf.any() else 0
    n_wf = int(wf.sum())
    k = k_mf + k_wf
    n = n_mf + n_wf
    pick = _rate(k, n)
    return {
        "first_shown_pick": pick,
        "first_shown_lift": pick - 0.5 if n else float("nan"),
        "first_shown_k": k,
        "first_shown_n": n,
        "first_shown_pick_mf": _rate(k_mf, n_mf),
        "first_shown_pick_wf": _rate(k_wf, n_wf),
    }


def enrich_position_mcnemar(
    extra: dict[str, Any],
    df: pd.DataFrame,
) -> dict[str, Any]:
    """Attach flip metrics to position p0↔p1 McNemar extra."""
    a = int(extra.get("mcnemar_a_both_man") or 0)
    b = int(extra.get("discordant_b_man_a_woman_b") or 0)
    c = int(extra.get("discordant_c_woman_a_man_b") or 0)
    d = int(extra.get("mcnemar_d_both_woman") or 0)
    n = int(extra.get("n_pairs") or 0)
    out = dict(extra)
    out.update(order_flip_from_mcnemar(a, b, c, d, n))
    out["flip_metric_axis"] = "position_p0_p1"
    out["first_shown_label"] = "choice=A (slot A displayed first)"
    out.update(first_slot_pick_rate(df, context_order=extra.get("context_order")))
    out["discordant_alignment_label"] = "slot_A_pattern (man@p0 & woman@p1)"
    out["outcome_target"] = "choice"
    return out


def enrich_position_mcnemar_prob(
    extra: dict[str, Any],
    df: pd.DataFrame,
) -> dict[str, Any]:
    """Flip metrics for position McNemar on prob argmax (p_man vs p_woman)."""
    a = int(extra.get("mcnemar_a_both_man") or 0)
    b = int(extra.get("discordant_b_man_a_woman_b") or 0)
    c = int(extra.get("discordant_c_woman_a_man_b") or 0)
    d = int(extra.get("mcnemar_d_both_woman") or 0)
    n = int(extra.get("n_pairs") or 0)
    out = dict(extra)
    out.update(order_flip_from_mcnemar(a, b, c, d, n))
    out["flip_metric_axis"] = "position_p0_p1"
    out["first_shown_label"] = "mean prob_constrained_A (slot A)"
    out.update(first_slot_pick_rate_prob(df, context_order=extra.get("context_order")))
    out["discordant_alignment_label"] = "slot_A_pattern (man@p0 & woman@p1)"
    out["outcome_target"] = "prob_constrained"
    return out


def enrich_context_mcnemar(
    extra: dict[str, Any],
    df: pd.DataFrame,
) -> dict[str, Any]:
    """Attach flip metrics to context mf↔wf McNemar extra (p0/p1 without_abstain)."""
    a = int(extra.get("mcnemar_a_both_man") or 0)
    b = int(extra.get("discordant_b_man_a_woman_b") or 0)
    c = int(extra.get("discordant_c_woman_a_man_b") or 0)
    d = int(extra.get("mcnemar_d_both_woman") or 0)
    n = int(extra.get("n_pairs") or 0)
    pos = str(extra.get("position_variant", "p0"))
    av = str(extra.get("abstain_variant", "without_abstain"))
    out = dict(extra)
    out.update(order_flip_from_mcnemar(a, b, c, d, n))
    out["flip_metric_axis"] = "context_mf_wf"
    out["first_shown_label"] = "first-mentioned gender in scenario text"
    out.update(first_narrative_pick_rate(df, position_variant=pos, abstain_variant=av))
    out["discordant_alignment_label"] = "first_in_narrative (man@mf & woman@wf)"
    out["outcome_target"] = "choice"
    return out


def enrich_context_mcnemar_prob(
    extra: dict[str, Any],
    df: pd.DataFrame,
) -> dict[str, Any]:
    """Flip metrics for context McNemar on prob argmax."""
    a = int(extra.get("mcnemar_a_both_man") or 0)
    b = int(extra.get("discordant_b_man_a_woman_b") or 0)
    c = int(extra.get("discordant_c_woman_a_man_b") or 0)
    d = int(extra.get("mcnemar_d_both_woman") or 0)
    n = int(extra.get("n_pairs") or 0)
    pos = str(extra.get("position_variant", "p0"))
    av = str(extra.get("abstain_variant", "without_abstain"))
    out = dict(extra)
    out.update(order_flip_from_mcnemar(a, b, c, d, n))
    out["flip_metric_axis"] = "context_mf_wf"
    out["first_shown_label"] = "mean prob on first-mentioned gender"
    out.update(first_narrative_pick_rate_prob(df, position_variant=pos, abstain_variant=av))
    out["discordant_alignment_label"] = "first_in_narrative (man@mf & woman@wf)"
    out["outcome_target"] = "prob_constrained"
    return out


def flip_rates_rows(results: list) -> list[dict[str, Any]]:
    """Flatten flip-rate fields from McNemar tests that have them."""
    rows: list[dict[str, Any]] = []
    for r in results:
        ex = r.extra
        if "order_flip" not in ex:
            continue
        rows.append(
            {
                "test_id": r.test_id,
                "axis": ex.get("flip_metric_axis"),
                "context_order": ex.get("context_order"),
                "position_variant": ex.get("position_variant"),
                "abstain_variant": ex.get("abstain_variant"),
                "n_pairs": ex.get("n_pairs"),
                "order_flip": ex.get("order_flip"),
                "stable_rate": ex.get("stable_rate"),
                "n_discordant": ex.get("n_discordant"),
                "discordant_b_rate": ex.get("discordant_b_rate"),
                "discordant_c_rate": ex.get("discordant_c_rate"),
                "first_shown_pick": ex.get("first_shown_pick"),
                "first_shown_lift": ex.get("first_shown_lift"),
                "first_shown_k": ex.get("first_shown_k"),
                "first_shown_n": ex.get("first_shown_n"),
                "first_shown_pick_mf": ex.get("first_shown_pick_mf"),
                "first_shown_pick_wf": ex.get("first_shown_pick_wf"),
                "first_shown_label": ex.get("first_shown_label"),
                "discordant_alignment_label": ex.get("discordant_alignment_label"),
                "outcome_target": ex.get("outcome_target"),
            }
        )
    return rows
