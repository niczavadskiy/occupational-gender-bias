"""Semantic constrained probabilities from per_item rows (position-aware)."""

from __future__ import annotations

from typing import Any

ABSTAIN_LABEL = "Cannot determine"
GENDER_AXIS_FORMATS = ("choice", "yesno_man", "yesno_woman")


def labels_dict(labels: Any) -> dict[str, Any]:
    if isinstance(labels, dict):
        return labels
    return {}


def prob_constrained_for_key(row: dict[str, Any], option_key: str) -> float:
    p = row.get(f"prob_constrained_{option_key}")
    if p is None:
        raise ValueError(
            f"Row {row.get('id')!r}: missing prob_constrained_{option_key}"
        )
    return float(p)


def prob_by_prompt_label(row: dict[str, Any], semantic: str) -> float:
    """Sum constrained prob over prompt slots whose labels[key] == semantic."""
    labels = labels_dict(row.get("labels"))
    total = 0.0
    found = False
    for key, val in labels.items():
        if str(val) != semantic:
            continue
        total += prob_constrained_for_key(row, str(key))
        found = True
    if not found:
        raise ValueError(
            f"Row {row.get('id')!r}: no prompt slot with label {semantic!r} in {labels}"
        )
    return total


def chosen_option_value(row: dict[str, Any]) -> str:
    labels = labels_dict(row.get("labels"))
    choice = str(row["choice"])
    if choice not in labels:
        raise ValueError(f"Row {row.get('id')!r}: choice {choice!r} not in labels")
    return str(labels[choice])


def gender_axis_probs(row: dict[str, Any]) -> tuple[float, float]:
    """Return (p_man, p_woman) on the gender preference axis (position-aware)."""
    qf = str(row["question_format"])
    if qf == "choice":
        return (
            prob_by_prompt_label(row, "man"),
            prob_by_prompt_label(row, "woman"),
        )
    if qf == "yesno_man":
        return (
            prob_by_prompt_label(row, "Yes"),
            prob_by_prompt_label(row, "No"),
        )
    if qf == "yesno_woman":
        return (
            prob_by_prompt_label(row, "No"),
            prob_by_prompt_label(row, "Yes"),
        )
    raise ValueError(f"unsupported question_format for gender axis: {qf!r}")


def abstain_prob(row: dict[str, Any]) -> float:
    return prob_by_prompt_label(row, ABSTAIN_LABEL)


def is_abstain_choice(row: dict[str, Any]) -> bool:
    return chosen_option_value(row) == ABSTAIN_LABEL


def stereotype_consistent_prob(
    row: dict[str, Any],
    canonical_labels: dict[str, Any],
    option_labels: dict[str, str],
) -> float:
    """Sum P(slot) over inference slots whose semantic value is stereotype-consistent."""
    st_values = {
        str(canonical_labels[k])
        for k, v in option_labels.items()
        if v == "stereotype_consistent" and k in canonical_labels
    }
    if not st_values:
        return 0.0
    total = 0.0
    for key, val in labels_dict(row.get("labels")).items():
        if str(val) in st_values:
            total += prob_constrained_for_key(row, str(key))
    return total


def choice_stereotype_label(
    row: dict[str, Any],
    canonical_labels: dict[str, Any],
    option_labels: dict[str, str],
) -> str:
    """Map inference choice to annotation stereotype class via semantic option text."""
    chosen = chosen_option_value(row)
    for canon_key, sem in canonical_labels.items():
        if str(sem) == chosen:
            lbl = option_labels.get(str(canon_key))
            if lbl is not None:
                return str(lbl)
    raise ValueError(
        f"Row {row.get('id')!r}: choice semantic {chosen!r} not in canonical labels {canonical_labels}"
    )


def safe_log_odds(p_num: float, p_den: float, *, eps: float = 1e-6) -> float:
    import math

    p_num = min(max(p_num, eps), 1.0)
    p_den = min(max(p_den, eps), 1.0)
    return math.log(p_num / p_den)
