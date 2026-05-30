"""
H11: gender-axis preference decodable from hidden states.

scenario_family_id == base_id (context × predicate). All evidence variants of
one family share the same train/val/test split.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

H11_QFMT = "choice"
H11_ABSTAIN_DEFAULT = "without_abstain"
ABSTAIN_VARIANTS = ("without_abstain", "with_abstain")
H11_EVIDENCE_ALL = (
    "no_evidence",
    "evidence_supports_man",
    "evidence_supports_woman",
)
ROWS_PER_FAMILY_ALL = 3


class EvidenceMode(str, Enum):
    NO_EVIDENCE = "no_evidence"
    EVIDENCE_MAN = "evidence_supports_man"
    EVIDENCE_WOMAN = "evidence_supports_woman"
    ALL = "all"


EVIDENCE_MODE_LEVELS: dict[EvidenceMode, tuple[str, ...]] = {
    EvidenceMode.NO_EVIDENCE: ("no_evidence",),
    EvidenceMode.EVIDENCE_MAN: ("evidence_supports_man",),
    EvidenceMode.EVIDENCE_WOMAN: ("evidence_supports_woman",),
    EvidenceMode.ALL: H11_EVIDENCE_ALL,
}


@dataclass(frozen=True)
class H11Batch:
    indices: np.ndarray
    example_ids: np.ndarray
    base_ids: np.ndarray
    scenario_family_id: np.ndarray
    evidence_shift: np.ndarray
    y: np.ndarray
    choice: np.ndarray
    predicates: np.ndarray
    base_contexts: np.ndarray
    prob_man: np.ndarray
    prob_woman: np.ndarray
    log_odds: np.ndarray
    prob_margin: np.ndarray
    evidence_mode: EvidenceMode
    abstain_variant: str


def is_h11_row(
    row: dict[str, Any],
    evidence_levels: tuple[str, ...],
    *,
    abstain_variant: str = H11_ABSTAIN_DEFAULT,
) -> bool:
    return (
        row["evidence_shift"] in evidence_levels
        and row["question_format"] == H11_QFMT
        and row["abstain_variant"] == abstain_variant
    )


def choice_to_prefers_man(choice: str) -> int:
    if choice == "A":
        return 1
    if choice == "B":
        return 0
    if choice == "C":
        return -1
    raise ValueError(f"H11 expects choice A, B, or C, got {choice!r}")


def _validate_family_structure(base_ids: np.ndarray, evidence_mode: EvidenceMode) -> None:
    unique, counts = np.unique(base_ids, return_counts=True)
    expected = 1 if evidence_mode != EvidenceMode.ALL else ROWS_PER_FAMILY_ALL
    bad = unique[counts != expected]
    if len(bad):
        raise ValueError(
            f"Expected {expected} row(s) per scenario_family for mode={evidence_mode.value}, "
            f"families {bad[:5].tolist()}..."
        )


def build_h11_batch(
    items: list[dict[str, Any]],
    evidence_mode: EvidenceMode = EvidenceMode.NO_EVIDENCE,
    *,
    abstain_variant: str = H11_ABSTAIN_DEFAULT,
) -> H11Batch:
    if abstain_variant not in ABSTAIN_VARIANTS:
        raise ValueError(
            f"abstain_variant must be one of {ABSTAIN_VARIANTS}, got {abstain_variant!r}"
        )
    evidence_levels = EVIDENCE_MODE_LEVELS[evidence_mode]
    indices: list[int] = []
    example_ids: list[int] = []
    base_ids: list[int] = []
    evidence_shifts: list[str] = []
    y_list: list[int] = []
    choices: list[str] = []
    predicates: list[str] = []
    contexts: list[str] = []
    prob_man: list[float] = []
    prob_woman: list[float] = []

    for i, row in enumerate(items):
        if not is_h11_row(row, evidence_levels, abstain_variant=abstain_variant):
            continue
        choice = row["choice"]
        indices.append(i)
        example_ids.append(int(row["example_id"]))
        base_ids.append(int(row["base_id"]))
        evidence_shifts.append(row["evidence_shift"])
        y_list.append(choice_to_prefers_man(choice))
        choices.append(choice)
        predicates.append(row["predicate"])
        contexts.append(row["base_context"])
        prob_man.append(float(row["prob_constrained_A"]))
        prob_woman.append(float(row["prob_constrained_B"]))

    if not indices:
        raise ValueError(
            f"No H11 rows for evidence_mode={evidence_mode.value} "
            f"(levels={evidence_levels})"
        )

    base_ids_arr = np.array(base_ids, dtype=np.int64)
    _validate_family_structure(base_ids_arr, evidence_mode)

    if evidence_mode == EvidenceMode.ALL:
        by_base: dict[int, set[str]] = {}
        for bid, ev in zip(base_ids, evidence_shifts):
            by_base.setdefault(bid, set()).add(ev)
        incomplete = [b for b, evs in by_base.items() if evs != set(H11_EVIDENCE_ALL)]
        if incomplete:
            raise ValueError(
                f"families missing evidence levels: {incomplete[:5]}..."
            )

    prob_man_arr = np.array(prob_man, dtype=np.float64)
    prob_woman_arr = np.array(prob_woman, dtype=np.float64)
    eps = 1e-6
    log_odds = np.log(
        np.clip(prob_man_arr, eps, 1.0) / np.clip(prob_woman_arr, eps, 1.0)
    )
    evidence_arr = np.array(evidence_shifts)

    return H11Batch(
        indices=np.array(indices, dtype=np.int64),
        example_ids=np.array(example_ids, dtype=np.int64),
        base_ids=base_ids_arr,
        scenario_family_id=base_ids_arr.copy(),
        evidence_shift=evidence_arr,
        y=np.array(y_list, dtype=np.int64),
        choice=np.array(choices),
        predicates=np.array(predicates),
        base_contexts=np.array(contexts),
        prob_man=prob_man_arr,
        prob_woman=prob_woman_arr,
        log_odds=log_odds,
        prob_margin=prob_man_arr - prob_woman_arr,
        evidence_mode=evidence_mode,
        abstain_variant=abstain_variant,
    )


def target_vector(batch: H11Batch, target: str) -> np.ndarray:
    if target == "log_odds":
        return batch.log_odds
    if target == "prob_margin":
        return batch.prob_margin
    if target == "choice":
        return batch.y.astype(np.float64)
    raise ValueError(f"unknown target: {target}")


def hard_label_class_balance(y: np.ndarray) -> dict[str, int]:
    return {"prefers_man": int(y.sum()), "prefers_woman": int(len(y) - y.sum())}


def h11_items_table(
    items: list[dict[str, Any]],
    evidence_mode: EvidenceMode = EvidenceMode.ALL,
) -> pd.DataFrame:
    batch = build_h11_batch(items, evidence_mode=evidence_mode)
    rows = [items[i] for i in batch.indices]
    df = pd.DataFrame(rows)
    df["scenario_family_id"] = batch.scenario_family_id
    df["prefers_man"] = batch.y
    df["prob_man"] = batch.prob_man
    df["prob_woman"] = batch.prob_woman
    df["log_odds_man"] = batch.log_odds
    df["prob_margin"] = batch.prob_margin
    return df
