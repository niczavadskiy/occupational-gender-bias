"""
H12: abstain preference decodable from hidden states.

Uses only choice prompts with abstain option enabled (A/B/C).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from probes.h11 import EVIDENCE_MODE_LEVELS, EvidenceMode, H11_EVIDENCE_ALL

H12_QFMT = "choice"
H12_ABSTAIN_VARIANT = "with_abstain"
ROWS_PER_FAMILY_ALL = 3


def h12_pipeline_run_slug(evidence_mode: EvidenceMode) -> str:
    """Directory name under probes/h12/ encoding abstain, evidence, and format."""
    return "_".join((H12_ABSTAIN_VARIANT, evidence_mode.value, H12_QFMT))


@dataclass(frozen=True)
class H12Batch:
    indices: np.ndarray
    example_ids: np.ndarray
    base_ids: np.ndarray
    scenario_family_id: np.ndarray
    evidence_shift: np.ndarray
    choice: np.ndarray
    y_abstain: np.ndarray
    prob_man: np.ndarray
    prob_woman: np.ndarray
    prob_abstain: np.ndarray
    abstain_logit: np.ndarray
    log_prob_abstain: np.ndarray
    evidence_mode: EvidenceMode
    abstain_variant: str


def is_h12_row(row: dict[str, Any], evidence_levels: tuple[str, ...]) -> bool:
    return (
        row["evidence_shift"] in evidence_levels
        and row["question_format"] == H12_QFMT
        and row["abstain_variant"] == H12_ABSTAIN_VARIANT
    )


def _validate_family_structure(base_ids: np.ndarray, evidence_mode: EvidenceMode) -> None:
    unique, counts = np.unique(base_ids, return_counts=True)
    expected = 1 if evidence_mode != EvidenceMode.ALL else ROWS_PER_FAMILY_ALL
    bad = unique[counts != expected]
    if len(bad):
        raise ValueError(
            f"Expected {expected} row(s) per scenario_family for mode={evidence_mode.value}, "
            f"families {bad[:5].tolist()}..."
        )


def build_h12_batch(
    items: list[dict[str, Any]],
    evidence_mode: EvidenceMode = EvidenceMode.NO_EVIDENCE,
) -> H12Batch:
    evidence_levels = EVIDENCE_MODE_LEVELS[evidence_mode]
    indices: list[int] = []
    example_ids: list[int] = []
    base_ids: list[int] = []
    evidence_shifts: list[str] = []
    choices: list[str] = []
    y_abstain: list[int] = []
    prob_man: list[float] = []
    prob_woman: list[float] = []
    prob_abstain: list[float] = []

    for i, row in enumerate(items):
        if not is_h12_row(row, evidence_levels):
            continue
        p_c = row.get("prob_constrained_C")
        if p_c is None:
            raise ValueError(
                "H12 expects with_abstain rows with prob_constrained_C present."
            )
        choice = str(row["choice"])
        indices.append(i)
        example_ids.append(int(row["example_id"]))
        base_ids.append(int(row["base_id"]))
        evidence_shifts.append(str(row["evidence_shift"]))
        choices.append(choice)
        y_abstain.append(1 if choice == "C" else 0)
        prob_man.append(float(row["prob_constrained_A"]))
        prob_woman.append(float(row["prob_constrained_B"]))
        prob_abstain.append(float(p_c))

    if not indices:
        raise ValueError(
            f"No H12 rows for evidence_mode={evidence_mode.value} "
            f"(levels={evidence_levels}, abstain_variant={H12_ABSTAIN_VARIANT})"
        )

    base_ids_arr = np.array(base_ids, dtype=np.int64)
    _validate_family_structure(base_ids_arr, evidence_mode)

    if evidence_mode == EvidenceMode.ALL:
        by_base: dict[int, set[str]] = {}
        for bid, ev in zip(base_ids, evidence_shifts):
            by_base.setdefault(bid, set()).add(ev)
        incomplete = [b for b, evs in by_base.items() if evs != set(H11_EVIDENCE_ALL)]
        if incomplete:
            raise ValueError(f"families missing evidence levels: {incomplete[:5]}...")

    p_a = np.array(prob_man, dtype=np.float64)
    p_b = np.array(prob_woman, dtype=np.float64)
    p_c = np.array(prob_abstain, dtype=np.float64)
    eps = 1e-6
    p_c_clip = np.clip(p_c, eps, 1.0 - eps)
    abstain_logit = np.log(p_c_clip / (1.0 - p_c_clip))
    log_prob_abstain = np.log(np.clip(p_c, eps, 1.0))

    return H12Batch(
        indices=np.array(indices, dtype=np.int64),
        example_ids=np.array(example_ids, dtype=np.int64),
        base_ids=base_ids_arr,
        scenario_family_id=base_ids_arr.copy(),
        evidence_shift=np.array(evidence_shifts),
        choice=np.array(choices),
        y_abstain=np.array(y_abstain, dtype=np.int64),
        prob_man=p_a,
        prob_woman=p_b,
        prob_abstain=p_c,
        abstain_logit=abstain_logit,
        log_prob_abstain=log_prob_abstain,
        evidence_mode=evidence_mode,
        abstain_variant=H12_ABSTAIN_VARIANT,
    )


def target_vector(batch: H12Batch, target: str) -> np.ndarray:
    if target == "abstain_logit":
        return batch.abstain_logit
    if target == "log_prob_abstain":
        return batch.log_prob_abstain
    if target == "choice_abstain":
        return batch.y_abstain.astype(np.float64)
    raise ValueError(f"unknown H12 target: {target}")
