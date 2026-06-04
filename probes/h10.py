"""
H10: stereotype-consistency signal decodable from hidden states.

Targets are built by joining inference rows with external annotations by example_id.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from probes.h11 import ABSTAIN_VARIANTS, EVIDENCE_MODE_LEVELS, EvidenceMode, H11_EVIDENCE_ALL
from probes.paths import REPO_ROOT

H10_QUESTION_FORMATS = ("choice", "yesno_man", "yesno_woman")
DEFAULT_ANNOTATION_RUN = "2026-06-02_13-10-31_anthropic_claude-opus-4.8"


def h10_pipeline_run_slug(
    abstain_variant: str,
    evidence_mode: EvidenceMode,
    question_formats: tuple[str, ...],
) -> str:
    """Directory name under probes/h10/ encoding abstain, evidence, and formats."""
    if abstain_variant not in ABSTAIN_VARIANTS:
        raise ValueError(f"abstain_variant must be one of {ABSTAIN_VARIANTS}, got {abstain_variant!r}")
    unknown = [q for q in question_formats if q not in H10_QUESTION_FORMATS]
    if unknown:
        raise ValueError(f"Unknown question_formats {unknown}; allowed: {H10_QUESTION_FORMATS}")
    formats = [f for f in H10_QUESTION_FORMATS if f in question_formats]
    if not formats:
        raise ValueError("question_formats must include at least one format")
    return "_".join((abstain_variant, evidence_mode.value, *formats))


@dataclass(frozen=True)
class H10Batch:
    indices: np.ndarray
    example_ids: np.ndarray
    base_ids: np.ndarray
    scenario_family_id: np.ndarray
    evidence_shift: np.ndarray
    question_format: np.ndarray
    abstain_variant: str
    evidence_mode: EvidenceMode
    p_st: np.ndarray
    p_non_st: np.ndarray
    st_logit: np.ndarray
    log_prob_st: np.ndarray
    y_choice_st: np.ndarray


def resolve_annotation_path(annotation_run: str | Path | None = None) -> Path:
    if annotation_run is None:
        annotation_run = DEFAULT_ANNOTATION_RUN
    p = Path(annotation_run)
    if p.is_file():
        return p
    if p.is_dir():
        candidate = p / "annotations.jsonl"
        if candidate.is_file():
            return candidate
    candidate = REPO_ROOT / "data" / "annotation" / "runs" / str(annotation_run) / "annotations.jsonl"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"annotations.jsonl not found for {annotation_run!r}")


def load_annotations_by_example_id(annotation_run: str | Path | None = None) -> dict[int, dict[str, Any]]:
    path = resolve_annotation_path(annotation_run)
    out: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ex_id = int(row["example_id"])
            if ex_id in out:
                raise ValueError(f"Duplicate example_id in annotation file: {ex_id}")
            out[ex_id] = row
    if not out:
        raise ValueError(f"No annotation rows found in {path}")
    return out


def _validate_family_structure(
    base_ids: np.ndarray, evidence_mode: EvidenceMode, question_formats: tuple[str, ...]
) -> None:
    unique, counts = np.unique(base_ids, return_counts=True)
    per_ev = len(question_formats)
    expected = per_ev if evidence_mode != EvidenceMode.ALL else per_ev * len(H11_EVIDENCE_ALL)
    bad = unique[counts != expected]
    if len(bad):
        raise ValueError(
            f"Expected {expected} row(s) per scenario_family for mode={evidence_mode.value}, "
            f"families {bad[:5].tolist()}..."
        )


def build_h10_batch(
    items: list[dict[str, Any]],
    annotations_by_example_id: dict[int, dict[str, Any]],
    *,
    evidence_mode: EvidenceMode = EvidenceMode.NO_EVIDENCE,
    abstain_variant: str = "without_abstain",
    question_formats: tuple[str, ...] = H10_QUESTION_FORMATS,
) -> H10Batch:
    if abstain_variant not in ABSTAIN_VARIANTS:
        raise ValueError(f"abstain_variant must be one of {ABSTAIN_VARIANTS}, got {abstain_variant!r}")
    if not question_formats:
        raise ValueError("question_formats must be non-empty")
    unknown_qf = [q for q in question_formats if q not in H10_QUESTION_FORMATS]
    if unknown_qf:
        raise ValueError(f"Unsupported question_format(s): {unknown_qf}")

    evidence_levels = EVIDENCE_MODE_LEVELS[evidence_mode]
    qf_set = set(question_formats)

    indices: list[int] = []
    example_ids: list[int] = []
    base_ids: list[int] = []
    evidence_shifts: list[str] = []
    q_formats: list[str] = []
    p_st: list[float] = []
    y_choice_st: list[int] = []

    for i, row in enumerate(items):
        if row["evidence_shift"] not in evidence_levels:
            continue
        if row["abstain_variant"] != abstain_variant:
            continue
        if row["question_format"] not in qf_set:
            continue

        ex_id = int(row["example_id"])
        ann = annotations_by_example_id.get(ex_id)
        if ann is None:
            raise ValueError(f"Missing annotation for example_id={ex_id}")
        option_labels = ann.get("annotation", {}).get("option_labels", {})
        if not option_labels:
            raise ValueError(f"Missing annotation.option_labels for example_id={ex_id}")

        p_st_ex = 0.0
        for opt, lbl in option_labels.items():
            if lbl != "stereotype_consistent":
                continue
            p = row.get(f"prob_constrained_{opt}")
            if p is None:
                continue
            p_st_ex += float(p)

        choice = str(row["choice"])
        choice_lbl = option_labels.get(choice)
        y_choice_st.append(1 if choice_lbl == "stereotype_consistent" else 0)

        indices.append(i)
        example_ids.append(ex_id)
        base_ids.append(int(row["base_id"]))
        evidence_shifts.append(str(row["evidence_shift"]))
        q_formats.append(str(row["question_format"]))
        p_st.append(p_st_ex)

    if not indices:
        raise ValueError(
            f"No H10 rows for evidence_mode={evidence_mode.value}, abstain_variant={abstain_variant}, "
            f"question_formats={question_formats}"
        )

    base_ids_arr = np.array(base_ids, dtype=np.int64)
    _validate_family_structure(base_ids_arr, evidence_mode, question_formats)

    if evidence_mode == EvidenceMode.ALL:
        by_base: dict[int, set[str]] = {}
        for bid, ev in zip(base_ids, evidence_shifts):
            by_base.setdefault(bid, set()).add(ev)
        incomplete = [b for b, evs in by_base.items() if evs != set(H11_EVIDENCE_ALL)]
        if incomplete:
            raise ValueError(f"families missing evidence levels: {incomplete[:5]}...")

    p_st_arr = np.array(p_st, dtype=np.float64)
    eps = 1e-6
    p_st_clip = np.clip(p_st_arr, eps, 1.0 - eps)
    p_non_st = 1.0 - p_st_clip
    st_logit = np.log(p_st_clip / p_non_st)
    log_prob_st = np.log(np.clip(p_st_arr, eps, 1.0))

    return H10Batch(
        indices=np.array(indices, dtype=np.int64),
        example_ids=np.array(example_ids, dtype=np.int64),
        base_ids=base_ids_arr,
        scenario_family_id=base_ids_arr.copy(),
        evidence_shift=np.array(evidence_shifts),
        question_format=np.array(q_formats),
        abstain_variant=abstain_variant,
        evidence_mode=evidence_mode,
        p_st=p_st_arr,
        p_non_st=p_non_st,
        st_logit=st_logit,
        log_prob_st=log_prob_st,
        y_choice_st=np.array(y_choice_st, dtype=np.int64),
    )


def target_vector(batch: H10Batch, target: str) -> np.ndarray:
    if target == "h10_st_logit":
        return batch.st_logit
    if target == "h10_log_prob_st":
        return batch.log_prob_st
    if target == "h10_choice_stereotype":
        return batch.y_choice_st.astype(np.float64)
    raise ValueError(f"unknown H10 target: {target}")
