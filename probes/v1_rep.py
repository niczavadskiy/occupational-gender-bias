"""
V1 representation probes: gender, narrative-first, and slot-A axes.

Slice (default): task=main, question_format=choice, without_abstain,
position_variant in {p0, p1}, context_order in {man_first, woman_first}.
Split group: (soc, profession, onet_action) — 951 groups × 4 rows = 3804.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from probes.row_probs import (
    chosen_option_value,
    gender_axis_probs,
    prob_constrained_for_key,
    safe_log_odds,
)

V1_REP_LAYOUT = "v1_main"
V1_REP_TASK = "main"
V1_REP_QUESTION_FORMAT = "choice"
V1_REP_ABSTAIN = "without_abstain"
V1_REP_POSITION_VARIANTS = ("p0", "p1")
V1_REP_CONTEXT_ORDERS = ("man_first", "woman_first")
V1_REP_ROWS_PER_GROUP = len(V1_REP_POSITION_VARIANTS) * len(V1_REP_CONTEXT_ORDERS)

# Regression predicts constrained probabilities (not log-odds).
V1_REGRESSION_TARGETS = (
    "gender_prob",      # p_man
    "narrative_prob",   # p_first (first-mentioned gender in narrative)
    "slot_prob",        # p_A
)
V1_CLASSIFICATION_TARGETS = (
    "gender_choice",
    "narrative_choice",
    "slot_choice",
)
V1_ALL_TARGETS = V1_REGRESSION_TARGETS + V1_CLASSIFICATION_TARGETS
V1_PROBE_ROOT = "v1_rep_prob"


class V1RepAxis(str, Enum):
    GENDER = "gender"
    NARRATIVE = "narrative"
    SLOT_A = "slot_a"


AXIS_FOR_TARGET: dict[str, V1RepAxis] = {
    "gender_prob": V1RepAxis.GENDER,
    "gender_choice": V1RepAxis.GENDER,
    "narrative_prob": V1RepAxis.NARRATIVE,
    "narrative_choice": V1RepAxis.NARRATIVE,
    "slot_prob": V1RepAxis.SLOT_A,
    "slot_choice": V1RepAxis.SLOT_A,
}


def v1_split_json_path(run_dir: Path) -> Path:
    from probes.splits import shared_split_dir

    return shared_split_dir(run_dir) / "group_split_v1_scenario.json"


def v1_scenario_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["soc"]),
        str(row["profession"]),
        str(row["onet_action"]),
    )


def is_v1_rep_row(
    row: dict[str, Any],
    *,
    abstain_variant: str = V1_REP_ABSTAIN,
    question_format: str = V1_REP_QUESTION_FORMAT,
    position_variants: tuple[str, ...] = V1_REP_POSITION_VARIANTS,
    context_orders: tuple[str, ...] = V1_REP_CONTEXT_ORDERS,
) -> bool:
    if row.get("task") != V1_REP_TASK:
        return False
    if row.get("question_format") != question_format:
        return False
    if row.get("abstain_variant") != abstain_variant:
        return False
    if str(row.get("position_variant", "")) not in position_variants:
        return False
    if str(row.get("context_order", "")) not in context_orders:
        return False
    return True


def build_v1_group_registry(items: list[dict[str, Any]], **filter_kw: Any) -> dict[tuple[str, str, str], int]:
    keys = sorted(
        v1_scenario_key(row)
        for row in items
        if is_v1_rep_row(row, **filter_kw)
    )
    return {key: idx for idx, key in enumerate(keys)}


def collect_v1_scenario_group_ids(items: list[dict[str, Any]], **filter_kw: Any) -> np.ndarray:
    registry = build_v1_group_registry(items, **filter_kw)
    return np.array(list(registry.values()), dtype=np.int64)


def v1_pipeline_run_slug(target: str) -> str:
    if target not in V1_ALL_TARGETS:
        raise ValueError(f"unknown v1_rep target {target!r}")
    return f"{V1_REP_ABSTAIN}_p0_p1_mf_wf_{target}"


def narrative_first_prob(row: dict[str, Any], p_man: float, p_woman: float) -> float:
    if str(row["context_order"]) == "man_first":
        return p_man
    return p_woman


def narrative_first_choice(row: dict[str, Any]) -> int:
    """1 = model prefers gender mentioned first in narrative, 0 = second."""
    val = chosen_option_value(row)
    first = "man" if str(row["context_order"]) == "man_first" else "woman"
    return 1 if val == first else 0


def prefers_man_from_row(row: dict[str, Any]) -> int:
    val = chosen_option_value(row)
    if val == "man":
        return 1
    if val == "woman":
        return 0
    raise ValueError(f"v1_rep row {row.get('id')!r}: unexpected choice label {val!r}")


def slot_a_choice(row: dict[str, Any]) -> int:
    return 1 if str(row["choice"]) == "A" else 0


@dataclass(frozen=True)
class V1RepBatch:
    indices: np.ndarray
    scenario_family_id: np.ndarray
    soc: np.ndarray
    profession: np.ndarray
    onet_action: np.ndarray
    context_order: np.ndarray
    position_variant: np.ndarray
    row_ids: np.ndarray
    prob_man: np.ndarray
    prob_woman: np.ndarray
    gender_log_odds: np.ndarray
    gender_choice: np.ndarray
    narrative_prob_first: np.ndarray
    narrative_log_odds: np.ndarray
    narrative_choice: np.ndarray
    slot_prob_a: np.ndarray
    slot_prob_b: np.ndarray
    slot_log_odds: np.ndarray
    slot_choice: np.ndarray
    layout: str = V1_REP_LAYOUT
    abstain_variant: str = V1_REP_ABSTAIN

    def group_key_label(self) -> str:
        return "v1_scenario (soc × profession × onet_action)"


def _validate_group_structure(group_ids: np.ndarray) -> None:
    unique, counts = np.unique(group_ids, return_counts=True)
    bad = unique[counts != V1_REP_ROWS_PER_GROUP]
    if len(bad):
        raise ValueError(
            f"Expected {V1_REP_ROWS_PER_GROUP} rows per v1 scenario group; "
            f"bad groups (first 5): {bad[:5].tolist()}"
        )


def build_v1_rep_batch(
    items: list[dict[str, Any]],
    *,
    abstain_variant: str = V1_REP_ABSTAIN,
    question_format: str = V1_REP_QUESTION_FORMAT,
    position_variants: tuple[str, ...] = V1_REP_POSITION_VARIANTS,
    context_orders: tuple[str, ...] = V1_REP_CONTEXT_ORDERS,
    group_registry: dict[tuple[str, str, str], int] | None = None,
) -> V1RepBatch:
    filter_kw = {
        "abstain_variant": abstain_variant,
        "question_format": question_format,
        "position_variants": position_variants,
        "context_orders": context_orders,
    }
    registry = group_registry or build_v1_group_registry(items, **filter_kw)

    indices: list[int] = []
    group_ids: list[int] = []
    soc: list[str] = []
    profession: list[str] = []
    onet: list[str] = []
    context_orders_out: list[str] = []
    position_variants_out: list[str] = []
    row_ids: list[str] = []
    prob_man: list[float] = []
    prob_woman: list[float] = []
    gender_choice: list[int] = []
    narrative_prob_first: list[float] = []
    narrative_choice: list[int] = []
    slot_prob_a: list[float] = []
    slot_prob_b: list[float] = []
    slot_choice: list[int] = []

    for i, row in enumerate(items):
        if not is_v1_rep_row(row, **filter_kw):
            continue
        key = v1_scenario_key(row)
        if key not in registry:
            raise ValueError(f"Row {row.get('id')!r}: scenario key {key!r} missing from registry")

        p_man, p_woman = gender_axis_probs(row)
        p_first = narrative_first_prob(row, p_man, p_woman)
        p_a = prob_constrained_for_key(row, "A")
        p_b = prob_constrained_for_key(row, "B")

        indices.append(i)
        group_ids.append(registry[key])
        soc.append(key[0])
        profession.append(key[1])
        onet.append(key[2])
        context_orders_out.append(str(row["context_order"]))
        position_variants_out.append(str(row["position_variant"]))
        row_ids.append(str(row["id"]))
        prob_man.append(p_man)
        prob_woman.append(p_woman)
        gender_choice.append(prefers_man_from_row(row))
        narrative_prob_first.append(p_first)
        narrative_choice.append(narrative_first_choice(row))
        slot_prob_a.append(p_a)
        slot_prob_b.append(p_b)
        slot_choice.append(slot_a_choice(row))

    if not indices:
        raise ValueError("No v1_rep rows matched filters")

    group_ids_arr = np.array(group_ids, dtype=np.int64)
    _validate_group_structure(group_ids_arr)

    prob_man_arr = np.array(prob_man, dtype=np.float64)
    prob_woman_arr = np.array(prob_woman, dtype=np.float64)
    slot_a_arr = np.array(slot_prob_a, dtype=np.float64)
    slot_b_arr = np.array(slot_prob_b, dtype=np.float64)
    narrative_first_arr = np.array(narrative_prob_first, dtype=np.float64)

    gender_log_odds = np.array(
        [safe_log_odds(a, b) for a, b in zip(prob_man_arr, prob_woman_arr, strict=True)],
        dtype=np.float64,
    )
    narrative_log_odds = np.array(
        [safe_log_odds(p, 1.0 - p) for p in narrative_first_arr],
        dtype=np.float64,
    )
    slot_log_odds = np.array(
        [safe_log_odds(a, b) for a, b in zip(slot_a_arr, slot_b_arr, strict=True)],
        dtype=np.float64,
    )

    return V1RepBatch(
        indices=np.array(indices, dtype=np.int64),
        scenario_family_id=group_ids_arr,
        soc=np.array(soc),
        profession=np.array(profession),
        onet_action=np.array(onet),
        context_order=np.array(context_orders_out),
        position_variant=np.array(position_variants_out),
        row_ids=np.array(row_ids),
        prob_man=prob_man_arr,
        prob_woman=prob_woman_arr,
        gender_log_odds=gender_log_odds,
        gender_choice=np.array(gender_choice, dtype=np.int64),
        narrative_prob_first=narrative_first_arr,
        narrative_log_odds=narrative_log_odds,
        narrative_choice=np.array(narrative_choice, dtype=np.int64),
        slot_prob_a=slot_a_arr,
        slot_prob_b=slot_b_arr,
        slot_log_odds=slot_log_odds,
        slot_choice=np.array(slot_choice, dtype=np.int64),
        abstain_variant=abstain_variant,
    )


def target_vector(batch: V1RepBatch, target: str) -> np.ndarray:
    if target == "gender_prob":
        return batch.prob_man
    if target == "gender_choice":
        return batch.gender_choice.astype(np.float64)
    if target == "narrative_prob":
        return batch.narrative_prob_first
    if target == "narrative_choice":
        return batch.narrative_choice.astype(np.float64)
    if target == "slot_prob":
        return batch.slot_prob_a
    if target == "slot_choice":
        return batch.slot_choice.astype(np.float64)
    raise ValueError(f"unknown v1_rep target: {target}")


def soft_target_for_silhouette(batch: V1RepBatch, target: str) -> np.ndarray:
    """Continuous axis signal for silhouette binarization."""
    if target in V1_REGRESSION_TARGETS:
        return target_vector(batch, target)
    axis = AXIS_FOR_TARGET[target]
    if axis == V1RepAxis.GENDER:
        return batch.gender_log_odds
    if axis == V1RepAxis.NARRATIVE:
        return batch.narrative_log_odds
    return batch.slot_log_odds


def hard_label_class_balance(batch: V1RepBatch, target: str) -> dict[str, int]:
    y = target_vector(batch, target).astype(int)
    pos = int(y.sum())
    return {"positive": pos, "negative": int(len(y) - pos)}


def v1_items_table(items: list[dict[str, Any]], **kwargs: Any) -> pd.DataFrame:
    batch = build_v1_rep_batch(items, **kwargs)
    rows = [items[i] for i in batch.indices]
    df = pd.DataFrame(rows)
    df["scenario_group_id"] = batch.scenario_family_id
    df["prob_man"] = batch.prob_man
    df["prob_woman"] = batch.prob_woman
    df["gender_log_odds"] = batch.gender_log_odds
    df["gender_choice"] = batch.gender_choice
    df["narrative_prob_first"] = batch.narrative_prob_first
    df["narrative_log_odds"] = batch.narrative_log_odds
    df["narrative_choice"] = batch.narrative_choice
    df["slot_prob_a"] = batch.slot_prob_a
    df["slot_prob_b"] = batch.slot_prob_b
    df["slot_log_odds"] = batch.slot_log_odds
    df["slot_choice"] = batch.slot_choice
    return df


def load_v1_rep_bundle(
    run: str | None,
) -> tuple[Path, dict, V1RepBatch, np.ndarray]:
    from probes.load_run import load_run

    run_dir, meta, items, hs = load_run(run)
    batch = build_v1_rep_batch(items)
    X_layers = np.asarray(hs[batch.indices], dtype=np.float32)
    return run_dir, meta, batch, X_layers
