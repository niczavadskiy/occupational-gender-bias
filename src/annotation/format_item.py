"""Build annotator-facing text from dataset rows."""

from __future__ import annotations

import math
from typing import Any


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    return True


def option_letters(has_c: bool) -> list[str]:
    return ["A", "B", "C"] if has_c else ["A", "B"]


def row_from_csv_series(row) -> dict[str, Any]:
    """Normalize a pandas Series from factorial CSV."""
    has_c = (
        row.get("abstain_variant") == "with_abstain"
        and _is_present(row.get("option_C"))
    )
    labels = {"A": row["option_A"], "B": row["option_B"]}
    if has_c:
        labels["C"] = row["option_C"]
    return {
        "example_id": int(row["example_id"]),
        "base_id": int(row["base_id"]),
        "scenario_text": row["scenario_text"],
        "question": row["question"],
        "labels": labels,
        "has_abstain": has_c,
        "predicate": row["predicate"],
        "evidence_shift": row["evidence_shift"],
        "question_format": row["question_format"],
        "abstain_variant": row["abstain_variant"],
    }


def row_from_jsonl_item(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize a prepared JSONL item from prepare_factorial.py."""
    labels = dict(item["labels"])
    has_c = bool(item.get("has_abstain")) and "C" in labels
    return {
        "example_id": int(item["example_id"]),
        "base_id": int(item["base_id"]),
        "scenario_text": item["scenario_text"],
        "question": item["question"],
        "labels": labels,
        "has_abstain": has_c,
        "predicate": item["predicate"],
        "evidence_shift": item["evidence_shift"],
        "question_format": item["question_format"],
        "abstain_variant": item["abstain_variant"],
    }


def format_item_block(row: dict[str, Any]) -> str:
    """Single item: id + Context / Question / Options."""
    eid = int(row["example_id"])
    lines = [
        f"--- id: {eid} ---",
        "Context:",
        row["scenario_text"],
        "Question:",
        row["question"],
        "Options:",
    ]
    for letter in ("A", "B", "C"):
        if letter in row["labels"]:
            lines.append(f"{letter}. {row['labels'][letter]}")
    return "\n".join(lines)


def format_batch_user_message(rows: list[dict[str, Any]]) -> str:
    """Batch user prompt."""
    blocks = [format_item_block(r) for r in rows]
    return f"Annotate all {len(rows)} items below.\n\n" + "\n\n".join(blocks)
