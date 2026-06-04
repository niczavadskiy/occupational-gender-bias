"""Validate annotator JSON output (batch)."""

from __future__ import annotations

import json
import re
from typing import Any

from annotation.format_item import option_letters

ANSWERABILITY = frozenset({"answerable", "unanswerable"})
OPTION_LABELS = frozenset({"stereotype_consistent", "anti_stereotype", "neutral"})
ALLOWED_ITEM_KEYS = frozenset({"id", "answerability", "option_labels"})


def extract_json(text: str) -> Any:
    """Parse JSON from model output; tolerate optional ```json fences."""
    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL | re.IGNORECASE)
    if fence:
        raw = fence.group(1)
    else:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw = raw[start : end + 1]
    return json.loads(raw)


def extract_json_object(text: str) -> dict[str, Any]:
    data = extract_json(text)
    if not isinstance(data, dict):
        raise ValueError("expected JSON object")
    return data


def validate_item_annotation(
    data: dict[str, Any],
    expected_letters: list[str],
    *,
    expected_id: int | None = None,
) -> dict[str, Any]:
    extra = set(data.keys()) - ALLOWED_ITEM_KEYS
    if extra:
        raise ValueError(f"unexpected keys {sorted(extra)}; allowed {sorted(ALLOWED_ITEM_KEYS)}")

    item_id = data.get("id")
    if expected_id is not None and int(item_id) != int(expected_id):
        raise ValueError(f"id {item_id!r} != expected {expected_id}")

    ans = data.get("answerability")
    if ans not in ANSWERABILITY:
        raise ValueError(f"invalid answerability: {ans!r}")

    labels = data.get("option_labels")
    if not isinstance(labels, dict):
        raise ValueError("option_labels must be an object")

    got = sorted(labels.keys())
    exp = sorted(expected_letters)
    if got != exp:
        raise ValueError(f"option_labels keys {got} != expected {exp}")

    out_labels: dict[str, str] = {}
    for letter in expected_letters:
        val = labels[letter]
        if val not in OPTION_LABELS:
            raise ValueError(f"invalid label for {letter}: {val!r}")
        out_labels[letter] = val

    return {
        "id": int(item_id),
        "answerability": ans,
        "option_labels": out_labels,
    }


def validate_batch_response(
    data: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Validate `{"items": [...]}`; return {example_id: annotation}."""
    if set(data.keys()) != {"items"}:
        raise ValueError('batch root must be {"items": [...]} only')

    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError("items must be an array")

    expected_ids = {int(r["example_id"]) for r in rows}
    if len(items) != len(expected_ids):
        raise ValueError(f"got {len(items)} items, expected {len(expected_ids)}")

    rows_by_id = {int(r["example_id"]): r for r in rows}
    seen: set[int] = set()
    out: dict[int, dict[str, Any]] = {}

    for raw_item in items:
        if not isinstance(raw_item, dict):
            raise ValueError("each item must be an object")
        eid = int(raw_item["id"])
        ann = validate_item_annotation(
            raw_item,
            option_letters(rows_by_id[eid]["has_abstain"]),
            expected_id=eid,
        )
        if eid in seen:
            raise ValueError(f"duplicate id {eid}")
        if eid not in expected_ids:
            raise ValueError(f"unexpected id {eid}")
        seen.add(eid)
        out[eid] = ann

    if seen != expected_ids:
        raise ValueError(f"missing ids: {sorted(expected_ids - seen)}")

    return out
