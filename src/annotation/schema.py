"""JSON Schema for OpenRouter structured batch output."""

from __future__ import annotations

from typing import Any

LABEL_ENUM = ["stereotype_consistent", "anti_stereotype", "neutral"]
ANSWERABILITY_ENUM = ["answerable", "unanswerable"]

_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "answerability": {"type": "string", "enum": ANSWERABILITY_ENUM},
        "option_labels": {
            "type": "object",
            "properties": {
                "A": {"type": "string", "enum": LABEL_ENUM},
                "B": {"type": "string", "enum": LABEL_ENUM},
                "C": {"type": "string", "enum": LABEL_ENUM},
            },
            "required": ["A", "B"],
            "additionalProperties": False,
        },
    },
    "required": ["id", "answerability", "option_labels"],
    "additionalProperties": False,
}

BATCH_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": _ITEM_SCHEMA,
        },
    },
    "required": ["items"],
    "additionalProperties": False,
}


def batch_response_format(*, strict: bool = False) -> dict[str, Any]:
    """OpenRouter `response_format` for a batch annotation call."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "batch_annotations",
            "strict": strict,
            "schema": BATCH_RESPONSE_SCHEMA,
        },
    }
