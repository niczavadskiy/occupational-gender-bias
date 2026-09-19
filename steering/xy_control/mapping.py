"""Fixed counterfactual mapping: X = man, Y = woman.

Replacement is token-level on word boundaries. ``woman`` is rewritten before
``man`` because ``woman`` contains the substring ``man``. The mapping is
identity on layout: A/B slot and sentence structure stay put; only the gender
designators change.
"""

from __future__ import annotations

import re
from typing import Mapping

MAPPING: dict[str, str] = {"X": "man", "Y": "woman"}
GENDER_TO_XY: dict[str, str] = {"man": "X", "woman": "Y"}
XY_TO_GENDER: dict[str, str] = dict(MAPPING)

# Longer token first so ``woman`` is never split into ``wo`` + ``man``.
_GENDER_TERM = re.compile(r"\b(woman|man)\b")
_LEFTOVER_GENDER = re.compile(r"\b(man|woman)\b")
_XY_TERM = re.compile(r"\b([XY])\b")

ALLOWED_GENDER_LABELS = frozenset({"man", "woman"})
ALLOWED_XY_LABELS = frozenset({"X", "Y"})


def gender_to_xy_text(text: str) -> str:
    """Replace gender designators, leaving every other character unchanged."""

    def repl(match: re.Match[str]) -> str:
        return GENDER_TO_XY[match.group(1)]

    return _GENDER_TERM.sub(repl, text)


def xy_to_gender_text(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return XY_TO_GENDER[match.group(1)]

    return _XY_TERM.sub(repl, text)


def leftover_gender_terms(text: str) -> list[str]:
    return [m.group(1) for m in _LEFTOVER_GENDER.finditer(text)]


def count_gender_terms(text: str) -> dict[str, int]:
    out = {"man": 0, "woman": 0}
    for term in leftover_gender_terms(text):
        out[term] += 1
    return out


def count_xy_terms(text: str) -> dict[str, int]:
    out = {"X": 0, "Y": 0}
    for m in _XY_TERM.finditer(text):
        out[m.group(1)] += 1
    return out


def gender_to_xy_labels(labels: Mapping[str, str]) -> dict[str, str]:
    """Map answer-slot semantics man/woman → X/Y. Slot letters stay the same."""
    out: dict[str, str] = {}
    for slot, semantic in labels.items():
        if semantic not in GENDER_TO_XY:
            raise ValueError(f"label {slot}={semantic!r} is not man/woman")
        out[str(slot)] = GENDER_TO_XY[semantic]
    return out


def validate_pair(gender_prompt: str, xy_prompt: str, labels: Mapping[str, str]) -> None:
    """Structural checks for one gendered / XY pair.

    Pre-existing tokens such as ``X-ray`` must stay put; only man/woman
    designators are rewritten, so total X-count in the XY prompt can exceed
    the number of ``man`` tokens.
    """
    expected = gender_to_xy_text(gender_prompt)
    if xy_prompt != expected:
        raise ValueError("xy_prompt is not the gender→XY rewrite of gender_prompt")
    leftover = leftover_gender_terms(xy_prompt)
    if leftover:
        raise ValueError(f"XY prompt still contains gender terms: {leftover!r}")
    if not list(_GENDER_TERM.finditer(gender_prompt)):
        raise ValueError("gender prompt has no man/woman designators")
    if gender_prompt.split("\n")[-1] != xy_prompt.split("\n")[-1]:
        raise ValueError("last line (Answer:) must be identical")
    if gender_prompt.count("\n") != xy_prompt.count("\n"):
        raise ValueError("newline count must be identical")
    missing = ALLOWED_GENDER_LABELS - set(labels.values())
    if missing:
        raise ValueError(f"labels missing {sorted(missing)}: {dict(labels)}")
    xy_labels = gender_to_xy_labels(labels)
    if set(xy_labels.values()) != ALLOWED_XY_LABELS:
        raise ValueError(f"xy labels {xy_labels}")
    gi = xi = 0
    for match in _GENDER_TERM.finditer(gender_prompt):
        span = gender_prompt[gi : match.start()]
        if xy_prompt[xi : xi + len(span)] != span:
            raise ValueError("non-gender span changed under XY rewrite")
        xi += len(span)
        xy_tok = GENDER_TO_XY[match.group(1)]
        if xy_prompt[xi : xi + len(xy_tok)] != xy_tok:
            raise ValueError("designator rewrite mismatch")
        xi += len(xy_tok)
        gi = match.end()
    if gender_prompt[gi:] != xy_prompt[xi:]:
        raise ValueError("trailing span mismatch")

