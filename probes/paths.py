"""Repo-relative paths; dataset metadata only from GitHub tree."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Canonical factorial v2 run (per_item + meta in git; HS on HF).
DEFAULT_RUN_NAME = "run_2026-05-27_19-44-46_Qwen3.5-2B-Base_factorial_v2"
DEFAULT_HF_DATASET = "bias-subspaces-group/qwen-bias-experiments"
CANONICAL_CSV = REPO_ROOT / "data" / "factorial_v2_with_gender.csv"


def default_run_dir() -> Path:
    return REPO_ROOT / "results" / DEFAULT_RUN_NAME


def resolve_run_dir(run: str | Path | None = None) -> Path:
    if run is None:
        return default_run_dir()
    p = Path(run)
    if p.is_absolute():
        return p
    # Allow short name under results/ or full path relative to repo.
    if (REPO_ROOT / "results" / p).exists():
        return REPO_ROOT / "results" / p
    if (REPO_ROOT / p).exists():
        return REPO_ROOT / p
    return p.resolve()
