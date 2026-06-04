"""Load per_item.jsonl into a DataFrame."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_NAME = "run_2026-05-27_19-44-46_Qwen3.5-2B-Base_factorial_v2"


def resolve_run_dir(run: str | Path | None = None) -> Path:
    if run is None:
        return REPO_ROOT / "results" / DEFAULT_RUN_NAME
    p = Path(run)
    if p.is_absolute():
        return p
    if (REPO_ROOT / "results" / p).is_dir():
        return REPO_ROOT / "results" / p
    if (REPO_ROOT / p).is_dir():
        return REPO_ROOT / p
    return REPO_ROOT / "results" / p


def load_per_item_jsonl(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "per_item.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_meta(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "meta.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_dataframe(run_dir: Path) -> pd.DataFrame:
    rows = load_per_item_jsonl(run_dir)
    df = pd.DataFrame(rows)
    if len(df) != 1350:
        raise ValueError(f"Expected 1350 rows, got {len(df)} in {run_dir}")
    return df
