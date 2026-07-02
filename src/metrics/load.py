"""Load per_item.jsonl into a DataFrame."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_NAME = "run_2026-05-27_19-44-46_Qwen3.5-2B-Base_factorial_v2"
PER_ITEM_JSONL = "per_item.jsonl"

# Legacy factorial v2 + v3 (4/6 option-position slots × optional context_order merge).
KNOWN_ROW_COUNTS = frozenset({1350, 5400, 10800, 21600})

LEGACY_PRIMARY_POSITION = "canonical"
V3_PRIMARY_POSITION = "p0"

METRICS_POSITION_ATTR = "metrics_position_variant"
METRICS_CONTEXT_ATTR = "metrics_context_order"


def per_item_path(run_dir: Path) -> Path:
    return run_dir / PER_ITEM_JSONL


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        try:
            key = p.resolve()
        except OSError:
            key = p
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def resolve_run_dir(run: str | Path | None = None) -> Path:
    """Resolve inference run directory containing per_item.jsonl."""
    if run is None:
        candidate = REPO_ROOT / "results" / DEFAULT_RUN_NAME
        return _require_run_dir(candidate)

    p = Path(run).expanduser()
    if p.is_file():
        if p.name != PER_ITEM_JSONL and p.suffix.lower() != ".jsonl":
            raise ValueError(f"Expected {PER_ITEM_JSONL} or *.jsonl, got {p}")
        return _require_run_dir(p.parent)

    candidates = _dedupe_paths([p, REPO_ROOT / "results" / p, REPO_ROOT / p])
    tried: list[str] = []
    for candidate in candidates:
        tried.append(str(candidate))
        if per_item_path(candidate).is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"No {PER_ITEM_JSONL} found for {run!r}. Tried:\n  " + "\n  ".join(tried)
    )


def _require_run_dir(run_dir: Path) -> Path:
    path = per_item_path(run_dir)
    if not path.is_file():
        raise FileNotFoundError(path)
    return run_dir.resolve()


def resolve_position_variant(df: pd.DataFrame, override: str | None = None) -> str | None:
    """Return explicit position slice, or None to keep all positions (default)."""
    if not override or override.lower() in ("", "auto", "all"):
        return None
    return override


def resolve_context_order(df: pd.DataFrame, override: str | None = None) -> str | None:
    """Return explicit context_order slice, or None to keep all context orders (default)."""
    if not override or override.lower() in ("", "auto", "all"):
        return None
    if "context_order" not in df.columns:
        return None
    vals = sorted(df["context_order"].dropna().astype(str).unique())
    if override not in vals:
        raise ValueError(f"context_order {override!r} not in data: {vals}")
    return override


def prepare_metrics_frame(
    df: pd.DataFrame,
    *,
    position_variant: str | None = None,
    context_order: str | None = None,
) -> pd.DataFrame:
    """Filter metrics frame by optional position/context slices (default: all positions and contexts)."""
    pos = resolve_position_variant(df, position_variant)
    ctx = resolve_context_order(df, context_order)

    out = df.copy()
    if pos is not None:
        out = out.loc[out["position_variant"].astype(str) == pos]
    if ctx is not None:
        out = out.loc[out["context_order"].astype(str) == ctx]

    out.attrs[METRICS_POSITION_ATTR] = pos
    out.attrs[METRICS_CONTEXT_ATTR] = ctx
    return out


def metrics_position_variant(df: pd.DataFrame) -> str | None:
    pos = df.attrs.get(METRICS_POSITION_ATTR)
    if pos:
        return str(pos)
    return None


def load_per_item_jsonl(run_dir: Path) -> list[dict[str, Any]]:
    path = per_item_path(run_dir)
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
    n = len(df)
    if n not in KNOWN_ROW_COUNTS:
        raise ValueError(
            f"Unexpected row count {n} in {run_dir}. "
            f"Known layouts: {sorted(KNOWN_ROW_COUNTS)} "
            f"(1350 legacy; 5400 main+answerability; 10800/21600 v3 position shuffle)."
        )
    return df


def main_task_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rows for H1–H3/H5/H6/post-hoc (exclude answerability companions)."""
    if "task" in df.columns:
        return df.loc[df["task"] == "main"].copy()
    return df.copy()
