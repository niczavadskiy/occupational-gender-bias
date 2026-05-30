"""Load inference run: meta.json, per_item.jsonl, hidden_states.npz."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from probes.paths import resolve_run_dir


def load_meta(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "meta.json"
    if not path.is_file():
        raise FileNotFoundError(f"meta.json not found: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_per_item(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "per_item.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"per_item.jsonl not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_hidden_states(run_dir: Path) -> tuple[np.ndarray, list[str]]:
    """
  Returns
    hs: float array [N, n_layers, d_model] — same row order as per_item.jsonl
    item_ids: list of ex* ids from npz
    """
    path = run_dir / "hidden_states.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"hidden_states.npz not found: {path}\n"
            "Download from HF: python -m probes.download_hf"
        )
    data = np.load(path)
    hs = np.asarray(data["hs"], dtype=np.float32)
    item_ids = [str(x) for x in data["item_ids"]]
    return hs, item_ids


def load_run(run: str | Path | None = None) -> tuple[Path, dict, list[dict], np.ndarray]:
    run_dir = resolve_run_dir(run)
    meta = load_meta(run_dir)
    items = load_per_item(run_dir)
    hs, item_ids = load_hidden_states(run_dir)

    if len(items) != hs.shape[0]:
        raise ValueError(
            f"Row count mismatch: per_item={len(items)} vs hidden_states={hs.shape[0]}"
        )
    expected_ids = [r["id"] for r in items]
    if item_ids != expected_ids:
        raise ValueError(
            "item_ids in hidden_states.npz do not match per_item.jsonl order. "
            f"First mismatch at index {next(i for i, (a, b) in enumerate(zip(item_ids, expected_ids)) if a != b)}"
        )
    if meta.get("n_items") not in (None, len(items)):
        raise ValueError(f"meta n_items={meta.get('n_items')} vs actual {len(items)}")

    return run_dir, meta, items, hs
