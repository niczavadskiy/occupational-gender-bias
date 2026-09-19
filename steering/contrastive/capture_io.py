"""Save / load last-token HS captures for contrastive rank PCA (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_capture(
    path: Path,
    *,
    H_by_L: dict[int, np.ndarray],
    rows: list[dict],
    meta: dict,
) -> tuple[Path, Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for L, H in H_by_L.items():
        arrays[f"H_L{int(L)}"] = np.asarray(H, dtype=np.float32)
    np.savez_compressed(path, **arrays)
    json_path = path.with_suffix(".json")
    payload = {
        **meta,
        "layers": sorted(int(L) for L in H_by_L),
        "n_rows": len(rows),
        "d_model": int(next(iter(H_by_L.values())).shape[1]) if H_by_L else None,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path, json_path


def load_capture(path: Path) -> tuple[dict[int, np.ndarray], list[dict], dict]:
    path = Path(path)
    json_path = path.with_suffix(".json")
    if not path.is_file():
        raise SystemExit(f"нет capture {path}")
    if not json_path.is_file():
        raise SystemExit(f"нет мета {json_path}")
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    H_by_L: dict[int, np.ndarray] = {}
    with np.load(path) as z:
        for key in z.files:
            if key.startswith("H_L"):
                H_by_L[int(key[3:])] = np.asarray(z[key], dtype=np.float32)
    rows = meta["rows"]
    n = len(rows)
    for L, H in H_by_L.items():
        if H.shape[0] != n:
            raise SystemExit(f"capture L{L}: {H.shape[0]} rows vs json {n}")
    return H_by_L, rows, meta
