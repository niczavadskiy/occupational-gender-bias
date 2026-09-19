"""Save / load paired last-token HS captures (gender + XY)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_pair_capture(
    path: Path,
    *,
    H_gender: dict[int, np.ndarray],
    H_xy: dict[int, np.ndarray],
    rows: list[dict],
    meta: dict,
) -> tuple[Path, Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for L, H in H_gender.items():
        arrays[f"H_gender_L{int(L)}"] = np.asarray(H, dtype=np.float32)
    for L, H in H_xy.items():
        arrays[f"H_xy_L{int(L)}"] = np.asarray(H, dtype=np.float32)
    np.savez_compressed(path, **arrays)
    json_path = path.with_suffix(".json")
    payload = {
        **meta,
        "layers": sorted(int(L) for L in H_gender),
        "n_rows": len(rows),
        "d_model": int(next(iter(H_gender.values())).shape[1]) if H_gender else None,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path, json_path


def load_pair_capture(path: Path) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], list[dict], dict]:
    path = Path(path)
    json_path = path.with_suffix(".json")
    if not path.is_file():
        raise SystemExit(f"нет capture {path}")
    if not json_path.is_file():
        raise SystemExit(f"нет мета {json_path}")
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    H_g: dict[int, np.ndarray] = {}
    H_xy: dict[int, np.ndarray] = {}
    with np.load(path) as z:
        for key in z.files:
            if key.startswith("H_gender_L"):
                H_g[int(key.split("L", 1)[1])] = np.asarray(z[key], dtype=np.float32)
            elif key.startswith("H_xy_L"):
                H_xy[int(key.split("L", 1)[1])] = np.asarray(z[key], dtype=np.float32)
    rows = meta["rows"]
    n = len(rows)
    for name, bank in (("gender", H_g), ("xy", H_xy)):
        for L, H in bank.items():
            if H.shape[0] != n:
                raise SystemExit(f"capture {name} L{L}: {H.shape[0]} rows vs json {n}")
    if set(H_g) != set(H_xy):
        raise SystemExit(f"gender layers {sorted(H_g)} != XY layers {sorted(H_xy)}")
    return H_g, H_xy, rows, meta
