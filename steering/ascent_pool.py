"""
Detect pre-peak ascent pool from a layer-wise AUC (or bacc) curve.

Pool = layers from the start of the sharp rise up to and including the peak.
First hit = min(pool). Post-peak layers are excluded (use other pipelines).

Criteria (combined):
  1. Relative height: a(L) >= peak - τ·(peak - chance)
  2. Knee: first strong positive gradient before the peak (γ · max_grad)

See steering/SEQUENTIAL_PRE_PEAK_ERASE.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AscentPoolResult:
    peak_layer: int
    peak_auc: float
    knee_layer: int | None
    thresh_auc: float
    pool: list[int]
    first_hit: int
    metric: str
    tau: float
    gamma: float
    chance: float
    curve: dict[int, float]
    grads: dict[int, float]
    note: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["curve"] = {str(k): v for k, v in self.curve.items()}
        d["grads"] = {str(k): v for k, v in self.grads.items()}
        return d


def load_auc_curve(
    path: Path,
    *,
    metric: str = "val_roc_auc",
    min_layer: int | None = None,
    max_layer: int | None = None,
) -> dict[int, float]:
    """Load {layer: score} from layer_scan results.json or a flat auc map."""
    import json

    raw = json.loads(path.read_text(encoding="utf-8"))
    curve: dict[int, float] = {}

    if isinstance(raw, dict) and "layers" in raw and isinstance(raw["layers"], list):
        for row in raw["layers"]:
            L = int(row["layer"])
            if metric not in row:
                # fallbacks
                for alt in ("val_balanced_accuracy", "val_accuracy", "roc_auc", "auc"):
                    if alt in row:
                        metric = alt
                        break
            if metric not in row:
                continue
            curve[L] = float(row[metric])
    elif isinstance(raw, dict) and "auc_curve" in raw:
        for k, v in raw["auc_curve"].items():
            curve[int(k)] = float(v)
    elif isinstance(raw, dict):
        # flat {"12": 0.77, ...} or {"L12": ...}
        for k, v in raw.items():
            if k in ("metric", "schema", "note"):
                continue
            ks = str(k).lstrip("Ll")
            if ks.isdigit():
                curve[int(ks)] = float(v)
    else:
        raise ValueError(f"unsupported AUC file shape: {path}")

    if min_layer is not None:
        curve = {L: a for L, a in curve.items() if L >= min_layer}
    if max_layer is not None:
        curve = {L: a for L, a in curve.items() if L <= max_layer}
    if not curve:
        raise ValueError(f"empty AUC curve from {path}")
    return dict(sorted(curve.items()))


def detect_ascent_pool(
    curve: dict[int, float],
    *,
    tau: float = 0.38,
    gamma: float = 0.25,
    chance: float = 0.5,
    metric: str = "val_roc_auc",
    min_pool: int = 2,
    max_pool: int = 8,
    fallback_top_m: int = 3,
    min_prev_auc_for_knee: float = 0.65,
    prefer_height_start: bool = True,
) -> AscentPoolResult:
    """
    Parameters
    ----------
    tau :
        Layer enters height-pool if a(L) >= peak - τ·(peak - chance).
        Contiguous fill from min(height_pool) to peak is the main pool.
    gamma :
        Knee = first layer (with prev AUC >= min_prev_auc_for_knee) whose
        gradient >= gamma * max positive gradient in that restricted set.
        Used as a diagnostic / optional narrowing; by default first_hit follows
        the height-pool start (earliest strong layer), not the early-layer
        embedding spike.
    min_prev_auc_for_knee :
        Ignore gradients whose left neighbor is still near chance (avoids L0→L1).
    """
    if not curve:
        raise ValueError("empty curve")
    layers = sorted(curve)
    peak_layer = max(layers, key=lambda L: curve[L])
    peak_auc = float(curve[peak_layer])
    span = max(peak_auc - chance, 1e-9)
    thresh = peak_auc - float(tau) * span

    pre = [L for L in layers if L <= peak_layer]
    grads: dict[int, float] = {}
    for i in range(1, len(pre)):
        L0, L1 = pre[i - 1], pre[i]
        grads[L1] = float(curve[L1] - curve[L0])

    # knee only on the late ascent (prev AUC already informative)
    grads_knee = {
        L: g
        for L, g in grads.items()
        if curve.get(L - 1, chance) >= float(min_prev_auc_for_knee) and g > 0
    }
    pos = list(grads_knee.values()) or [g for g in grads.values() if g > 0]
    g_ref = max(pos) if pos else 0.0
    knee: int | None = None
    if g_ref > 0 and grads_knee:
        for L in sorted(grads_knee):
            if grads_knee[L] >= float(gamma) * g_ref:
                knee = L
                break

    height_pool = [L for L in pre if curve[L] >= thresh]
    note = "height_contiguous_to_peak"

    if height_pool:
        start_h = min(height_pool)
        if prefer_height_start or knee is None:
            start = start_h
        else:
            # narrow toward peak if knee is later than height start
            start = max(start_h, knee)
            if start != start_h:
                note = "height_start_narrowed_by_knee"
        pool = [L for L in pre if start <= L <= peak_layer]
    elif knee is not None:
        start = knee
        pool = [L for L in pre if start <= L <= peak_layer]
        note = "knee_only"
    else:
        pool = []
        note = "empty"

    if len(pool) < min_pool:
        ranked = sorted(pre, key=lambda L: curve[L], reverse=True)[:fallback_top_m]
        start = min(ranked)
        pool = [L for L in pre if start <= L <= peak_layer]
        note = f"fallback_top_{fallback_top_m}_fill_to_peak"

    if len(pool) > max_pool:
        # keep earliest ascent: trim from the *right* only if somehow oversized
        # before peak — actually keep the left start, drop middle? No: keep
        # [start .. start+max_pool-1] capped at peak would drop the peak.
        # Prefer keeping peak: take the last max_pool layers of the pool
        # BUT only if start was wrongly early; if note is height_*, recompute
        # with stricter tau rather than silent trim when possible.
        pool = pool[-max_pool:]
        note = note + f"|trim_right_keep_peak_max={max_pool}"

    if not pool:
        pool = [peak_layer]
        note = "degenerate_peak_only"

    first_hit = min(pool)
    return AscentPoolResult(
        peak_layer=peak_layer,
        peak_auc=peak_auc,
        knee_layer=knee,
        thresh_auc=float(thresh),
        pool=pool,
        first_hit=first_hit,
        metric=metric,
        tau=float(tau),
        gamma=float(gamma),
        chance=float(chance),
        curve={L: float(curve[L]) for L in layers},
        grads=grads,
        note=note,
    )


def detect_ascent_pool_from_file(
    path: Path,
    *,
    metric: str = "val_roc_auc",
    **kwargs: Any,
) -> AscentPoolResult:
    curve = load_auc_curve(path, metric=metric)
    return detect_ascent_pool(curve, metric=metric, **kwargs)
