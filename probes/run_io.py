"""Probe run directories and meta.json (mirrors inference run layout)."""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

from probes.h11 import H11Batch
from probes.splits import GroupTVTSplit


def runtime_env() -> dict[str, Any]:
    """Probe host (linear probes run on CPU; no torch/GPU in meta)."""
    return {
        "probe_device": "cpu",
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "sklearn_version": sklearn.__version__,
    }


def new_probe_run_dir(
    run_dir: Path,
    probe_script: str,
    out_dir: str | Path | None = None,
) -> tuple[Path, str]:
    """Returns (output_directory, probe_run_id timestamp slug)."""
    if out_dir is not None:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path, path.name
    probe_run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = run_dir / "probes" / probe_script / probe_run_id
    path.mkdir(parents=True, exist_ok=True)
    return path, probe_run_id


def h11_probe_fields(batch: H11Batch, target: str, split: GroupTVTSplit) -> dict[str, Any]:
    n_fam = len(np.unique(batch.scenario_family_id))
    return {
        "hypothesis": "H11",
        "evidence_mode": batch.evidence_mode.value,
        "target": target,
        "n_samples": int(len(batch.scenario_family_id)),
        "n_scenario_families": n_fam,
        "rows_per_family": int(len(batch.scenario_family_id) / max(1, n_fam)),
        "split": split.summary(),
        "filter": {
            "question_format": "choice",
            "abstain_variant": batch.abstain_variant,
            "evidence_shift_levels": sorted({str(x) for x in batch.evidence_shift}),
        },
    }


def init_h11_pipeline_meta(
    *,
    pipeline_run_id: str,
    abstain_variant: str,
    parent_run_dir: Path,
    parent_meta: dict[str, Any],
    batch: H11Batch,
    target: str,
    split: GroupTVTSplit,
    evidence_modes: list[str],
) -> dict[str, Any]:
    """Single meta.json for probes/h11/<abstain_variant>/ pipeline runs."""
    fields = h11_probe_fields(batch, target, split)
    return {
        "probe_script": "h11_pipeline",
        "pipeline_run_id": pipeline_run_id,
        "datetime": datetime.now().isoformat(),
        "pipeline_runtime_s": 0.0,
        "abstain_variant": abstain_variant,
        "parent_run_id": parent_meta.get("run_id"),
        "parent_run_dir": parent_run_dir.name,
        "model_id": parent_meta.get("model_id"),
        "n_layers_plus_emb": parent_meta.get("n_layers_plus_emb"),
        "d_model": parent_meta.get("d_model"),
        **runtime_env(),
        "hypothesis": fields["hypothesis"],
        "target": target,
        "evidence_mode": fields["evidence_mode"],
        "evidence_modes": evidence_modes,
        "n_samples": fields["n_samples"],
        "n_scenario_families": fields["n_scenario_families"],
        "rows_per_family": fields["rows_per_family"],
        "split": fields["split"],
        "filter": fields["filter"],
        "stages": {},
        "artifacts": {},
    }


def record_pipeline_stage(
    meta: dict[str, Any],
    stage: str,
    *,
    runtime_s: float,
    summary: dict[str, Any],
    stage_artifacts: dict[str, str],
    timer: ProbeTimer,
) -> None:
    meta["stages"][stage] = {
        "status": "completed",
        "runtime_s": round(runtime_s, 2),
        "summary": summary,
    }
    for rel, fname in stage_artifacts.items():
        meta["artifacts"][rel] = fname
    meta["datetime"] = datetime.now().isoformat()
    meta["pipeline_runtime_s"] = round(timer.elapsed(), 2)


def write_pipeline_meta(base_dir: Path, meta: dict[str, Any]) -> Path:
    return write_meta(base_dir, meta)


def build_probe_meta(
    *,
    probe_script: str,
    probe_run_id: str,
    parent_run_dir: Path,
    parent_meta: dict[str, Any],
    probe_runtime_s: float,
    artifacts: dict[str, str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """meta.json for a probe script run (parent fields + probe fields)."""
    meta: dict[str, Any] = {
        "probe_script": probe_script,
        "probe_run_id": probe_run_id,
        "datetime": datetime.now().isoformat(),
        "probe_runtime_s": round(probe_runtime_s, 2),
        "parent_run_id": parent_meta.get("run_id"),
        "parent_run_dir": parent_run_dir.name,
        "model_id": parent_meta.get("model_id"),
        "n_layers_plus_emb": parent_meta.get("n_layers_plus_emb"),
        "d_model": parent_meta.get("d_model"),
        **runtime_env(),
        "artifacts": artifacts,
    }
    if extra:
        meta.update(extra)
    return meta


def write_meta(out_dir: Path, meta: dict[str, Any]) -> Path:
    path = out_dir / "meta.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return path


def write_results_json(out_dir: Path, payload: dict[str, Any], name: str = "results.json") -> Path:
    path = out_dir / name
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


class ProbeTimer:
    def __init__(self) -> None:
        self._t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self._t0


def find_best_layer_from_layer_scan(
    run_dir: Path,
    evidence_mode: str,
    target: str,
) -> int | None:
    """Read best layer from latest layer_scan probe run meta or results."""
    scan_root = run_dir / "probes" / "layer_scan"
    if not scan_root.is_dir():
        return None

    candidates = sorted(
        (p for p in scan_root.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    metric = "val_r2" if target not in ("choice",) else "val_balanced_accuracy"

    for run_path in candidates:
        meta_path = run_path / "meta.json"
        if not meta_path.is_file():
            continue
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)
        summary = meta.get("summary_by_mode", {})
        mode_summary = summary.get(evidence_mode, {})
        if mode_summary.get("target") == target and "best_layer" in mode_summary:
            return int(mode_summary["best_layer"])

        results_path = run_path / "results.json"
        if results_path.is_file():
            with results_path.open(encoding="utf-8") as f:
                data = json.load(f)
            layers = data.get("modes", {}).get(evidence_mode, {}).get("layers", [])
            if layers:
                best = max(layers, key=lambda r: r.get(metric, float("-inf")))
                return int(best["layer"])
    return None
