"""
Full H11 probe pipeline: layer_scan → control → silhouette → direction → compare.

  python -m probes.h11_run --abstain-variant without_abstain
  python -m probes.h11_run --abstain-variant with_abstain
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from probes.compare_directions import (
    _cosine_summary,
    cosine_matrix,
    fit_all_layer_directions,
    pairwise_rows,
    projection_correlation_matrix,
)
from probes.control_task import CONTROL_MODES, control_scan
from probes.core import (
    CLASSIFICATION_TARGETS,
    add_common_args,
    classification_metrics,
    direction_w_from_probe,
    fit_probe,
    load_canonical_family_split,
    load_probe_bundle,
    load_split_for_run,
    predict_probe,
    regression_metrics,
)
from probes.h11 import ABSTAIN_VARIANTS, EvidenceMode, build_h11_batch, target_vector
from probes.layer_scan import layer_scan_one
from probes.load_run import load_per_item
from probes.paths import resolve_run_dir
from probes.run_io import (
    ProbeTimer,
    init_h11_pipeline_meta,
    record_pipeline_stage,
    write_pipeline_meta,
    write_results_json,
)
from probes.silhouette import binary_labels, silhouette_one_layer

STAGES = (
    "layer_scan",
    "control_task",
    "silhouette",
    "extract_direction",
    "compare_directions",
)


def _metric_key(target: str) -> str:
    return "val_r2" if target not in CLASSIFICATION_TARGETS else "val_balanced_accuracy"


def _best_layer(layers: list[dict], target: str) -> dict:
    metric = _metric_key(target)
    return max(layers, key=lambda r: r.get(metric, float("-inf")))


def _stage_dir(base: Path, stage: str) -> Path:
    path = base / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def _rel_artifacts(stage: str, names: dict[str, str]) -> dict[str, str]:
    return {f"{stage}/{fname}": fname for fname in names.values()}


def run_layer_scan(
    *,
    base_dir: Path,
    run_dir: Path,
    modes: list[EvidenceMode],
    abstain_variant: str,
    args: argparse.Namespace,
) -> tuple[dict, dict[str, str], dict]:
    t0 = time.time()
    out_dir = _stage_dir(base_dir, "layer_scan")
    artifacts: dict[str, str] = {}
    results_payload: dict = {"modes": {}}
    summary_by_mode: dict = {}
    metric = _metric_key(args.target)

    for mode in modes:
        _, _, batch, X_layers = load_probe_bundle(
            args.run, mode, abstain_variant=abstain_variant
        )
        split = load_split_for_run(
            run_dir,
            batch,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            force=False,
        )
        y = target_vector(batch, args.target)
        print(f"\n[layer_scan / {mode.value}] n={len(y)}")

        layers = layer_scan_one(
            X_layers,
            y,
            split,
            target=args.target,
            ridge_alpha=args.ridge_alpha,
            clf_C=args.C,
            max_iter=args.max_iter,
        )
        results_payload["modes"][mode.value] = {"layers": layers}

        csv_name = f"layer_scan_{mode.value}_{args.target}.csv"
        with (out_dir / csv_name).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(layers[0].keys()))
            writer.writeheader()
            writer.writerows(layers)
        artifacts[csv_name] = csv_name

        best = _best_layer(layers, args.target)
        summary_by_mode[mode.value] = {
            "target": args.target,
            "best_layer": int(best["layer"]),
            metric: best.get(metric),
        }
        print(f"  best layer {best['layer']}: {metric}={best.get(metric, float('nan')):.4f}")

    write_results_json(out_dir, results_payload)
    artifacts["results.json"] = "results.json"
    summary = {"summary_by_mode": summary_by_mode}
    return summary, _rel_artifacts("layer_scan", artifacts), {"elapsed": time.time() - t0}


def run_control_task(
    *,
    base_dir: Path,
    run_dir: Path,
    mode: EvidenceMode,
    abstain_variant: str,
    args: argparse.Namespace,
) -> tuple[dict, dict[str, str], dict]:
    t0 = time.time()
    out_dir = _stage_dir(base_dir, "control_task")
    _, _, batch, X_layers = load_probe_bundle(
        args.run, mode, abstain_variant=abstain_variant
    )
    split = load_split_for_run(
        run_dir,
        batch,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        force=False,
    )
    y = target_vector(batch, args.target)
    rows = control_scan(
        X_layers,
        y,
        batch.scenario_family_id,
        split,
        target=args.target,
        ridge_alpha=args.ridge_alpha,
        clf_C=args.C,
        max_iter=args.max_iter,
        seed=args.seed,
    )

    csv_name = f"control_{mode.value}_{args.target}.csv"
    with (out_dir / csv_name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_results_json(out_dir, {"rows": rows})

    metric = _metric_key(args.target)
    summary: dict = {"control_modes": list(CONTROL_MODES)}
    main_rows = [r for r in rows if r["control"] == "main"]
    if main_rows:
        best_main = max(main_rows, key=lambda r: r[metric])
        summary["best_main_layer"] = int(best_main["layer"])
        summary["best_main_metric"] = best_main[metric]
        for ctrl in ("within_scenario_family", "global", "permuted_alignment"):
            same = next(
                (r for r in rows if r["control"] == ctrl and r["layer"] == best_main["layer"]),
                None,
            )
            if same:
                summary[f"control_{ctrl}_at_best_layer"] = same[metric]

    artifacts = {csv_name: csv_name, "results.json": "results.json"}
    return summary, _rel_artifacts("control_task", artifacts), {"elapsed": time.time() - t0}


def run_silhouette(
    *,
    base_dir: Path,
    run_dir: Path,
    mode: EvidenceMode,
    abstain_variant: str,
    args: argparse.Namespace,
) -> tuple[dict, dict[str, str], dict]:
    t0 = time.time()
    out_dir = _stage_dir(base_dir, "silhouette")
    _, _, batch, X_layers = load_probe_bundle(
        args.run, mode, abstain_variant=abstain_variant
    )
    split = load_split_for_run(
        run_dir,
        batch,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        force=False,
    )
    labels = binary_labels(batch.log_odds, split.train_mask, args.label_method)
    rows = []
    for layer in range(X_layers.shape[1]):
        X = X_layers[:, layer, :]
        row: dict = {"layer": layer}
        for name, mask in (
            ("train", split.train_mask),
            ("val", split.val_mask),
            ("test", split.test_mask),
        ):
            row[f"{name}_silhouette"] = silhouette_one_layer(X, labels, mask)
        rows.append(row)

    csv_name = f"silhouette_{mode.value}_{args.label_method}.csv"
    with (out_dir / csv_name).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    write_results_json(out_dir, {"layers": rows, "label_method": args.label_method})

    best = max(rows, key=lambda r: r.get("val_silhouette", float("-inf")))
    summary = {
        "label_method": args.label_method,
        "best_layer": int(best["layer"]),
        "val_silhouette": best.get("val_silhouette"),
    }
    artifacts = {csv_name: csv_name, "results.json": "results.json"}
    return summary, _rel_artifacts("silhouette", artifacts), {"elapsed": time.time() - t0}


def run_extract_direction(
    *,
    base_dir: Path,
    run_dir: Path,
    mode: EvidenceMode,
    abstain_variant: str,
    args: argparse.Namespace,
    best_layer: int,
) -> tuple[dict, dict[str, str], dict]:
    t0 = time.time()
    out_dir = _stage_dir(base_dir, "extract_direction")
    _, _, batch, X_layers = load_probe_bundle(
        args.run, mode, abstain_variant=abstain_variant
    )
    split = load_split_for_run(
        run_dir,
        batch,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        force=False,
    )
    y = target_vector(batch, args.target)
    layer = best_layer
    X = X_layers[:, layer, :]
    model = fit_probe(
        X,
        y,
        split.train_mask,
        target=args.target,
        ridge_alpha=args.ridge_alpha,
        clf_C=args.C,
        max_iter=args.max_iter,
    )

    metrics: dict = {"layer": layer}
    for name, mask in (
        ("train", split.train_mask),
        ("val", split.val_mask),
        ("test", split.test_mask),
    ):
        pred, _ = predict_probe(model, X, mask, target=args.target)
        m = (
            classification_metrics(y[mask].astype(int), pred.astype(int), None)
            if args.target in CLASSIFICATION_TARGETS
            else regression_metrics(y[mask], pred)
        )
        for k, v in m.items():
            metrics[f"{name}_{k}"] = v

    w_raw = direction_w_from_probe(model, target=args.target)
    if args.target in CLASSIFICATION_TARGETS:
        coef = model.named_steps["clf"].coef_.ravel()
    else:
        coef = model.named_steps["ridge"].coef_.ravel()
    projections = X @ w_raw

    npz_name = f"direction_{mode.value}_{args.target}_L{layer}.npz"
    np.savez_compressed(
        out_dir / npz_name,
        layer=layer,
        coef_scaled=coef.astype(np.float32),
        w_raw=w_raw.astype(np.float32),
        projections=projections.astype(np.float32),
        log_odds=batch.log_odds.astype(np.float32),
        scenario_family_id=batch.scenario_family_id.astype(np.int32),
        example_ids=batch.example_ids.astype(np.int32),
        evidence_shift=batch.evidence_shift,
        split_train=split.train_mask,
        split_val=split.val_mask,
        split_test=split.test_mask,
    )
    write_results_json(out_dir, {"metrics": metrics, "layer": layer})

    proj_r = float(np.corrcoef(projections, batch.log_odds)[0, 1])
    summary = {"metrics": metrics, "projection_log_odds_r": proj_r}
    artifacts = {npz_name: npz_name, "results.json": "results.json"}
    return summary, _rel_artifacts("extract_direction", artifacts), {"elapsed": time.time() - t0}


def run_compare_directions(
    *,
    base_dir: Path,
    run_dir: Path,
    mode: EvidenceMode,
    abstain_variant: str,
    args: argparse.Namespace,
) -> tuple[dict, dict[str, str], dict]:
    t0 = time.time()
    out_dir = _stage_dir(base_dir, "compare_directions")
    _, _, batch, X_layers = load_probe_bundle(
        args.run, mode, abstain_variant=abstain_variant
    )
    split = load_split_for_run(
        run_dir,
        batch,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        force=False,
    )
    y = target_vector(batch, args.target)
    w = fit_all_layer_directions(
        X_layers,
        y,
        split.train_mask,
        target=args.target,
        ridge_alpha=args.ridge_alpha,
        clf_C=args.C,
        max_iter=args.max_iter,
    )
    cos_w = cosine_matrix(w)
    cos_proj = projection_correlation_matrix(X_layers, w)
    pairs = pairwise_rows(cos_w)
    n_layers = int(w.shape[0])
    stem = f"compare_directions_{mode.value}_{args.target}"
    csv_name = f"{stem}_cos_w_pairs.csv"
    matrix_csv = f"{stem}_cos_w_matrix.csv"

    with (out_dir / csv_name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer_a", "layer_b", "cosine"])
        writer.writeheader()
        writer.writerows(pairs)
    with (out_dir / matrix_csv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer"] + list(range(n_layers)))
        for i in range(n_layers):
            writer.writerow([i] + [float(cos_w[i, j]) for j in range(n_layers)])

    npz_name = f"{stem}_weights.npz"
    np.savez_compressed(
        out_dir / npz_name,
        w_raw=w.astype(np.float32),
        cosine_w=cos_w.astype(np.float32),
        projection_correlation=cos_proj.astype(np.float32),
        layers=np.arange(n_layers, dtype=np.int32),
    )
    summary_dict = _cosine_summary(cos_w, n_layers)
    write_results_json(
        out_dir,
        {
            "n_layers": n_layers,
            "evidence_mode": mode.value,
            "target": args.target,
            "pairwise_cosine_w": pairs,
            "cosine_w_matrix": cos_w.tolist(),
            "projection_correlation_matrix": cos_proj.tolist(),
            "summary": summary_dict,
        },
    )
    summary = {"summary": summary_dict}
    artifacts = {
        csv_name: csv_name,
        matrix_csv: matrix_csv,
        npz_name: npz_name,
        "results.json": "results.json",
    }
    return summary, _rel_artifacts("compare_directions", artifacts), {"elapsed": time.time() - t0}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--abstain-variant",
        choices=ABSTAIN_VARIANTS,
        required=True,
        help="without_abstain or with_abstain (A/B vs A/B/C prompts)",
    )
    parser.add_argument(
        "--evidence-modes",
        nargs="*",
        default=None,
        help="layer_scan modes (default: same as --evidence-mode)",
    )
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=10_000)
    parser.add_argument(
        "--label-method",
        choices=("sign", "median_train"),
        default="median_train",
    )
    parser.add_argument(
        "--steps",
        nargs="*",
        choices=STAGES,
        default=None,
        help="Run subset of stages (default: all)",
    )
    args = parser.parse_args(argv)

    abstain_variant = args.abstain_variant
    primary_mode = EvidenceMode(args.evidence_mode)
    scan_modes = (
        [EvidenceMode(m) for m in args.evidence_modes]
        if args.evidence_modes
        else [primary_mode]
    )
    steps = list(STAGES if args.steps is None else args.steps)

    timer = ProbeTimer()
    run_dir = resolve_run_dir(args.run)
    items = load_per_item(run_dir)

    try:
        batch_all = build_h11_batch(
            items, EvidenceMode.ALL, abstain_variant=abstain_variant
        )
        _, parent_meta, _, _ = load_probe_bundle(
            args.run, EvidenceMode.ALL, abstain_variant=abstain_variant
        )
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1

    family_split = load_canonical_family_split(
        run_dir,
        batch_all,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        force=args.force_split,
    )
    print(f"Split: {family_split.summary()}")

    pipeline_run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = run_dir / "probes" / "h11" / abstain_variant
    base_dir.mkdir(parents=True, exist_ok=True)

    _, _, primary_batch, _ = load_probe_bundle(
        args.run, primary_mode, abstain_variant=abstain_variant
    )
    primary_split = load_split_for_run(
        run_dir,
        primary_batch,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        force=False,
    )

    meta = init_h11_pipeline_meta(
        pipeline_run_id=pipeline_run_id,
        abstain_variant=abstain_variant,
        parent_run_dir=run_dir,
        parent_meta=parent_meta,
        batch=primary_batch,
        target=args.target,
        split=primary_split,
        evidence_modes=[m.value for m in scan_modes],
    )
    write_pipeline_meta(base_dir, meta)
    print(f"Pipeline → {base_dir / 'meta.json'}")

    best_layer: int | None = None

    if "layer_scan" in steps:
        print("\n=== layer_scan ===")
        summary, artifacts, timing = run_layer_scan(
            base_dir=base_dir,
            run_dir=run_dir,
            modes=scan_modes,
            abstain_variant=abstain_variant,
            args=args,
        )
        primary_summary = summary["summary_by_mode"].get(primary_mode.value, {})
        if "best_layer" in primary_summary:
            best_layer = int(primary_summary["best_layer"])
        record_pipeline_stage(
            meta,
            "layer_scan",
            runtime_s=timing["elapsed"],
            summary=summary,
            stage_artifacts=artifacts,
            timer=timer,
        )
        write_pipeline_meta(base_dir, meta)

    if "control_task" in steps:
        print("\n=== control_task ===")
        summary, artifacts, timing = run_control_task(
            base_dir=base_dir,
            run_dir=run_dir,
            mode=primary_mode,
            abstain_variant=abstain_variant,
            args=args,
        )
        record_pipeline_stage(
            meta,
            "control_task",
            runtime_s=timing["elapsed"],
            summary=summary,
            stage_artifacts=artifacts,
            timer=timer,
        )
        write_pipeline_meta(base_dir, meta)

    if "silhouette" in steps:
        print("\n=== silhouette ===")
        summary, artifacts, timing = run_silhouette(
            base_dir=base_dir,
            run_dir=run_dir,
            mode=primary_mode,
            abstain_variant=abstain_variant,
            args=args,
        )
        record_pipeline_stage(
            meta,
            "silhouette",
            runtime_s=timing["elapsed"],
            summary=summary,
            stage_artifacts=artifacts,
            timer=timer,
        )
        write_pipeline_meta(base_dir, meta)

    if "extract_direction" in steps:
        if best_layer is None:
            print("extract_direction requires layer_scan best_layer; run layer_scan first", file=sys.stderr)
            return 1
        print(f"\n=== extract_direction (L{best_layer}) ===")
        summary, artifacts, timing = run_extract_direction(
            base_dir=base_dir,
            run_dir=run_dir,
            mode=primary_mode,
            abstain_variant=abstain_variant,
            args=args,
            best_layer=best_layer,
        )
        record_pipeline_stage(
            meta,
            "extract_direction",
            runtime_s=timing["elapsed"],
            summary=summary,
            stage_artifacts=artifacts,
            timer=timer,
        )
        write_pipeline_meta(base_dir, meta)

    if "compare_directions" in steps:
        print("\n=== compare_directions ===")
        summary, artifacts, timing = run_compare_directions(
            base_dir=base_dir,
            run_dir=run_dir,
            mode=primary_mode,
            abstain_variant=abstain_variant,
            args=args,
        )
        record_pipeline_stage(
            meta,
            "compare_directions",
            runtime_s=timing["elapsed"],
            summary=summary,
            stage_artifacts=artifacts,
            timer=timer,
        )
        write_pipeline_meta(base_dir, meta)

    print(f"\nDone. meta → {base_dir / 'meta.json'} ({meta['pipeline_runtime_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
