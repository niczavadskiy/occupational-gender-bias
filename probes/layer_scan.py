"""
H11 layer-scan: per-layer linear probe, train → val & test.

  python -m probes.layer_scan
  python -m probes.layer_scan --evidence-modes no_evidence all
"""

from __future__ import annotations

import argparse
import csv
import sys

from probes.core import (
    CLASSIFICATION_TARGETS,
    add_common_args,
    eval_split,
    load_canonical_family_split,
    load_probe_bundle,
    load_split_for_run,
)
from probes.h11 import EvidenceMode, build_h11_batch, target_vector
from probes.load_run import load_per_item
from probes.paths import resolve_run_dir
from probes.run_io import (
    ProbeTimer,
    build_probe_meta,
    h11_probe_fields,
    new_probe_run_dir,
    write_meta,
    write_results_json,
)


def layer_scan_one(X_layers, y, split, *, target, ridge_alpha, clf_C, max_iter):
    results = []
    for layer in range(X_layers.shape[1]):
        X = X_layers[:, layer, :]
        row = {"layer": layer}
        for name, mask in (
            ("train", split.train_mask),
            ("val", split.val_mask),
            ("test", split.test_mask),
        ):
            m = eval_split(
                X, y, split.train_mask, mask,
                target=target,
                ridge_alpha=ridge_alpha,
                clf_C=clf_C,
                max_iter=max_iter,
            )
            for k, v in m.items():
                row[f"{name}_{k}"] = v
        row["n_train"] = int(split.train_mask.sum())
        row["n_val"] = int(split.val_mask.sum())
        row["n_test"] = int(split.test_mask.sum())
        results.append(row)
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--evidence-modes", nargs="*", default=None)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=10_000)
    args = parser.parse_args(argv)

    modes = (
        [EvidenceMode(m) for m in args.evidence_modes]
        if args.evidence_modes
        else list(EvidenceMode)
    )

    timer = ProbeTimer()
    run_dir = resolve_run_dir(args.run)
    items = load_per_item(run_dir)
    batch_all = build_h11_batch(items, EvidenceMode.ALL)

    try:
        _, parent_meta, _, _ = load_probe_bundle(args.run, EvidenceMode.ALL)
    except FileNotFoundError as e:
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

    out_dir, probe_run_id = new_probe_run_dir(run_dir, "layer_scan", args.out_dir)
    artifacts: dict[str, str] = {}
    results_payload: dict = {"modes": {}}
    summary_by_mode: dict = {}
    metric = "val_r2" if args.target not in CLASSIFICATION_TARGETS else "val_balanced_accuracy"

    for mode in modes:
        _, _, batch, X_layers = load_probe_bundle(args.run, mode)
        split = load_split_for_run(
            run_dir, batch,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            force=False,
        )
        y = target_vector(batch, args.target)

        print(f"\n[{mode.value}] n={len(y)} families={len(set(batch.scenario_family_id))}")

        layers = layer_scan_one(
            X_layers, y, split,
            target=args.target,
            ridge_alpha=args.ridge_alpha,
            clf_C=args.C,
            max_iter=args.max_iter,
        )
        results_payload["modes"][mode.value] = {"layers": layers}

        csv_name = f"layer_scan_{mode.value}_{args.target}.csv"
        csv_path = out_dir / csv_name
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(layers[0].keys()))
            writer.writeheader()
            writer.writerows(layers)
        artifacts[csv_name] = csv_name

        best = max(layers, key=lambda r: r.get(metric, float("-inf")))
        summary_by_mode[mode.value] = {
            "target": args.target,
            "best_layer": best["layer"],
            metric: best.get(metric),
        }
        print(f"  best layer {best['layer']}: {metric}={best.get(metric, float('nan')):.4f}")

    results_name = "results.json"
    write_results_json(out_dir, results_payload, results_name)
    artifacts[results_name] = results_name

    extra = {
        **h11_probe_fields(batch_all, args.target, family_split),
        "summary_by_mode": summary_by_mode,
        "evidence_modes_run": [m.value for m in modes],
    }
    meta = build_probe_meta(
        probe_script="layer_scan",
        probe_run_id=probe_run_id,
        parent_run_dir=run_dir,
        parent_meta=parent_meta,
        probe_runtime_s=timer.elapsed(),
        artifacts=artifacts,
        extra=extra,
    )
    meta_path = write_meta(out_dir, meta)
    print(f"\nWrote {meta_path}")
    print(f"Artifacts in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
