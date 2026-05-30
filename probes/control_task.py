"""
Hewitt–Liang-style control tasks for H11 probing.

  python -m probes.control_task --evidence-mode all
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

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

CONTROL_MODES = (
    "main",
    "within_scenario_family",
    "global",
    "permuted_alignment",
)


def apply_control(
    y: np.ndarray,
    family_ids: np.ndarray,
    mode: str,
    *,
    seed: int,
) -> np.ndarray:
    y_ctrl = y.copy()
    rng = np.random.default_rng(seed)

    if mode == "main":
        return y_ctrl
    if mode == "within_scenario_family":
        for g in np.unique(family_ids):
            mask = family_ids == g
            y_ctrl[mask] = rng.permutation(y_ctrl[mask])
        return y_ctrl
    if mode == "global":
        return y_ctrl[rng.permutation(len(y_ctrl))]
    if mode == "permuted_alignment":
        return y[rng.permutation(len(y))]
    raise ValueError(mode)


def control_scan(X_layers, y, family_ids, split, *, target, ridge_alpha, clf_C, max_iter, seed):
    rows = []
    for control in CONTROL_MODES:
        ctrl_seed = seed + CONTROL_MODES.index(control) * 997
        y_use = apply_control(y, family_ids, control, seed=ctrl_seed)
        for layer in range(X_layers.shape[1]):
            X = X_layers[:, layer, :]
            val_m = eval_split(
                X, y_use, split.train_mask, split.val_mask,
                target=target, ridge_alpha=ridge_alpha, clf_C=clf_C, max_iter=max_iter,
            )
            test_m = eval_split(
                X, y_use, split.train_mask, split.test_mask,
                target=target, ridge_alpha=ridge_alpha, clf_C=clf_C, max_iter=max_iter,
            )
            row = {"control": control, "layer": layer}
            for k, v in val_m.items():
                row[f"val_{k}"] = v
            for k, v in test_m.items():
                row[f"test_{k}"] = v
            rows.append(row)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=10_000)
    args = parser.parse_args(argv)

    mode = EvidenceMode(args.evidence_mode)
    timer = ProbeTimer()

    try:
        run_dir, parent_meta, batch, X_layers = load_probe_bundle(args.run, mode)
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1

    items = load_per_item(run_dir)
    batch_all = build_h11_batch(items, EvidenceMode.ALL)
    load_canonical_family_split(
        run_dir, batch_all,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        force=args.force_split,
    )
    split = load_split_for_run(
        run_dir, batch,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        force=False,
    )
    y = target_vector(batch, args.target)

    rows = control_scan(
        X_layers, y, batch.scenario_family_id, split,
        target=args.target,
        ridge_alpha=args.ridge_alpha,
        clf_C=args.C,
        max_iter=args.max_iter,
        seed=args.seed,
    )

    out_dir, probe_run_id = new_probe_run_dir(run_dir, "control_task", args.out_dir)
    csv_name = f"control_{mode.value}_{args.target}.csv"
    csv_path = out_dir / csv_name
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metric = "val_r2" if args.target not in CLASSIFICATION_TARGETS else "val_balanced_accuracy"
    extra = {
        **h11_probe_fields(batch, args.target, split),
        "control_modes": list(CONTROL_MODES),
    }
    main_rows = [r for r in rows if r["control"] == "main"]
    if main_rows:
        best_main = max(main_rows, key=lambda r: r[metric])
        extra["best_main_layer"] = best_main["layer"]
        extra["best_main_metric"] = best_main[metric]
        for ctrl in ("within_scenario_family", "global", "permuted_alignment"):
            same = next(
                (r for r in rows if r["control"] == ctrl and r["layer"] == best_main["layer"]),
                None,
            )
            if same:
                extra[f"control_{ctrl}_at_best_layer"] = same[metric]

    write_results_json(out_dir, {"rows": rows})
    meta = build_probe_meta(
        probe_script="control_task",
        probe_run_id=probe_run_id,
        parent_run_dir=run_dir,
        parent_meta=parent_meta,
        probe_runtime_s=timer.elapsed(),
        artifacts={csv_name: csv_name, "results.json": "results.json"},
        extra=extra,
    )
    print(f"Wrote {write_meta(out_dir, meta)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
