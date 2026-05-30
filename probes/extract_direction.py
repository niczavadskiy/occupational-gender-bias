"""Extract probe direction w.  python -m probes.extract_direction --pick-best-layer"""

from __future__ import annotations

import argparse
import sys

import numpy as np

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
from probes.h11 import EvidenceMode, build_h11_batch, target_vector
from probes.load_run import load_per_item
from probes.paths import resolve_run_dir
from probes.run_io import (
    ProbeTimer,
    build_probe_meta,
    find_best_layer_from_layer_scan,
    h11_probe_fields,
    new_probe_run_dir,
    write_meta,
    write_results_json,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--pick-best-layer", action="store_true")
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

    batch_all = build_h11_batch(load_per_item(run_dir), EvidenceMode.ALL)
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

    layer = args.layer
    if args.pick_best_layer:
        layer = find_best_layer_from_layer_scan(run_dir, mode.value, args.target)
        if layer is None:
            print("No layer_scan run found; pass --layer", file=sys.stderr)
            return 1
        print(f"Best layer from layer_scan: {layer}")
    if layer is None:
        print("Specify --layer or --pick-best-layer", file=sys.stderr)
        return 1

    X = X_layers[:, layer, :]
    model = fit_probe(
        X, y, split.train_mask,
        target=args.target,
        ridge_alpha=args.ridge_alpha,
        clf_C=args.C,
        max_iter=args.max_iter,
    )

    metrics = {"layer": layer}
    for name, mask in (("train", split.train_mask), ("val", split.val_mask), ("test", split.test_mask)):
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

    out_dir, probe_run_id = new_probe_run_dir(run_dir, "extract_direction", args.out_dir)
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
    extra = {
        **h11_probe_fields(batch, args.target, split),
        "metrics": metrics,
        "projection_log_odds_r": float(np.corrcoef(projections, batch.log_odds)[0, 1]),
    }
    meta = build_probe_meta(
        probe_script="extract_direction",
        probe_run_id=probe_run_id,
        parent_run_dir=run_dir,
        parent_meta=parent_meta,
        probe_runtime_s=timer.elapsed(),
        artifacts={npz_name: npz_name, "results.json": "results.json"},
        extra=extra,
    )
    print(f"Wrote {write_meta(out_dir, meta)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
