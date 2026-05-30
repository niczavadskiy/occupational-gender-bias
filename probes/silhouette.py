"""Silhouette score vs gender-preference label.  python -m probes.silhouette"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from probes.core import (
    add_common_args,
    load_canonical_family_split,
    load_probe_bundle,
    load_split_for_run,
)
from probes.h11 import EvidenceMode, build_h11_batch
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


def binary_labels(log_odds, train_mask, method):
    if method == "sign":
        return (log_odds > 0).astype(int)
    if method == "median_train":
        return (log_odds >= float(np.median(log_odds[train_mask]))).astype(int)
    raise ValueError(method)


def silhouette_one_layer(X, labels, mask):
    Xs, ys = X[mask], labels[mask]
    if len(np.unique(ys)) < 2 or Xs.shape[0] < 3:
        return float("nan")
    return float(silhouette_score(StandardScaler().fit_transform(Xs), ys))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--label-method", choices=("sign", "median_train"), default="median_train")
    parser.add_argument("--layer", type=int, default=None)
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

    labels = binary_labels(batch.log_odds, split.train_mask, args.label_method)
    layer_range = [args.layer] if args.layer is not None else range(X_layers.shape[1])
    rows = []
    for layer in layer_range:
        X = X_layers[:, layer, :]
        row = {"layer": layer}
        for name, mask in (("train", split.train_mask), ("val", split.val_mask), ("test", split.test_mask)):
            row[f"{name}_silhouette"] = silhouette_one_layer(X, labels, mask)
        rows.append(row)

    out_dir, probe_run_id = new_probe_run_dir(run_dir, "silhouette", args.out_dir)
    csv_name = f"silhouette_{mode.value}_{args.label_method}.csv"
    with (out_dir / csv_name).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    write_results_json(out_dir, {"layers": rows, "label_method": args.label_method})
    extra = {
        **h11_probe_fields(batch, "log_odds_binary", split),
        "label_method": args.label_method,
    }
    meta = build_probe_meta(
        probe_script="silhouette",
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
