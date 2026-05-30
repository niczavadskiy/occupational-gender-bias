"""Manifest + split check (no hidden_states).  python -m probes.inspect_h11"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from probes.core import load_canonical_family_split, load_split_for_run
from probes.h11 import EvidenceMode, build_h11_batch, hard_label_class_balance, h11_items_table
from probes.load_run import load_per_item
from probes.paths import DEFAULT_RUN_NAME, resolve_run_dir
from probes.run_io import (
    ProbeTimer,
    build_probe_meta,
    new_probe_run_dir,
    write_meta,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN_NAME)
    parser.add_argument("--evidence-mode", choices=[m.value for m in EvidenceMode], default=EvidenceMode.ALL.value)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-split", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    timer = ProbeTimer()
    run_dir = resolve_run_dir(args.run)
    items = load_per_item(run_dir)
    mode = EvidenceMode(args.evidence_mode)
    batch_all = build_h11_batch(items, EvidenceMode.ALL)
    batch = build_h11_batch(items, evidence_mode=mode)

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

    print(f"Run: {run_dir.name}")
    print(f"Families: {len(np.unique(batch_all.scenario_family_id))}")
    print(f"Split: {split.summary()}")
    for name in ("train", "val", "test"):
        print(f"  {name}: {int(split.mask(name).sum())} rows")

    out_dir, probe_run_id = new_probe_run_dir(run_dir, "inspect_h11", args.out_dir)
    manifest_name = f"manifest_{mode.value}.csv"
    df = h11_items_table(items, evidence_mode=mode)
    df["split"] = split.split_name_per_row(batch.scenario_family_id)
    df[
        [
            "example_id", "base_id", "scenario_family_id", "evidence_shift", "split",
            "base_context", "predicate", "choice", "prefers_man", "log_odds_man",
            "prob_man", "prob_woman",
        ]
    ].to_csv(out_dir / manifest_name, index=False)

    with (run_dir / "meta.json").open(encoding="utf-8") as f:
        import json
        parent_meta = json.load(f)

    meta = build_probe_meta(
        probe_script="inspect_h11",
        probe_run_id=probe_run_id,
        parent_run_dir=run_dir,
        parent_meta=parent_meta,
        probe_runtime_s=timer.elapsed(),
        artifacts={
            manifest_name: manifest_name,
            "split_file": "../_shared/group_split_scenario_family.json",
        },
        extra={
            "evidence_mode": mode.value,
            "hard_choice": hard_label_class_balance(batch.y),
            "split": split.summary(),
        },
    )
    print(f"Wrote {write_meta(out_dir, meta)}")
    print(f"Manifest → {out_dir / manifest_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
