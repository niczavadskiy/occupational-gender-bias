"""Stage C for polarity-pooled INLP (test families + MMLU domain_test).

  python -m steering.run_polarity_pool_inlp_stagec --set-id promale --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

from steering.polarity_pool_inlp import (
    DEFAULT_CONFIG,
    SETS_JSON,
    STEERING_DIR,
    get_set,
    load_polarity_sets,
    load_yaml,
)
from steering.run_inlp_stagec import main as stagec_main

REPO_ROOT = STEERING_DIR.parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set-id", required=True)
    ap.add_argument("--sets", type=Path, default=SETS_JSON)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--subspaces", type=Path, default=None)
    ap.add_argument("--sample", type=Path, default=None)
    ap.add_argument("--from-stage-b", type=Path, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit-items", type=int, default=None)
    ap.add_argument("--limit-mmlu", type=int, default=None)
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "results" / "steering")
    ap.add_argument("--log-every", type=int, default=100)
    args, rest = ap.parse_known_args(argv)

    sets_doc = load_polarity_sets(args.sets)
    entry = get_set(sets_doc, args.set_id)
    cfg = load_yaml(args.config)
    model = args.model or cfg["source"]["model_id"]
    tag_b = entry.get("tag_stage_b")
    tag = args.tag or entry.get("tag_stage_c") or f"{tag_b}_c"
    if args.smoke and not str(tag).endswith("_smoke"):
        tag = f"{tag}_smoke"

    sample = args.sample or (
        STEERING_DIR / "samples" / f"inlp_polarity_pool_{args.set_id}_test_v1.json"
    )
    pool_root = args.out_root / "inlp_polarity_pool"
    stage_b_tag = f"{tag_b}_smoke" if args.smoke else tag_b
    stage_b_dir = args.from_stage_b or (pool_root / "inlp_stage_b" / stage_b_tag)

    sub_tag = entry.get("subspace_tag") or f"polarity_pool_{args.set_id}_v1"
    target = cfg.get("target", "gender_prob")
    subspaces = args.subspaces or (STEERING_DIR / "subspaces" / f"inlp_{target}_{sub_tag}.npz")

    forwarded = [
        "--model",
        model,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--subspaces",
        str(subspaces),
        "--sample",
        str(sample),
        "--from-stage-b",
        str(stage_b_dir),
        "--tag",
        tag,
        "--out-root",
        str(pool_root),
        "--log-every",
        str(args.log_every),
    ]
    if args.limit_items is not None:
        forwarded.extend(["--limit-items", str(args.limit_items)])
    elif args.smoke:
        forwarded.extend(["--limit-items", "4"])
    if args.limit_mmlu is not None:
        forwarded.extend(["--limit-mmlu", str(args.limit_mmlu)])
    elif args.smoke:
        forwarded.extend(["--limit-mmlu", "40"])
    forwarded.extend(rest)
    return int(stagec_main(forwarded) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
