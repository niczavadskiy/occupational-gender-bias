"""Stage B for polarity-pooled INLP (MMLU gate on Stage A shortlist).

  python -m steering.run_polarity_pool_inlp_stageb --set-id promale --device cuda
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
from steering.run_inlp_stageb import main as stageb_main

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
    ap.add_argument("--from-stage-a", type=Path, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit-mmlu", type=int, default=None)
    ap.add_argument("--cap-loss-max", type=float, default=None)
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "results" / "steering")
    ap.add_argument("--log-every", type=int, default=100)
    args, rest = ap.parse_known_args(argv)

    sets_doc = load_polarity_sets(args.sets)
    entry = get_set(sets_doc, args.set_id)
    cfg = load_yaml(args.config)
    model = args.model or cfg["source"]["model_id"]
    tag_a = entry.get("tag_stage_a")
    tag = args.tag or entry.get("tag_stage_b") or f"{tag_a}_b"
    if args.smoke and not str(tag).endswith("_smoke"):
        tag = f"{tag}_smoke"

    pool_root = args.out_root / "inlp_polarity_pool"
    stage_a_tag = f"{tag_a}_smoke" if args.smoke else tag_a
    stage_a_dir = args.from_stage_a or (pool_root / "inlp_stage_a" / stage_a_tag)
    if not stage_a_dir.is_dir() and not args.smoke:
        alt = pool_root / "inlp_stage_a" / tag_a
        if alt.is_dir():
            stage_a_dir = alt

    sub_tag = entry.get("subspace_tag") or f"polarity_pool_{args.set_id}_v1"
    target = cfg.get("target", "gender_prob")
    subspaces = args.subspaces or (STEERING_DIR / "subspaces" / f"inlp_{target}_{sub_tag}.npz")
    cap = args.cap_loss_max
    if cap is None:
        cap = float((cfg.get("capability") or {}).get("cap_loss_max", 0.03))

    forwarded = [
        "--model",
        model,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--subspaces",
        str(subspaces),
        "--from-stage-a",
        str(stage_a_dir),
        "--tag",
        tag,
        "--out-root",
        str(pool_root),
        "--cap-loss-max",
        str(cap),
        "--skip-smoke",
        "--log-every",
        str(args.log_every),
    ]
    if args.limit_mmlu is not None:
        forwarded.extend(["--limit-mmlu", str(args.limit_mmlu)])
    elif args.smoke:
        forwarded.extend(["--limit-mmlu", "40"])
    forwarded.extend(rest)
    return int(stageb_main(forwarded) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
