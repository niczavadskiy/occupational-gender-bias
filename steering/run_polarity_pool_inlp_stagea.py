"""Stage A for polarity-pooled INLP (val families, peak±prepeak grid).

Ranks are clamped to subspace k_found so k_found=1 still runs (rank-1 center).

  python -m steering.run_polarity_pool_inlp_stagea --set-id promale --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

from steering.polarity_pool_inlp import (
    DEFAULT_CONFIG,
    SETS_JSON,
    STEERING_DIR,
    clamp_ranks_to_k_found,
    get_set,
    load_json,
    load_polarity_sets,
    load_yaml,
    max_k_found,
    resolve_alphas,
    resolve_peak_layers,
    resolve_ranks,
)
from steering.run_inlp_stagea import main as stagea_main

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
    ap.add_argument("--sample", type=Path, default=None, help="default: samples/…_val_v1.json")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit-items", type=int, default=None)
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "results" / "steering")
    ap.add_argument("--no-random-control", action="store_true")
    ap.add_argument("--log-every", type=int, default=100)
    args, rest = ap.parse_known_args(argv)

    sets_doc = load_polarity_sets(args.sets)
    entry = get_set(sets_doc, args.set_id)
    cfg = load_yaml(args.config)
    model = args.model or cfg["source"]["model_id"]
    tag = args.tag or entry.get("tag_stage_a") or f"inlp_2b_peak_prepeak_pool_{args.set_id}"
    if args.smoke and not tag.endswith("_smoke"):
        tag = f"{tag}_smoke"

    sample = args.sample or (
        STEERING_DIR / "samples" / f"inlp_polarity_pool_{args.set_id}_val_v1.json"
    )
    sub_tag = entry.get("subspace_tag") or f"polarity_pool_{args.set_id}_v1"
    target = cfg.get("target", "gender_prob")
    subspaces = args.subspaces or (STEERING_DIR / "subspaces" / f"inlp_{target}_{sub_tag}.npz")
    if not subspaces.is_file():
        raise SystemExit(f"missing subspaces {subspaces}")

    layers = resolve_peak_layers(cfg)
    ranks_pref = resolve_ranks(cfg)
    alphas = resolve_alphas(cfg)
    if args.smoke:
        layers = [int(cfg["layers"]["anchor"])]
        ranks_pref = [1]
        alphas = [1.0]

    sub_meta = load_json(subspaces.with_suffix(".json"))
    k_cap = max_k_found(sub_meta, layers)
    ranks = clamp_ranks_to_k_found(ranks_pref, k_cap)
    print(
        f"[polarity_pool Stage A] set={args.set_id} layers={layers} "
        f"k_found_max={k_cap} ranks_pref={ranks_pref} → ranks={ranks}",
        flush=True,
    )
    if not ranks:
        raise SystemExit(f"no runnable ranks for k_found_max={k_cap}")

    out_root = args.out_root / "inlp_polarity_pool"
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
        "--layers",
        ",".join(str(x) for x in layers),
        "--ranks",
        ",".join(str(x) for x in ranks),
        "--alphas",
        ",".join(str(x) for x in alphas),
        "--tag",
        tag,
        "--out-root",
        str(out_root),
        "--primary-axis",
        "gender",
        "--log-every",
        str(args.log_every),
    ]
    if args.no_random_control:
        forwarded.append("--no-random-control")
    if args.limit_items is not None:
        forwarded.extend(["--limit-items", str(args.limit_items)])
    elif args.smoke:
        forwarded.extend(["--limit-items", "4"])
    forwarded.extend(rest)
    return int(stagea_main(forwarded) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
