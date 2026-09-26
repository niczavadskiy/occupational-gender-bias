"""Fit classical INLP subspace on one polarity pool (train families only).

Layers default to config belt: probe peak + peak−1 + peak−2.

  python -m steering.build_polarity_pool_inlp_subspace \\
    --set-id promale --device cuda --dtype float32
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from steering.build_conditional_inlp_subspace import (
    _bin_y_for_centers,
    arrays_signature,
    inlp_on_layer,
    random_orthonormal,
    subspace_centers,
)
from steering.build_inlp_live_subspace import REGRESSION_TARGETS, steering_ranks_for, y_for_target
from steering.intervene import Scorer, load_model
from steering.polarity_pool_inlp import (
    DEFAULT_CONFIG,
    SETS_JSON,
    STEERING_DIR,
    expand_inlp_pool_candidates,
    get_set,
    load_json,
    load_polarity_sets,
    load_yaml,
    resolve_alphas,
    resolve_peak_layers,
    resolve_ranks,
    sample_paths_for,
)
from steering.run_hs_recovery_auc import capture_condition, stack_hs

REPO_ROOT = STEERING_DIR.parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set-id", required=True)
    ap.add_argument("--sets", type=Path, default=SETS_JSON)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--sample", type=Path, default=None, help="full pool sample JSON (default: samples/…)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--layers", default=None, help="override belt, e.g. 16,15,14")
    ap.add_argument("--k-max", type=int, default=None)
    ap.add_argument("--ridge-alpha", type=float, default=None)
    ap.add_argument("--chance-corr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ranks", default=None, help="base steering ranks written into meta")
    ap.add_argument("--out-dir", type=Path, default=STEERING_DIR / "subspaces")
    ap.add_argument("--tag", default=None, help="override subspace_tag from sets JSON")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit-items", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=40)
    args = ap.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    sets_doc = load_polarity_sets(args.sets)
    entry = get_set(sets_doc, args.set_id)
    cfg = load_yaml(args.config)
    model_id = args.model or cfg["source"]["model_id"]
    target = str(cfg.get("target", "gender_prob"))
    layers = (
        [int(x) for x in args.layers.split(",") if x.strip()]
        if args.layers
        else resolve_peak_layers(cfg)
    )
    if args.smoke:
        layers = [int(cfg["layers"]["anchor"])]
    k_max = int(args.k_max if args.k_max is not None else cfg["inlp"]["k_max"])
    ridge = float(
        args.ridge_alpha
        if args.ridge_alpha is not None
        else cfg.get("probe", {}).get("ridge_alpha", 1.0)
    )
    chance = float(
        args.chance_corr
        if args.chance_corr is not None
        else cfg["inlp"]["chance"]["corr_threshold"]
    )
    seed = int(args.seed if args.seed is not None else cfg["split"]["seed"])
    base_ranks = (
        [int(x) for x in args.ranks.split(",") if x.strip()]
        if args.ranks
        else resolve_ranks(cfg)
    )
    tag = args.tag or entry.get("subspace_tag") or f"polarity_pool_{args.set_id}_v1"

    sample_path = args.sample or sample_paths_for(entry)["full"]
    if not sample_path.is_file():
        raise SystemExit(
            f"missing {sample_path}; run: python -m steering.build_polarity_pool_inlp_sample "
            f"--set-id {args.set_id} --sets {args.sets} --config {args.config}"
        )
    sample = load_json(sample_path)
    split = sample["split"]
    train_fams = set(int(x) for x in split["train_family_ids"])
    val_fams = set(int(x) for x in split["val_family_ids"])
    test_fams = set(int(x) for x in split["test_family_ids"])
    heldout = val_fams | test_fams

    items = [it for it in sample["items"] if int(it["scenario_family_id"]) in (train_fams | heldout)]
    if args.limit_items is not None:
        items = items[: args.limit_items]
    elif args.smoke:
        items = items[: max(4, min(8, len(items)))]

    regression = target in REGRESSION_TARGETS
    task = "regression" if regression else "classification"
    solver = "ridge" if regression else "logistic"

    print(f"\n[1] load {model_id}...")
    t0 = time.time()
    model, tokenizer = load_model(model_id, dtype=args.dtype, device=args.device)
    for p in model.parameters():
        p.requires_grad_(False)
    scorer = Scorer(model, tokenizer)
    print(f"  ready in {time.time() - t0:.1f}s")

    print(f"\n[2] capture HS layers={layers} on {len(items)} families (train+heldout)...")
    rows, hs_lists = capture_condition(
        model, scorer, items, None, layers, log_every=args.log_every
    )
    H_by_L = stack_hs(hs_lists)
    families = np.array([r["scenario_family_id"] for r in rows], dtype=np.int64)
    train_mask_f = np.array([int(f) in train_fams for f in families])
    held_mask_f = np.array([int(f) in heldout for f in families])
    print(
        f"  rows={len(rows)} train_fams={train_mask_f.sum() // 4} "
        f"heldout_fams={held_mask_f.sum() // 4} layers={layers}"
    )

    y = y_for_target(rows, target)
    if int(np.sum(~np.isfinite(y))):
        raise SystemExit(f"non-finite labels for {target}")

    # inlp_on_layer expects train_fams / test_fams sets
    fit_train = train_fams
    fit_test = heldout if heldout else train_fams

    print(f"\n[3] fit INLP target={target} k_max={k_max} solver={solver}")
    arrays: dict[str, np.ndarray] = {}
    layers_detail: list[dict] = []
    rng = np.random.default_rng(seed)

    for L in layers:
        print(f"  L{L}...")
        fit = inlp_on_layer(
            H_by_L[L],
            y,
            families,
            fit_train,
            fit_test,
            k_max=k_max,
            solver=solver,
            clf_C=0.03,
            max_iter=2000,
            chance_auc=0.55,
            seed=seed + L,
            task=task,
            ridge_alpha=ridge,
            chance_corr=chance,
        )
        W, c = fit["W"], fit["centers"]
        arrays[f"L{L}__W"] = W
        arrays[f"L{L}__centers"] = c
        arrays[f"L{L}__auc_curve"] = np.asarray(fit["auc_curve"], dtype=np.float32)
        k = int(W.shape[1])
        Wr = random_orthonormal(W.shape[0], max(k, 1), rng).astype(np.float32)
        train_row = np.array([fid in fit_train for fid in families])
        cr = subspace_centers(
            np.asarray(H_by_L[L][train_row], dtype=np.float64),
            _bin_y_for_centers(y[train_row], task),
            Wr.astype(np.float64),
        ).astype(np.float32)
        arrays[f"L{L}__W_random_s0"] = Wr
        arrays[f"L{L}__centers_random_s0"] = cr
        ranks = steering_ranks_for(k, base_ranks)
        # ensure protocol ranks present if k allows
        for r in base_ranks:
            if r <= k and r not in ranks:
                ranks.append(r)
        ranks = sorted(set(ranks))
        layers_detail.append(
            {
                "layer": L,
                "target": target,
                "task": task,
                "k_found": k,
                "k_chance": fit["k_chance"],
                "auc_curve": [float(a) for a in fit["auc_curve"]],
                "auc_0": float(fit["auc_curve"][0]) if fit["auc_curve"] else float("nan"),
                "steering_ranks": ranks,
                "random_control_ranks": ranks,
                "solver": fit["solver"],
                "role": "peak" if L == int(cfg["layers"]["peak"]) else "prepeak",
            }
        )
        print(f"    k_found={k} ranks={ranks} role={layers_detail[-1]['role']}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"inlp_{target}_{tag}"
    npz_path = args.out_dir / f"{stem}.npz"
    cands = expand_inlp_pool_candidates(cfg, smoke=args.smoke)
    meta = {
        "schema": "steering.inlp_polarity_pool_subspace/v1",
        "hypothesis": "inlp_polarity_pool",
        "protocol": "pooled_polarity_one_subspace",
        "map_version": tag,
        "datetime": datetime.now().isoformat(),
        "model_id": model_id,
        "target": target,
        "task": task,
        "set_id": args.set_id,
        "polarity": entry["polarity"],
        "layers": layers,
        "probe_peak": int(cfg["layers"]["peak"]),
        "prepeak": list(cfg["layers"]["prepeak"]),
        "k_max": k_max,
        "probe": {"ridge_alpha": ridge},
        "chance_criterion": {"corr_threshold": chance},
        "sample": str(sample_path),
        "n_rows": len(rows),
        "split": {
            "n_train": len(train_fams),
            "n_val": len(val_fams),
            "n_test": len(test_fams),
            "seed": seed,
            "rule": "family_split3_by_soc then pool (matched XY)",
        },
        "candidate_grid": {
            "layers": resolve_peak_layers(cfg),
            "ranks": resolve_ranks(cfg),
            "alphas": resolve_alphas(cfg),
            "ids": [c["id"] for c in cands],
        },
        "seed": seed,
        "solver": solver,
        "layers_detail": layers_detail,
        "arrays_sha256": arrays_signature(arrays),
        "read": "Polarity-pooled Phase A; use with run_polarity_pool_inlp_stagea.",
    }
    np.savez_compressed(npz_path, **arrays)
    json_path = args.out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n=== done ===\n  {npz_path}\n  {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
