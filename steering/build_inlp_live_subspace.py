"""
Classical Phase A via live HS capture (no frozen hidden_states.npz).

Fits iterative INLP on baseline (unconditioned) last-token HS for
gender_prob / gender_choice / slot_prob / slot_choice. Writes Stage-A
compatible npz+json under steering/subspaces/.

Vast / occupational-gender-bias has no probes.v1_rep bundle — use this
instead of steering.build_inlp_subspace.

Examples:

    python -m steering.build_inlp_live_subspace \\
        --target gender_prob --layers 14,15,16 --k-max 32 --tag v1

    python -m steering.build_inlp_live_subspace \\
        --target gender_choice --layers 15,16,18 --k-max 32 --tag v1
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
    parse_csv,
    parse_int_list,
    random_orthonormal,
    subspace_centers,
)
from steering.intervene import Scorer, load_model
from steering.run_hs_recovery_auc import capture_condition, family_split, pick_items, stack_hs
from steering.run_inlp_stagea import load_json

REPO_ROOT = Path(__file__).resolve().parents[1]
STEERING_DIR = Path(__file__).resolve().parent
DEFAULT_SAMPLE = STEERING_DIR / "samples" / "h1_stagea_sample_v1.json"

REGRESSION_TARGETS = frozenset({"gender_prob", "slot_prob", "narrative_prob"})


def y_for_target(rows: list[dict], target: str) -> np.ndarray:
    if target == "gender_prob":
        return np.array([r.get("p_man_norm", float("nan")) for r in rows], dtype=np.float64)
    if target == "slot_prob":
        return np.array([r.get("p_A_norm", float("nan")) for r in rows], dtype=np.float64)
    if target == "gender_choice":
        return np.array([r["y_gender"] for r in rows], dtype=np.float64)
    if target == "slot_choice":
        return np.array([r["y_slot"] for r in rows], dtype=np.float64)
    raise SystemExit(f"unknown target {target}")


def steering_ranks_for(k_found: int, base: list[int]) -> list[int]:
    ranks = set(int(r) for r in base if r <= k_found)
    if k_found > 0:
        power = 1
        while power <= k_found:
            ranks.add(power)
            power *= 2
        ranks.add(k_found)
    return sorted(r for r in ranks if 1 <= r <= k_found)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    ap.add_argument("--n-items", type=int, default=None)
    ap.add_argument("--target", default="gender_prob")
    ap.add_argument("--layers", default="14,15,16")
    ap.add_argument("--k-max", type=int, default=32)
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--clf-C", type=float, default=0.03)
    ap.add_argument("--chance-corr", type=float, default=0.15)
    ap.add_argument("--chance-auc", type=float, default=0.55)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ranks", default="1,2,4,8,16", help="base steering ranks written into meta")
    ap.add_argument("--out-dir", type=Path, default=STEERING_DIR / "subspaces")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--log-every", type=int, default=40)
    args = ap.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    target = str(args.target).strip()
    regression = target in REGRESSION_TARGETS
    task = "regression" if regression else "classification"
    solver = "ridge" if regression else "logistic"
    layers = parse_int_list(args.layers)
    if not layers:
        raise SystemExit("--layers empty")
    base_ranks = parse_int_list(args.ranks) or [1, 2, 4, 8, 16]

    sample = load_json(args.sample)
    items = pick_items(sample["items"], args.n_items)

    print(f"\n[1] load {args.model}...")
    t0 = time.time()
    model, tokenizer = load_model(args.model, dtype=args.dtype, device=args.device)
    for p in model.parameters():
        p.requires_grad_(False)
    scorer = Scorer(model, tokenizer)
    print(f"  ready in {time.time() - t0:.1f}s")

    print(f"\n[2] capture baseline HS on layers {layers}...")
    rows, hs_lists = capture_condition(
        model, scorer, items, None, layers, log_every=args.log_every
    )
    H_by_L = stack_hs(hs_lists)
    families = np.array([r["scenario_family_id"] for r in rows], dtype=np.int64)
    train_fams, test_fams = family_split(list(families), seed=args.seed, train_frac=args.train_frac)
    print(f"  rows={len(rows)}  train_fams={len(train_fams)}  test_fams={len(test_fams)}")

    y = y_for_target(rows, target)
    n_bad = int(np.sum(~np.isfinite(y)))
    if n_bad:
        raise SystemExit(f"target={target}: {n_bad} non-finite labels")
    if regression:
        print(f"  y mean={y.mean():.4f} std={y.std():.4f}")
    else:
        print(f"  y balance pos={int(y.sum())} / neg={int(len(y) - y.sum())}")

    print(f"\n[3] fit INLP target={target} task={task} solver={solver} k_max={args.k_max}")
    arrays: dict[str, np.ndarray] = {}
    layers_detail: list[dict] = []
    rng = np.random.default_rng(args.seed)

    for L in layers:
        print(f"  L{L}...")
        fit = inlp_on_layer(
            H_by_L[L],
            y,
            families,
            train_fams,
            test_fams,
            k_max=args.k_max,
            solver=solver,
            clf_C=args.clf_C,
            max_iter=2000,
            chance_auc=args.chance_auc,
            seed=args.seed + L,
            task=task,
            ridge_alpha=args.ridge_alpha,
            chance_corr=args.chance_corr,
        )
        W, c = fit["W"], fit["centers"]
        arrays[f"L{L}__W"] = W
        arrays[f"L{L}__centers"] = c
        arrays[f"L{L}__auc_curve"] = np.asarray(fit["auc_curve"], dtype=np.float32)
        k = int(W.shape[1])
        Wr = random_orthonormal(W.shape[0], max(k, 1), rng).astype(np.float32)
        train_mask = np.array([fid in train_fams for fid in families])
        cr = subspace_centers(
            np.asarray(H_by_L[L][train_mask], dtype=np.float64),
            _bin_y_for_centers(y[train_mask], task),
            Wr.astype(np.float64),
        ).astype(np.float32)
        arrays[f"L{L}__W_random_s0"] = Wr
        arrays[f"L{L}__centers_random_s0"] = cr
        ranks = steering_ranks_for(k, base_ranks)
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
                "note": fit.get("note"),
            }
        )
        sc0 = fit["auc_curve"][0] if fit["auc_curve"] else float("nan")
        print(f"    k_found={k}  score(0)={sc0:.3f}  k_chance={fit['k_chance']}  ranks={ranks}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"inlp_{target}_{args.tag}"
    npz_path = args.out_dir / f"{stem}.npz"
    meta = {
        "schema": "steering.inlp_subspace/v1",
        "hypothesis": "inlp_live",
        "map_version": args.tag,
        "datetime": datetime.now().isoformat(),
        "model_id": args.model,
        "target": target,
        "task": task,
        "layers": layers,
        "k_max": args.k_max,
        "probe": {
            "ridge_alpha": args.ridge_alpha if regression else None,
            "C": None if regression else args.clf_C,
        },
        "chance_criterion": {
            "corr_threshold": args.chance_corr,
            "auc_threshold": args.chance_auc,
        },
        "sample": str(args.sample),
        "n_rows": len(rows),
        "train_frac": args.train_frac,
        "seed": args.seed,
        "solver": solver,
        "layers_detail": layers_detail,
        "arrays_sha256": arrays_signature(arrays),
        "centers_note": (
            "c_j midpoints with y_bin=(p≥0.5) for regression; hard labels for classification"
        ),
        "read": "Live-capture classical Phase A; use with run_inlp_stagea.",
    }
    np.savez_compressed(npz_path, **arrays)
    (args.out_dir / f"{stem}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\n=== done ===")
    print(f"  {npz_path}")
    print(f"  {args.out_dir / (stem + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
