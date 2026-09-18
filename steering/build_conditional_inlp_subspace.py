"""
Conditional INLP under fixed base stack S0 (default: gender INLP L15 k16).

Captures last-token HS **while S0 is active**, then fits iterative linear
directions for gender_choice and/or slot_choice on requested layers.

Output npz/json is Stage-A compatible (same keys as build_inlp_subspace):
  Lℓ__W, Lℓ__centers, Lℓ__W_random_s0, Lℓ__centers_random_s0, …

Solver:
  - logistic (sklearn StandardScaler + LogisticRegression) if available
  - else iterative mean-diff (no sklearn)

Examples:

    # after second-hit shortlist
    python -m steering.build_conditional_inlp_subspace \\
        --model Qwen/Qwen3.5-2B-Base --device cuda --dtype float32 \\
        --from-shortlist results/steering/second_hit/second_hit_v1/shortlist.json \\
        --targets gender_choice,slot_choice --tag cond_under_l15_v1

    # explicit layers
    python -m steering.build_conditional_inlp_subspace \\
        --layers 16,17,18,20,23 --targets gender_choice --k-max 32 --tag smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from steering.intervene import Scorer, load_model
from steering.run_hs_recovery_auc import (
    build_inlp_spec,
    capture_condition,
    family_split,
    pick_items,
    roc_auc,
    stack_hs,
)
from steering.run_inlp_stagea import load_json

REPO_ROOT = Path(__file__).resolve().parents[1]
STEERING_DIR = Path(__file__).resolve().parent
DEFAULT_SAMPLE = STEERING_DIR / "samples" / "h1_stagea_sample_v1.json"
DEFAULT_SUBSPACES = STEERING_DIR / "subspaces" / "inlp_gender_choice_v1.npz"


def parse_int_list(s: str | None) -> list[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def arrays_signature(arrays: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for key in sorted(arrays):
        h.update(key.encode("utf-8"))
        h.update(np.ascontiguousarray(arrays[key], dtype=np.float32).tobytes())
    return h.hexdigest()


def random_orthonormal(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    Q, _ = np.linalg.qr(rng.normal(size=(d, k)))
    return np.asarray(Q[:, :k], dtype=np.float64)


def subspace_centers(H_raw: np.ndarray, y: np.ndarray, W: np.ndarray) -> np.ndarray:
    s = H_raw @ W
    y = y.astype(int)
    mu0 = s[y == 0].mean(axis=0)
    mu1 = s[y == 1].mean(axis=0)
    return 0.5 * (mu0 + mu1)


def residualize(H: np.ndarray, w: np.ndarray) -> np.ndarray:
    return H - np.outer(H @ w, w)


def fit_direction_mean_diff(H_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
    y = y_tr.astype(int)
    pos, neg = H_tr[y == 1], H_tr[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("mean_diff needs both classes")
    w = pos.mean(axis=0) - neg.mean(axis=0)
    n = float(np.linalg.norm(w))
    if n < 1e-12:
        raise ValueError("zero mean_diff")
    return w / n


def fit_direction_logistic(H_tr: np.ndarray, y_tr: np.ndarray, C: float, max_iter: int) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    solver="lbfgs",
                    class_weight=None,
                ),
            ),
        ]
    )
    pipe.fit(H_tr, y_tr)
    scaler: StandardScaler = pipe.named_steps["scaler"]
    clf: LogisticRegression = pipe.named_steps["clf"]
    coef_scaled = np.asarray(clf.coef_, dtype=np.float64).ravel()
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    w_raw = coef_scaled / np.maximum(scale, 1e-12)
    n = float(np.linalg.norm(w_raw))
    if n < 1e-12:
        raise ValueError("zero logistic direction")
    return w_raw / n


def fit_direction_ridge(H_tr: np.ndarray, y_tr: np.ndarray, alpha: float) -> np.ndarray:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    pipe.fit(H_tr, y_tr.astype(np.float64))
    scaler: StandardScaler = pipe.named_steps["scaler"]
    ridge: Ridge = pipe.named_steps["ridge"]
    coef_scaled = np.asarray(ridge.coef_, dtype=np.float64).ravel()
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    w_raw = coef_scaled / np.maximum(scale, 1e-12)
    n = float(np.linalg.norm(w_raw))
    if n < 1e-12:
        raise ValueError("zero ridge direction")
    return w_raw / n


def _val_score(scores: np.ndarray, y: np.ndarray, *, task: str) -> float:
    """ROC-AUC (classification) or Pearson r (regression)."""
    y = np.asarray(y, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if task == "regression":
        if len(y) < 2 or float(np.std(y)) < 1e-12 or float(np.std(scores)) < 1e-12:
            return float("nan")
        return float(np.corrcoef(y, scores)[0, 1])
    # classification
    return roc_auc(scores, y.astype(np.int32))


def inlp_on_layer(
    H: np.ndarray,
    y: np.ndarray,
    families: np.ndarray,
    train_fams: set[int],
    test_fams: set[int],
    *,
    k_max: int,
    solver: str,
    clf_C: float,
    max_iter: int,
    chance_auc: float,
    seed: int,
    task: str = "classification",
    ridge_alpha: float = 1.0,
    chance_corr: float = 0.15,
) -> dict:
    train = np.array([fid in train_fams for fid in families])
    val = np.array([fid in test_fams for fid in families])
    H_tr = np.asarray(H[train], dtype=np.float64).copy()
    H_va = np.asarray(H[val], dtype=np.float64).copy()
    H_tr_raw = H_tr.copy()
    y_tr = y[train]
    y_va = y[val]
    if task == "classification":
        y_tr = y_tr.astype(int)
        y_va = y_va.astype(int)

    directions: list[np.ndarray] = []
    auc_curve: list[float] = []
    k_chance: int | None = None
    method = solver
    stop_thr = float(chance_corr if task == "regression" else chance_auc)

    for j in range(k_max + 1):
        try:
            if task == "regression":
                if solver == "mean_diff":
                    # continuous mean-diff: high vs low half
                    med = float(np.median(y_tr))
                    y_bin = (y_tr >= med).astype(int)
                    w_score = fit_direction_mean_diff(H_tr, y_bin)
                else:
                    w_score = fit_direction_ridge(H_tr, y_tr, ridge_alpha)
                    method = "ridge"
            elif solver == "logistic":
                w_score = fit_direction_logistic(H_tr, y_tr.astype(int), clf_C, max_iter)
            else:
                w_score = fit_direction_mean_diff(H_tr, y_tr.astype(int))
            score = _val_score(H_va @ w_score, y_va, task=task)
        except Exception:
            score = float("nan")
            w_score = None
        auc_curve.append(float(score) if np.isfinite(score) else float("nan"))

        # regression: stop when |corr| small; classification: AUC near chance
        if np.isfinite(score):
            done = abs(score) <= stop_thr if task == "regression" else score <= stop_thr
            if done:
                k_chance = j
                break
        if j == k_max or w_score is None:
            break

        w = w_score.copy()
        if directions:
            Wprev = np.stack(directions, axis=1)
            w = w - Wprev @ (Wprev.T @ w)
            n = float(np.linalg.norm(w))
            if n < 1e-12:
                break
            w = w / n
        directions.append(w.astype(np.float64))
        H_tr = residualize(H_tr, w)
        H_va = residualize(H_va, w)

    if not directions:
        rng = np.random.default_rng(seed)
        d = H.shape[1]
        W = random_orthonormal(d, 1, rng)
        centers = subspace_centers(H_tr_raw, _bin_y_for_centers(y_tr, task), W)
        return {
            "W": W.astype(np.float32),
            "centers": centers.astype(np.float32),
            "auc_curve": auc_curve or [float("nan")],
            "k_found": 1,
            "k_chance": k_chance,
            "solver": method,
            "task": task,
            "note": "no direction extracted; placeholder rank-1 random",
        }

    W = np.stack(directions, axis=1)
    centers = subspace_centers(H_tr_raw, _bin_y_for_centers(y_tr, task), W)
    return {
        "W": W.astype(np.float32),
        "centers": centers.astype(np.float32),
        "auc_curve": auc_curve,
        "k_found": int(W.shape[1]),
        "k_chance": k_chance,
        "solver": method,
        "task": task,
    }


def _bin_y_for_centers(y: np.ndarray, task: str) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if task == "regression":
        return (y >= 0.5).astype(int)
    return y.astype(int)


def layers_from_shortlist(path: Path, *, axis_filter: set[str] | None) -> list[int]:
    doc = load_json(path)
    layers: list[int] = []
    for e in doc.get("shortlist", []):
        ax = str(e.get("axis", ""))
        if axis_filter and ax not in axis_filter:
            continue
        L = int(e["layer"])
        if L not in layers:
            layers.append(L)
    return layers


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    ap.add_argument("--n-items", type=int, default=None)
    ap.add_argument("--base-subspaces", type=Path, default=DEFAULT_SUBSPACES)
    ap.add_argument("--base-layer", type=int, default=15)
    ap.add_argument("--base-rank", type=int, default=16)
    ap.add_argument("--base-alpha", type=float, default=1.0)
    ap.add_argument("--layers", default="", help="comma layers; or use --from-shortlist")
    ap.add_argument("--from-shortlist", type=Path, default=None)
    ap.add_argument("--shortlist-axes", default="gender,slot", help="which shortlist axes to take layers from")
    ap.add_argument("--targets", default="gender_choice,slot_choice")
    ap.add_argument("--k-max", type=int, default=32)
    ap.add_argument("--solver", choices=["auto", "logistic", "mean_diff"], default="auto")
    ap.add_argument("--clf-C", type=float, default=0.03)
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--chance-auc", type=float, default=0.55)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=STEERING_DIR / "subspaces")
    ap.add_argument("--tag", default="cond_under_l15_v1")
    ap.add_argument("--log-every", type=int, default=40)
    args = ap.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    layers = parse_int_list(args.layers)
    if args.from_shortlist is not None:
        axes = set(parse_csv(args.shortlist_axes))
        sl = layers_from_shortlist(args.from_shortlist, axis_filter=axes)
        for L in sl:
            if L not in layers:
                layers.append(L)
        print(f"shortlist layers from {args.from_shortlist.name}: {sl}")
    if not layers:
        layers = [16, 17, 18, 20, 23]
        print(f"no layers given → default {layers}")

    targets = parse_csv(args.targets)
    solver = args.solver
    if solver == "auto":
        try:
            import sklearn  # noqa: F401

            solver = "logistic"
        except ImportError:
            solver = "mean_diff"
            print("sklearn missing → solver=mean_diff")

    sample = load_json(args.sample)
    items = pick_items(sample["items"], args.n_items)
    if not args.base_subspaces.is_file():
        raise SystemExit(f"MISSING {args.base_subspaces}")
    base_arrays = {k: np.asarray(v, dtype=np.float32) for k, v in np.load(args.base_subspaces).items()}
    base_spec = build_inlp_spec(
        base_arrays,
        layer=args.base_layer,
        rank=args.base_rank,
        alpha=args.base_alpha,
        kind="center",
    )

    print(f"\n[1] load {args.model}...")
    t0 = time.time()
    model, tokenizer = load_model(args.model, dtype=args.dtype, device=args.device)
    for p in model.parameters():
        p.requires_grad_(False)
    scorer = Scorer(model, tokenizer)
    print(f"  ready in {time.time() - t0:.1f}s")

    print(f"\n[2] capture HS under S0=L{args.base_layer}k{args.base_rank} on layers {layers}...")
    rows, hs_lists = capture_condition(
        model, scorer, items, [base_spec], layers, log_every=args.log_every
    )
    H_by_L = stack_hs(hs_lists)
    families = np.array([r["scenario_family_id"] for r in rows], dtype=np.int64)
    train_fams, test_fams = family_split(list(families), seed=args.seed, train_frac=args.train_frac)
    print(f"  rows={len(rows)}  train_fams={len(train_fams)}  test_fams={len(test_fams)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for target in targets:
        print(f"\n[3] fit conditional INLP target={target} solver={solver} k_max={args.k_max}")
        if target == "gender_choice":
            y = np.array([r["y_gender"] for r in rows], dtype=np.int32)
        elif target == "slot_choice":
            y = np.array([r["y_slot"] for r in rows], dtype=np.int32)
        else:
            raise SystemExit(f"unknown target {target}")

        arrays: dict[str, np.ndarray] = {}
        layers_detail = []
        rng = np.random.default_rng(args.seed + hash(target) % 10_000)

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
                max_iter=args.max_iter,
                chance_auc=args.chance_auc,
                seed=args.seed + L,
            )
            W = fit["W"]
            c = fit["centers"]
            arrays[f"L{L}__W"] = W
            arrays[f"L{L}__centers"] = c
            k = int(W.shape[1])
            Wr = random_orthonormal(W.shape[0], k, rng).astype(np.float32)
            # centers for random on same steered train HS
            train_mask = np.array([fid in train_fams for fid in families])
            cr = subspace_centers(
                np.asarray(H_by_L[L][train_mask], dtype=np.float64),
                y[train_mask],
                Wr.astype(np.float64),
            ).astype(np.float32)
            arrays[f"L{L}__W_random_s0"] = Wr
            arrays[f"L{L}__centers_random_s0"] = cr

            powers = [1, 2, 4, 8, 16, 32, 64]
            steering_ranks = [r for r in powers if r <= k]
            if k not in steering_ranks:
                steering_ranks.append(k)
            layers_detail.append(
                {
                    "layer": L,
                    "k_found": k,
                    "k_chance": fit["k_chance"],
                    "auc_curve": fit["auc_curve"],
                    "steering_ranks": steering_ranks,
                    "random_control_ranks": steering_ranks,
                    "solver": fit["solver"],
                    "note": fit.get("note"),
                }
            )
            print(
                f"    k_found={k}  AUC(0)={fit['auc_curve'][0]:.3f}  "
                f"k_chance={fit['k_chance']}"
            )

        stem = f"inlp_cond_{target}_{args.tag}"
        npz_path = args.out_dir / f"{stem}.npz"
        meta = {
            "schema": "steering.conditional_inlp_subspace/v1",
            "target": target,
            "tag": args.tag,
            "datetime": datetime.now().isoformat(),
            "model": args.model,
            "conditioned_on": {
                "layer": args.base_layer,
                "rank": args.base_rank,
                "alpha": args.base_alpha,
                "subspaces": str(args.base_subspaces),
                "kind": "center",
            },
            "sample": str(args.sample),
            "n_rows": len(rows),
            "n_families": len(set(families.tolist())),
            "train_frac": args.train_frac,
            "seed": args.seed,
            "solver": solver,
            "k_max": args.k_max,
            "layers": layers,
            "layers_detail": layers_detail,
            "arrays_sha256": arrays_signature(arrays),
            "read": (
                "W fit on HS captured under S0. Use with run_stacked_inlp_stagea "
                "(always keep S0 on) — not vanilla run_inlp_stagea alone."
            ),
        }
        np.savez_compressed(npz_path, **arrays)
        (args.out_dir / f"{stem}.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        written.append(str(npz_path))
        print(f"  wrote {npz_path}")

    # save capture meta once
    cap_meta = {
        "schema": "steering.conditional_hs_capture/v1",
        "tag": args.tag,
        "n_rows": len(rows),
        "layers": layers,
        "base": {"layer": args.base_layer, "rank": args.base_rank},
        "written_subspaces": written,
    }
    (args.out_dir / f"cond_capture_{args.tag}.json").write_text(
        json.dumps(cap_meta, indent=2) + "\n", encoding="utf-8"
    )
    print("\n=== done ===")
    for w in written:
        print(f"  {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
