"""
sequential_pre_peak_erase — multi-site erase starting at the earliest
ascent-pool layer (≤ AUC peak), then peeling upward by synergy.

Does NOT replace the post-hook second_hit pipeline (L15→16–24).

Phases:
  1. Detect ascent pool from layer_scan AUC curve (τ / knee γ).
  2. Fit INLP subspaces on pool layers from baseline last-token HS.
  3. First hit = min(pool): screen ranks → keep S0.
  4. Later hits: screen remaining pool layers under current stack;
     add while synergy ≥ synergy_min.

Examples:

    # CPU: only write plan.json
    python -m steering.run_sequential_pre_peak_erase \\
        --config steering/configs/sequential_pre_peak_erase_gender_2b_v1.yaml \\
        --plan-only --tag prepeak_gender_plan

    # Full GPU run (smoke)
    python -m steering.run_sequential_pre_peak_erase \\
        --config steering/configs/sequential_pre_peak_erase_gender_2b_v1.yaml \\
        --device cuda --dtype float32 --n-items 2 --tag prepeak_gender_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from steering.ascent_pool import detect_ascent_pool, load_auc_curve
from steering.build_conditional_inlp_subspace import (
    arrays_signature,
    family_split,
    inlp_on_layer,
    random_orthonormal,
    subspace_centers,
)
from steering.intervene import Scorer, SubspaceSpec, load_model, steered
from steering.run_hs_recovery_auc import (
    capture_condition,
    pick_items,
    stack_hs,
)
from steering.run_inlp_stagea import behavioral_metrics, load_json, row_margins

REPO_ROOT = Path(__file__).resolve().parents[1]
STEERING_DIR = Path(__file__).resolve().parent


def resolve_path(p: str | Path | None) -> Path | None:
    if p is None or p == "null" or p == "":
        return None
    path = Path(p)
    if path.is_file():
        return path
    cand = REPO_ROOT / path
    if cand.is_file():
        return cand
    cand2 = STEERING_DIR / path
    if cand2.is_file():
        return cand2
    return path  # may not exist yet


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise SystemExit(f"bad config: {path}")
    return cfg


def curve_from_config(cfg: dict, *, layer_scan_override: Path | None) -> tuple[dict[int, float], str]:
    metric = str((cfg.get("layer_scan") or {}).get("metric") or "auto")
    if layer_scan_override is not None:
        return load_auc_curve(layer_scan_override, metric=metric), metric
    ls = cfg.get("layer_scan") or {}
    path = resolve_path(ls.get("path"))
    if path is not None and path.is_file():
        return load_auc_curve(path, metric=metric), metric
    if cfg.get("auc_curve"):
        return {int(k): float(v) for k, v in cfg["auc_curve"].items()}, metric
    raise SystemExit(
        "нет AUC/R²: укажите layer_scan.path, --layer-scan, или auc_curve в yaml"
    )


def primary_R(metrics: dict, primary: str) -> tuple[float, float]:
    """Return (R_mean, R_ci_lo) for gender or slot axis."""
    key = "gender_axis" if primary == "gender" else "slot_axis"
    block = metrics[key]["reduction"]
    return float(block["mean"]), float(block["ci_lo"])


def run_rows(scorer, model, items, specs, *, label: str, log_every: int) -> list[dict]:
    rows: list[dict] = []
    total = sum(len(i["rows"]) for i in items)
    t0 = time.time()
    for item in items:
        for src in item["rows"]:
            with steered(model, specs if specs else []):
                out = scorer.score(src["prompt"], list(src["valid_labels"]))
            labels = src["labels"]
            rows.append(
                {
                    "id": src["id"],
                    "scenario_family_id": item["scenario_family_id"],
                    "soc_major_title": item["soc_major_title"],
                    "profession": item.get("profession"),
                    "position_variant": src["position_variant"],
                    "context_order": src["context_order"],
                    "labels": labels,
                    "choice": out["choice"],
                    "logit_A": out["logit_A"],
                    "logit_B": out["logit_B"],
                    **row_margins(out, labels),
                    "baseline_choice_recorded": src.get("baseline_choice"),
                }
            )
            if log_every and len(rows) % log_every == 0:
                rate = len(rows) / max(time.time() - t0, 1e-9)
                eta = (total - len(rows)) / max(rate, 1e-9) / 60
                print(f"    [{label}] {len(rows)}/{total}  {rate:.2f} row/s  ETA {eta:.1f} min", flush=True)
    return rows


def make_spec(arrays: dict, layer: int, rank: int, alpha: float) -> SubspaceSpec:
    W = arrays[f"L{layer}__W"]
    c = arrays[f"L{layer}__centers"]
    k = min(int(rank), int(W.shape[1]))
    return SubspaceSpec(
        layer=layer,
        W=np.ascontiguousarray(W[:, :k], dtype=np.float32),
        c=np.ascontiguousarray(c[:k], dtype=np.float32),
        kind="center",
        alpha=float(alpha),
        label=f"inlp__L{layer}__k{k}",
    )


def fit_pool_subspaces(
    *,
    H_by_L: dict[int, np.ndarray],
    y: np.ndarray,
    families: np.ndarray,
    layers: list[int],
    k_max: int,
    solver: str,
    clf_C: float,
    seed: int,
    task: str = "classification",
    ridge_alpha: float = 1.0,
    chance_corr: float = 0.15,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    train_fams, test_fams = family_split(list(families), seed=seed, train_frac=0.7)
    arrays: dict[str, np.ndarray] = {}
    details = []
    rng = np.random.default_rng(seed)
    for L in layers:
        print(f"  fit L{L}...")
        fit = inlp_on_layer(
            H_by_L[L],
            y,
            families,
            train_fams,
            test_fams,
            k_max=k_max,
            solver=solver,
            clf_C=clf_C,
            max_iter=2000,
            chance_auc=0.55,
            seed=seed + L,
            task=task,
            ridge_alpha=ridge_alpha,
            chance_corr=chance_corr,
        )
        W, c = fit["W"], fit["centers"]
        arrays[f"L{L}__W"] = W
        arrays[f"L{L}__centers"] = c
        k = int(W.shape[1])
        Wr = random_orthonormal(W.shape[0], k, rng).astype(np.float32)
        train_mask = np.array([fid in train_fams for fid in families])
        from steering.build_conditional_inlp_subspace import _bin_y_for_centers

        cr = subspace_centers(
            np.asarray(H_by_L[L][train_mask], dtype=np.float64),
            _bin_y_for_centers(y[train_mask], task),
            Wr.astype(np.float64),
        ).astype(np.float32)
        arrays[f"L{L}__W_random_s0"] = Wr
        arrays[f"L{L}__centers_random_s0"] = cr
        details.append(
            {
                "layer": L,
                "k_found": k,
                "k_chance": fit["k_chance"],
                "auc_curve": fit["auc_curve"],
                "solver": fit["solver"],
                "task": fit.get("task", task),
            }
        )
        sc0 = fit["auc_curve"][0] if fit["auc_curve"] else float("nan")
        print(f"    k_found={k}  score(0)={sc0:.3f}  k_chance={fit['k_chance']}  task={task}")
    return arrays, details


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--layer-scan", type=Path, default=None, help="override AUC layer_scan JSON")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--n-items", type=int, default=None)
    ap.add_argument("--tag", default="prepeak_v1")
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "results" / "steering")
    ap.add_argument("--log-every", type=int, default=40)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-hits", type=int, default=None, help="cap number of sites in the stack")
    args = ap.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    cfg_path = resolve_path(args.config) or args.config
    if not cfg_path.is_file():
        raise SystemExit(f"MISSING config {args.config}")
    cfg = load_config(cfg_path)

    axis = str(cfg.get("axis", "gender"))
    target = str(cfg.get("target", "gender_prob"))
    primary = "slot" if axis == "slot" or "slot" in str(cfg.get("primary_metric", "")) else "gender"
    model_id = args.model or cfg.get("model") or "Qwen/Qwen3.5-2B-Base"
    is_prob = target in ("gender_prob", "slot_prob", "narrative_prob")
    task = "regression" if is_prob else "classification"

    curve, metric = curve_from_config(cfg, layer_scan_override=resolve_path(args.layer_scan))
    ap_cfg = dict(cfg.get("ascent_pool") or {})
    if is_prob and "chance" not in ap_cfg:
        ap_cfg["chance"] = 0.0
    if is_prob and "min_prev_auc_for_knee" not in ap_cfg:
        ap_cfg["min_prev_auc_for_knee"] = 0.05
    pool_res = detect_ascent_pool(
        curve,
        tau=float(ap_cfg.get("tau", 0.38)),
        gamma=float(ap_cfg.get("gamma", 0.25)),
        chance=float(ap_cfg.get("chance", 0.5 if not is_prob else 0.0)),
        metric=metric if metric != "auto" else ("val_r2" if is_prob else "val_roc_auc"),
        max_pool=int(ap_cfg.get("max_pool", 8)),
        min_prev_auc_for_knee=float(ap_cfg.get("min_prev_auc_for_knee", 0.65 if not is_prob else 0.05)),
    )

    out_dir = args.out_root / "sequential_pre_peak" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema": "steering.sequential_pre_peak_erase/v1",
        "tag": args.tag,
        "datetime": datetime.now().isoformat(),
        "config": str(cfg_path),
        "axis": axis,
        "target": target,
        "primary": primary,
        "model": model_id,
        "ascent_pool": pool_res.to_dict(),
        "ranks": cfg.get("ranks", [4, 8, 16]),
        "alphas": cfg.get("alphas", [1.0]),
        "synergy_min": float(cfg.get("synergy_min", 0.005)),
        "read": (
            "first_hit=min(pool). Later hits require synergy_vs_stack >= synergy_min. "
            "Post-peak layers are out of scope (use second_hit / SEQUENTIAL_ERASE)."
        ),
    }
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(
        f"ascent pool: peak=L{pool_res.peak_layer} ({pool_res.peak_auc:.3f})  "
        f"knee={pool_res.knee_layer}  pool={pool_res.pool}  first_hit=L{pool_res.first_hit}"
    )
    print(f"wrote {out_dir / 'plan.json'}")
    if args.plan_only:
        return 0

    ranks = [int(x) for x in cfg.get("ranks", [4, 8, 16])]
    alphas = [float(x) for x in cfg.get("alphas", [1.0])]
    synergy_min = float(cfg.get("synergy_min", 0.005))
    k_max = int(cfg.get("k_max_fit", 32))
    clf_C = float(cfg.get("clf_C", 0.03))
    sample_path = resolve_path(cfg.get("sample")) or (STEERING_DIR / "samples" / "h1_stagea_sample_v1.json")
    sample = load_json(sample_path)
    items = pick_items(sample["items"], args.n_items)
    pool = list(pool_res.pool)
    max_hits = args.max_hits if args.max_hits is not None else len(pool)

    solver = "ridge" if task == "regression" else "logistic"
    if solver == "logistic":
        try:
            import sklearn  # noqa: F401
        except ImportError:
            solver = "mean_diff"
            print("sklearn missing → mean_diff")
    else:
        try:
            import sklearn  # noqa: F401
        except ImportError:
            solver = "mean_diff"
            print("sklearn missing → mean_diff (binarized halves)")

    print(f"\n[1] load {model_id}...")
    t0 = time.time()
    model, tokenizer = load_model(model_id, dtype=args.dtype, device=args.device)
    for p in model.parameters():
        p.requires_grad_(False)
    scorer = Scorer(model, tokenizer)
    print(f"  ready in {time.time() - t0:.1f}s")

    print(f"\n[2] capture baseline HS on pool {pool}...")
    rows, hs_lists = capture_condition(
        model, scorer, items, None, pool, log_every=args.log_every
    )
    H_by_L = stack_hs(hs_lists)
    families = np.array([r["scenario_family_id"] for r in rows], dtype=np.int64)
    if target in ("slot_prob", "slot_choice"):
        if target == "slot_prob":
            y = np.array([r.get("p_A_norm", float("nan")) for r in rows], dtype=np.float64)
        else:
            y = np.array([r["y_slot"] for r in rows], dtype=np.float64)
    else:
        if target == "gender_prob":
            y = np.array([r.get("p_man_norm", float("nan")) for r in rows], dtype=np.float64)
        else:
            y = np.array([r["y_gender"] for r in rows], dtype=np.float64)
    if not np.isfinite(y).all():
        n_bad = int((~np.isfinite(y)).sum())
        raise SystemExit(f"target={target}: {n_bad} non-finite labels — check capture probs")

    print(f"\n[3] fit INLP on pool (baseline HS, target={target}, task={task})...")
    arrays, fit_details = fit_pool_subspaces(
        H_by_L=H_by_L,
        y=y,
        families=families,
        layers=pool,
        k_max=k_max,
        solver=solver,
        clf_C=clf_C,
        seed=args.seed,
        task=task,
        ridge_alpha=float(cfg.get("ridge_alpha", 1.0)),
        chance_corr=float(cfg.get("chance_corr", 0.15)),
    )
    npz_path = out_dir / f"inlp_prepeak_{target}_{args.tag}.npz"
    meta_sub = {
        "schema": "steering.prepeak_inlp_subspace/v1",
        "target": target,
        "tag": args.tag,
        "pool": pool,
        "layers_detail": fit_details,
        "arrays_sha256": arrays_signature(arrays),
        "conditioned_on": None,
    }
    np.savez_compressed(npz_path, **arrays)
    (out_dir / f"inlp_prepeak_{target}_{args.tag}.json").write_text(
        json.dumps(meta_sub, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  wrote {npz_path}")

    print("\n[4] baseline preference...")
    base_rows = run_rows(scorer, model, items, [], label="baseline", log_every=args.log_every)

    # --- first hit ---
    L0 = pool_res.first_hit
    if L0 not in pool:
        L0 = pool[0]
    print(f"\n[5] first hit screen L{L0} ranks={ranks}...")
    first_candidates = []
    for k in ranks:
        if f"L{L0}__W" not in arrays or arrays[f"L{L0}__W"].shape[1] < 1:
            continue
        k_eff = min(k, arrays[f"L{L0}__W"].shape[1])
        for a in alphas:
            first_candidates.append((k_eff, a))
    # unique
    seen = set()
    uniq = []
    for k, a in first_candidates:
        if (k, a) in seen:
            continue
        seen.add((k, a))
        uniq.append((k, a))

    best = None
    hit_log = []
    for k, a in uniq:
        sp = make_spec(arrays, L0, k, a)
        cid = f"hit0__L{L0}__k{k}__a{a:g}"
        print(f"  {cid}")
        rows_s = run_rows(scorer, model, items, [sp], label=cid, log_every=args.log_every)
        m = behavioral_metrics(base_rows, rows_s, n_boot=args.n_boot, seed=args.seed)
        R, ci_lo = primary_R(m, primary)
        entry = {
            "id": cid,
            "layer": L0,
            "rank": k,
            "alpha": a,
            "R_primary": R,
            "R_ci_lo": ci_lo,
            "R_gender": m["gender_axis"]["reduction"]["mean"],
            "R_slot": m["slot_axis"]["reduction"]["mean"],
            "synergy_vs_prev": None,
        }
        hit_log.append(entry)
        print(f"    R_primary={R:.4f}  ci_lo={ci_lo:.4f}")
        if ci_lo > 0 and (best is None or R > best["R_primary"]):
            best = {**entry, "spec": sp}

    if best is None:
        # take max R even if CI straddles 0
        best_row = max(hit_log, key=lambda e: e["R_primary"]) if hit_log else None
        if best_row is None:
            raise SystemExit("first hit: нет кандидатов")
        best = {
            **best_row,
            "spec": make_spec(arrays, L0, best_row["rank"], best_row["alpha"]),
        }
        print(f"  WARN: no CI_lo>0; keeping best R={best['R_primary']:.4f}")

    stack_specs = [best["spec"]]
    stack_meta = [
        {"layer": best["layer"], "rank": best["rank"], "alpha": best["alpha"], "R_primary": best["R_primary"]}
    ]
    R_stack = best["R_primary"]
    print(f"  KEEP S0 = L{best['layer']} k{best['rank']}  R={R_stack:.4f}")

    # --- subsequent hits ---
    remaining = [L for L in pool if L != L0]
    for L in remaining:
        if len(stack_specs) >= max_hits:
            print(f"\nstop: max_hits={max_hits}")
            break
        print(f"\n[next] screen L{L} under stack {[(s.layer, s.rank) for s in stack_specs]}...")
        layer_best = None
        for k in ranks:
            if arrays[f"L{L}__W"].shape[1] < 1:
                continue
            k_eff = min(k, arrays[f"L{L}__W"].shape[1])
            for a in alphas:
                sp = make_spec(arrays, L, k_eff, a)
                cid = f"stack+L{L}__k{k_eff}__a{a:g}"
                rows_s = run_rows(
                    scorer, model, items, stack_specs + [sp], label=cid, log_every=args.log_every
                )
                m = behavioral_metrics(base_rows, rows_s, n_boot=args.n_boot, seed=args.seed)
                R, ci_lo = primary_R(m, primary)
                syn = R - R_stack
                entry = {
                    "id": cid,
                    "layer": L,
                    "rank": k_eff,
                    "alpha": a,
                    "R_primary": R,
                    "R_ci_lo": ci_lo,
                    "R_gender": m["gender_axis"]["reduction"]["mean"],
                    "R_slot": m["slot_axis"]["reduction"]["mean"],
                    "synergy_vs_prev": syn,
                }
                hit_log.append(entry)
                print(f"  {cid}  R={R:.4f}  synergy={syn:+.4f}")
                if syn >= synergy_min and ci_lo > 0:
                    if layer_best is None or syn > layer_best["synergy_vs_prev"]:
                        layer_best = {**entry, "spec": sp}
        if layer_best is None:
            print(f"  no synergy ≥ {synergy_min} on L{L} → skip")
            continue
        stack_specs.append(layer_best["spec"])
        stack_meta.append(
            {
                "layer": layer_best["layer"],
                "rank": layer_best["rank"],
                "alpha": layer_best["alpha"],
                "R_primary": layer_best["R_primary"],
                "synergy_vs_prev": layer_best["synergy_vs_prev"],
            }
        )
        R_stack = layer_best["R_primary"]
        print(f"  ADD L{layer_best['layer']} k{layer_best['rank']}  stack R={R_stack:.4f}")

    summary = {
        **plan,
        "finished": datetime.now().isoformat(),
        "subspaces": str(npz_path),
        "stack": stack_meta,
        "final_R_primary": R_stack,
        "n_families": len(items),
        "hit_log": [{k: v for k, v in e.items() if k != "spec"} for e in hit_log],
        "stop_reason": "pool_exhausted_or_max_hits",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    (out_dir / "stack.json").write_text(
        json.dumps({"stack": stack_meta, "primary": primary, "final_R_primary": R_stack}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\n=== final stack ===")
    for s in stack_meta:
        print(f"  L{s['layer']} k{s['rank']}  R={s['R_primary']:.4f}  syn={s.get('synergy_vs_prev')}")
    print(f"wrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
