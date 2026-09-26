"""Pooled polarity XY-control Stage A.

One v_raw on all train families of a polarity set; α/layer chosen on pooled val.

    python -m steering.xy_control.run_polarity_pool \
      --set-id promale --scale 2b --device cuda \
      --config steering/xy_control/configs/xy_control_2b_peak_prepeak_a6.yaml \
      --tag xy_2b_peak_prepeak_a6_pool_promale
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from steering.intervene import Scorer, load_model, num_hidden_layers, specs_for_candidate
from steering.run_h1_stagea import hook_check
from steering.run_hs_recovery_auc import pick_items
from steering.run_inlp_stagea import behavioral_metrics, family_aggregate
from steering.xy_control.build_vectors import capture_pair, sha256_file
from steering.xy_control.build_vectors_per_soc import _load_dotenv, _probe_paths
from steering.xy_control.capture_io import save_pair_capture
from steering.xy_control.mapping import MAPPING
from steering.xy_control.paired_sample import (
    FULL_SPLIT,
    dataset_provenance,
    filter_soc,
    load_pooled_items,
    split_path,
)
from steering.xy_control.per_soc import eval_ids_for_split
from steering.xy_control.polarity_pool import (
    SETS_JSON,
    expand_pool_candidates,
    fit_pool_vectors,
    load_set_bundle,
    plan_pool_split,
    vector_id_for_pool,
)
from steering.xy_control.run_stagea import BASELINE_ID, filter_items, gap_block, run_config

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
PATHS = {
    "2b": {"model": "Qwen/Qwen3.5-2B-Base"},
    "4b": {"model": "Qwen/Qwen3.5-4B-Base"},
}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _best_of(ranking: list[dict], *, prior_only: bool) -> dict | None:
    cands = [r for r in ranking if r.get("role") == "candidate"]
    if prior_only:
        cands = [r for r in cands if r.get("alpha_matches_prior")]
    if not cands:
        return None
    return max(cands, key=lambda r: float(r.get("gap_abs_reduction") or 0))


def run_one(args: argparse.Namespace) -> Path:
    scale = args.scale
    defaults = PATHS[scale]
    cfg_path = args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    model_id = args.model or cfg["source"]["model_id"] or defaults["model"]
    split_cfg = cfg["source"]["split"]
    train_frac = float(args.train_frac if args.train_frac is not None else split_cfg["train_frac"])
    val_frac = float(args.val_frac if args.val_frac is not None else split_cfg["val_frac"])
    seed = int(args.seed if args.seed is not None else split_cfg["seed"])

    doc, set_entry, soc_domains, domain = load_set_bundle(
        set_id=args.set_id,
        scale=scale,
        sets_path=args.sets,
        catalog_path=args.catalog,
    )
    tag = args.tag or set_entry.get("tag_stage_a") or f"xy_{scale}_pool_{args.set_id}"
    if args.smoke and not tag.endswith("_smoke"):
        tag = f"{tag}_smoke"
    eval_split = "all" if args.smoke and args.eval_split in ("heldout", "val") else args.eval_split

    dataset_path = split_path(FULL_SPLIT)
    dataset_meta = dataset_provenance(dataset_path)
    pooled = load_pooled_items()
    n_items = args.limit_items if args.limit_items is not None else (3 if args.smoke else None)

    titles = {d["soc_major_title"] for d in soc_domains}
    pool_items: list[dict] = []
    seen: set[int] = set()
    for title in sorted(titles):
        for it in pick_items(filter_soc(pooled, title), n_items):
            fid = int(it["scenario_family_id"])
            if fid in seen:
                continue
            seen.add(fid)
            pool_items.append(it)
    pool_items.sort(key=lambda it: int(it["scenario_family_id"]))
    if len(pool_items) < 4:
        raise SystemExit(f"pool {args.set_id}: too few families ({len(pool_items)})")

    split_meta = plan_pool_split(
        pool_items, seed=seed, train_frac=train_frac, val_frac=val_frac
    )
    train_fams = set(split_meta["train_family_ids"])
    val_fams = set(split_meta["val_family_ids"])
    test_fams = set(split_meta["test_family_ids"])
    capture_items = [it for it in pool_items if int(it["scenario_family_id"]) in train_fams]
    capture_items.sort(key=lambda it: int(it["scenario_family_id"]))

    out_root = args.out_root / "polarity_pool" / tag
    out_root.mkdir(parents=True, exist_ok=True)

    from steering.contrastive.build_vectors import load_probe_bank, needed_layers

    layers = needed_layers(cfg)
    if args.smoke:
        layers = [int(cfg["layers"]["anchor"])]

    vectors_path = args.vectors or (
        HERE / "vectors" / f"xy_control_{scale}_polarity_pool_vectors_{tag}.npz"
    )

    _load_dotenv()
    print(f"\n[{scale}/{args.set_id}] load {model_id}...")
    t_load = time.time()
    model, tokenizer = load_model(model_id, dtype=args.dtype, device=args.device)
    for p in model.parameters():
        p.requires_grad_(False)
    scorer = Scorer(model, tokenizer)
    print(f"  ready in {time.time() - t_load:.1f}s, {num_hidden_layers(model.config)} blocks")
    print(
        f"\n[{scale}/{args.set_id}] capture {len(capture_items)} train families "
        f"on layers {layers} (pooled {domain['label']})..."
    )
    rows, hs_g_lists, hs_xy_lists = capture_pair(
        model, scorer, capture_items, layers, log_every=args.log_every
    )
    H_g = {L: np.stack(vs, axis=0) for L, vs in hs_g_lists.items()}
    H_xy = {L: np.stack(vs, axis=0) for L, vs in hs_xy_lists.items()}
    if not args.no_save_capture:
        cap_path = HERE / "captures" / f"xy_control_{scale}_polarity_pool_capture_{tag}.npz"
        save_pair_capture(
            cap_path,
            H_gender=H_g,
            H_xy=H_xy,
            rows=rows,
            meta={
                "schema": "steering.xy_control_polarity_pool_capture/v1",
                "scale": scale,
                "set_id": args.set_id,
                "model_id": model_id,
                "tag": tag,
                "mapping": dict(MAPPING),
            },
        )
        print(f"  capture → {cap_path}")

    probe_bank = load_probe_bank(_probe_paths(scale, None))
    print(f"\n[{scale}/{args.set_id}] fit ONE pooled v_raw on train={len(train_fams)}")
    vectors, entries = fit_pool_vectors(
        cfg,
        H_g,
        H_xy,
        rows,
        domain,
        train_fams=train_fams,
        val_fams=val_fams,
        test_fams=test_fams,
        probe_bank=probe_bank,
    )
    calib = {e["key"]: e for e in entries}
    from steering.build_conditional_inlp_subspace import arrays_signature

    vec_meta = {
        "schema": "steering.xy_control_polarity_pool_vectors/v1",
        "scale": scale,
        "set_id": args.set_id,
        "polarity": domain["polarity"],
        "model_id": model_id,
        "map_config": cfg_path.name,
        "map_sha256": sha256_file(cfg_path),
        "layers": layers,
        "mapping": dict(MAPPING),
        **dataset_meta,
        "n_vectors": len(vectors),
        "arrays_sha256": arrays_signature(vectors) if vectors else "",
        "vectors": entries,
        "domains": [
            {
                "slug": domain["slug"],
                "set_id": args.set_id,
                "polarity": domain["polarity"],
                "label": domain["label"],
                "member_slugs": domain["member_slugs"],
                "member_titles": domain["member_titles"],
                "vector_id": vector_id_for_pool(domain["polarity"]),
                **{k: split_meta[k] for k in (
                    "train_family_ids",
                    "val_family_ids",
                    "test_family_ids",
                    "n_families",
                    "n_train",
                    "n_val",
                    "n_test",
                    "per_soc",
                    "split_rule",
                )},
            }
        ],
    }
    HERE.joinpath("vectors").mkdir(parents=True, exist_ok=True)
    np.savez(vectors_path, **vectors)
    vectors_path.with_suffix(".json").write_text(
        json.dumps(vec_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  vectors → {vectors_path}")

    eval_ids = eval_ids_for_split(train_fams, val_fams, test_fams, eval_split)
    items = filter_items(pool_items, eval_ids)
    if not items:
        raise SystemExit(f"empty eval for {args.set_id} split={eval_split}")

    candidates = expand_pool_candidates(domain, cfg, smoke=args.smoke)
    missing = [
        c["id"] for c in candidates if f"{c['vector_id']}__L{c['layers'][0]}" not in vectors
    ]
    if missing:
        raise SystemExit(f"missing vectors for {missing[:3]}")

    configs: list[tuple[str, dict | None]] = [(BASELINE_ID, None)]
    configs += [(c["id"], c) for c in candidates]
    n_rows = sum(len(i["rows"]) for i in items)
    print(
        f"\n[{scale}/{args.set_id}] Stage A pooled val\n"
        f"  {len(configs)} configs × {n_rows} rows  "
        f"({len(items)} families, split={eval_split})  "
        f"prior α {domain['hypothesized_alpha_sign']}"
    )

    pool_dir = out_root / domain["slug"]
    pool_dir.mkdir(parents=True, exist_ok=True)
    ranking: list[dict] = []
    baseline_rows: list[dict] | None = None
    t_all = time.time()

    for idx, (cfg_id, cand) in enumerate(configs, 1):
        specs = specs_for_candidate(cand, vectors, calib) if cand else []
        print(f"  [{idx}/{len(configs)}] {cfg_id}")
        t0 = time.time()
        rows_eval = run_config(
            scorer, model, items, specs, label=cfg_id, log_every=args.log_every
        )
        runtime = time.time() - t0
        fam = family_aggregate(rows_eval)
        gaps = gap_block(fam)
        paired = (
            None
            if baseline_rows is None
            else behavioral_metrics(baseline_rows, rows_eval, n_boot=1000, seed=0)
        )
        if cfg_id == BASELINE_ID:
            baseline_rows = rows_eval
        base_gap = gap_block(family_aggregate(baseline_rows))["gender_gap"] if baseline_rows else 0.0
        gap_abs_reduction = abs(base_gap) - abs(gaps["gender_gap"])
        metrics = {
            "config_id": cfg_id,
            "role": cand["role"] if cand else "baseline",
            "candidate": cand,
            "runtime_s": round(runtime, 1),
            "hook_check": hook_check(rows_eval),
            "mapping": dict(MAPPING),
            "eval_split": eval_split,
            "scale": scale,
            "set_id": args.set_id,
            "slug": domain["slug"],
            "polarity": domain["polarity"],
            "layer": cand["layers"][0] if cand else "",
            "alpha": cand["alpha"] if cand else "",
            "alpha_matches_prior": cand["alpha_matches_prior"] if cand else "",
            "gender_gap": gaps["gender_gap"],
            "mean_abs_delta": gaps["mean_abs_delta"],
            "p_delta_positive": gaps["p_delta_positive"],
            "slot_gap": gaps["slot_gap"],
            "gap_abs_reduction": gap_abs_reduction,
            "n_eval_families": len(items),
            "n_train_families": len(train_fams),
            "paired": paired,
        }
        ranking.append(
            {
                "config_id": cfg_id,
                "role": metrics["role"],
                "layer": metrics["layer"],
                "alpha": metrics["alpha"],
                "alpha_matches_prior": metrics["alpha_matches_prior"],
                "gender_gap": metrics["gender_gap"],
                "mean_abs_delta": metrics["mean_abs_delta"],
                "p_delta_positive": metrics["p_delta_positive"],
                "slot_gap": metrics["slot_gap"],
                "gap_abs_reduction": metrics["gap_abs_reduction"],
            }
        )
        (pool_dir / cfg_id).mkdir(parents=True, exist_ok=True)
        (pool_dir / cfg_id / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"    GenderGap {gaps['gender_gap']:+.4f}  mean|Δ| {gaps['mean_abs_delta']:.4f}  "
            f"P(Δ>0) {gaps['p_delta_positive']:.3f}  {runtime:.1f}s"
        )

    _write_csv(pool_dir / "ranking.csv", ranking)
    best = _best_of(ranking, prior_only=False)
    best_prior = _best_of(ranking, prior_only=True)
    base = next((r for r in ranking if r["config_id"] == BASELINE_ID), None)
    summary = {
        "scale": scale,
        "set_id": args.set_id,
        "slug": domain["slug"],
        "soc_major_title": domain["label"],
        "polarity": domain["polarity"],
        "n_soc": domain["n_soc"],
        "member_slugs": ",".join(domain["member_slugs"]),
        "n_train_families": len(train_fams),
        "n_eval_families": len(items),
        "n_test_families": len(test_fams),
        "baseline_gender_gap": base["gender_gap"] if base else "",
        "best_id": best["config_id"] if best else "",
        "best_layer": best["layer"] if best else "",
        "best_alpha": best["alpha"] if best else "",
        "best_gender_gap": best["gender_gap"] if best else "",
        "best_gap_abs_reduction": best["gap_abs_reduction"] if best else "",
        "best_matches_prior": best["alpha_matches_prior"] if best else "",
        "best_prior_id": best_prior["config_id"] if best_prior else "",
        "best_prior_alpha": best_prior["alpha"] if best_prior else "",
        "best_prior_gap_abs_reduction": best_prior["gap_abs_reduction"] if best_prior else "",
    }
    _write_csv(out_root / "summary.csv", [summary])
    _write_csv(pool_dir / "summary.csv", [summary])
    shutil.copy2(vectors_path, out_root / vectors_path.name)
    shutil.copy2(vectors_path.with_suffix(".json"), out_root / vectors_path.with_suffix(".json").name)
    shutil.copy2(dataset_path, out_root / dataset_path.name)
    (out_root / "set.json").write_text(
        json.dumps(
            {
                "schema": "steering.xy_control_polarity_pool_stage_a/v1",
                "sets_doc": str(args.sets or SETS_JSON),
                "set": set_entry,
                "domain": domain,
                "split": split_meta,
                "config": str(cfg_path),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_root / "run_meta.json").write_text(
        json.dumps(
            {
                "schema": "steering.xy_control_polarity_pool_stage_a/v1",
                "datetime": datetime.now().isoformat(),
                "scale": scale,
                "set_id": args.set_id,
                "model": model_id,
                "tag": tag,
                "eval_split": eval_split,
                "protocol": "pooled_polarity_one_v",
                "runtime_s": round(time.time() - t_all, 1),
                **dataset_meta,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"\n=== polarity pool Stage A [{tag}] {domain['label']}  "
        f"train={len(train_fams)} val={len(val_fams)} test={len(test_fams)} ==="
    )
    print(f"  {out_root}")
    print(
        f"  best {summary['best_id']}  Δ|G| {summary['best_gap_abs_reduction']}  "
        f"base {summary['baseline_gender_gap']}"
    )
    return out_root


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set-id", required=True, help="promale | profemale | …")
    ap.add_argument("--scale", choices=["2b", "4b"], default="2b")
    ap.add_argument("--sets", type=Path, default=SETS_JSON)
    ap.add_argument("--catalog", type=Path, default=None)
    ap.add_argument(
        "--config",
        type=Path,
        default=HERE / "configs" / "xy_control_2b_peak_prepeak_a6.yaml",
    )
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--limit-items", type=int, default=None)
    ap.add_argument("--eval-split", choices=["heldout", "val", "test", "train", "all"], default="val")
    ap.add_argument("--train-frac", type=float, default=None)
    ap.add_argument("--val-frac", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--vectors", type=Path, default=None)
    ap.add_argument("--no-save-capture", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "results" / "steering" / "xy_control")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--log-every", type=int, default=40)
    args = ap.parse_args(argv)
    run_one(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
