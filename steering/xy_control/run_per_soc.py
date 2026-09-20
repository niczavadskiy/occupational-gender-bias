"""Per-SOC XY-control Stage A: fit v and steer each FDR-significant domain.

Non-significant SOCs are skipped (identity). Male and female domains each
get their own v_raw. Models run in order: 2B then 4B.

    python -m steering.xy_control.run_per_soc --device cuda
    python -m steering.xy_control.run_per_soc --scale 2b --device cuda
    python -m steering.xy_control.run_per_soc --scale 4b --smoke --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from steering.intervene import Scorer, load_model, num_hidden_layers, specs_for_candidate
from steering.run_h1_stagea import hook_check
from steering.run_hs_recovery_auc import pick_items
from steering.run_inlp_stagea import behavioral_metrics, family_aggregate
from steering.xy_control.build_vectors import CONFIG_BY_SCALE, capture_pair, sha256_file
from steering.xy_control.build_vectors_per_soc import _load_dotenv, _probe_paths
from steering.xy_control.capture_io import load_pair_capture, save_pair_capture
from steering.xy_control.domains import CATALOG_JSON, load_catalog
from steering.xy_control.mapping import MAPPING
from steering.xy_control.paired_sample import filter_soc, load_pooled_items
from steering.xy_control.per_soc import (
    eval_ids_for_split,
    expand_soc_candidates,
    fit_domain_vectors,
    plan_domain_splits,
    skip_row,
    train_items_from_plan,
    vector_id_for,
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


def run_scale(args: argparse.Namespace, scale: str) -> Path:
    defaults = PATHS[scale]
    cfg_path = args.config or CONFIG_BY_SCALE[scale]
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    model_id = args.model or cfg["source"]["model_id"] or defaults["model"]
    split_cfg = cfg["source"]["split"]
    train_frac = float(args.train_frac if args.train_frac is not None else split_cfg["train_frac"])
    val_frac = float(args.val_frac if args.val_frac is not None else split_cfg["val_frac"])
    seed = int(args.seed if args.seed is not None else split_cfg["seed"])
    if args.tag and args.scale == "both":
        tag = f"{args.tag}_{scale}"
    else:
        tag = args.tag or f"xy_{scale}_per_soc_v1"
    if args.smoke and not tag.endswith("_smoke"):
        tag = f"{tag}_smoke"
    eval_split = "all" if args.smoke and args.eval_split in ("heldout", "val") else args.eval_split

    catalog = load_catalog(args.catalog)
    block = catalog["scales"][scale]
    wanted = {s.strip() for s in args.only_soc.split(",")} if args.only_soc else None
    steer = list(block["steer"])
    if wanted:
        steer = [d for d in steer if d["slug"] in wanted]
    if args.smoke:
        steer = steer[: args.max_domains or 1]
    elif args.max_domains is not None:
        steer = steer[: args.max_domains]

    pooled = load_pooled_items()
    n_items = args.limit_items if args.limit_items is not None else (3 if args.smoke else None)

    skip_rows: list[dict] = []
    for d in block["skip"]:
        skip_rows.append(skip_row(scale=scale, title=d["soc_major_title"], reason="not_significant"))
    if wanted or args.max_domains is not None or args.smoke:
        selected = {d["slug"] for d in steer}
        for d in block["steer"]:
            if d["slug"] not in selected:
                skip_rows.append(
                    skip_row(scale=scale, title=d["soc_major_title"], reason="not_selected")
                )

    out_root = args.out_root / "per_soc" / tag
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "skip.csv").parent.mkdir(parents=True, exist_ok=True)

    from steering.contrastive.build_vectors import load_probe_bank, needed_layers

    layers = needed_layers(cfg)
    if args.smoke:
        layers = [int(cfg["layers"]["anchor"])]

    soc_pool: list[dict] = []
    seen: set[int] = set()
    for domain in steer:
        soc_items = pick_items(filter_soc(pooled, domain["soc_major_title"]), n_items)
        for it in soc_items:
            fid = int(it["scenario_family_id"])
            if fid in seen:
                continue
            seen.add(fid)
            soc_pool.append(it)
    soc_pool.sort(key=lambda it: int(it["scenario_family_id"]))
    planned, too_few = plan_domain_splits(
        soc_pool, steer, seed=seed, train_frac=train_frac, val_frac=val_frac
    )
    for domain, n_fams in too_few:
        skip_rows.append(
            skip_row(
                scale=scale,
                title=domain["soc_major_title"],
                reason="too_few_families",
                extra={"n_families": n_fams, "polarity": domain["polarity"]},
            )
        )
        print(f"  SKIP {domain['soc_major_title']}: {n_fams} families")
    capture_items = train_items_from_plan(soc_pool, planned)

    vec_tag = args.vectors_tag or tag
    vectors_path = args.vectors or (
        HERE / "vectors" / f"xy_control_{scale}_per_soc_vectors_{vec_tag}.npz"
    )
    vectors: dict[str, np.ndarray] = {}
    calib: dict[str, dict] = {}
    domain_splits: dict[str, dict] = {}
    vec_meta: dict = {}

    if args.from_vectors and not args.smoke:
        if not vectors_path.is_file() or not vectors_path.with_suffix(".json").is_file():
            raise SystemExit(f"нет векторов {vectors_path}")
        reuse_vectors = True
    else:
        reuse_vectors = False
    if reuse_vectors:
        print(f"\n[{scale}] reuse vectors {vectors_path}")
        vec_meta = json.loads(vectors_path.with_suffix(".json").read_text(encoding="utf-8"))
        with np.load(vectors_path) as z:
            vectors = {k: np.asarray(z[k], dtype=np.float32) for k in z.files}
        calib = {e["key"]: e for e in vec_meta["vectors"]}
        for dmeta in vec_meta.get("domains", []):
            domain_splits[dmeta["slug"]] = dmeta
        H_g = H_xy = rows = None
        model = tokenizer = scorer = None
    else:
        _load_dotenv()
        if args.from_capture is not None:
            print(f"\n[{scale}] load capture {args.from_capture}")
            H_g, H_xy, rows, _ = load_pair_capture(args.from_capture)
            print(f"\n[{scale}] load {model_id} for eval...")
            t_load = time.time()
            model, tokenizer = load_model(model_id, dtype=args.dtype, device=args.device)
            scorer = Scorer(model, tokenizer)
            print(f"  ready in {time.time() - t_load:.1f}s, {num_hidden_layers(model.config)} blocks")
        else:
            if not capture_items:
                raise SystemExit(f"[{scale}] no families to capture")
            print(f"\n[{scale}] load {model_id}...")
            t_load = time.time()
            model, tokenizer = load_model(model_id, dtype=args.dtype, device=args.device)
            for p in model.parameters():
                p.requires_grad_(False)
            scorer = Scorer(model, tokenizer)
            print(f"  ready in {time.time() - t_load:.1f}s, {num_hidden_layers(model.config)} blocks")
            print(
                f"\n[{scale}] capture {len(capture_items)} families on layers {layers}..."
            )
            rows, hs_g_lists, hs_xy_lists = capture_pair(
                model, scorer, capture_items, layers, log_every=args.log_every
            )
            H_g = {L: np.stack(vs, axis=0) for L, vs in hs_g_lists.items()}
            H_xy = {L: np.stack(vs, axis=0) for L, vs in hs_xy_lists.items()}
            if not args.no_save_capture:
                cap_path = (
                    HERE / "captures" / f"xy_control_{scale}_per_soc_capture_{tag}.npz"
                )
                save_pair_capture(
                    cap_path,
                    H_gender=H_g,
                    H_xy=H_xy,
                    rows=rows,
                    meta={
                        "schema": "steering.xy_control_per_soc_capture/v1",
                        "scale": scale,
                        "model_id": model_id,
                        "tag": tag,
                        "mapping": dict(MAPPING),
                    },
                )
                print(f"  capture → {cap_path}")

        probe_bank = load_probe_bank(_probe_paths(scale, None))
        print(f"\n[{scale}] fit v_raw per significant SOC")
        for domain in steer:
            title = domain["soc_major_title"]
            split_meta = planned.get(domain["slug"])
            if split_meta is None:
                continue
            train_fams = set(split_meta["train_family_ids"])
            val_fams = set(split_meta["val_family_ids"])
            test_fams = set(split_meta["test_family_ids"])
            arr, ent = fit_domain_vectors(
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
            vectors.update(arr)
            for e in ent:
                calib[e["key"]] = e
            domain_splits[domain["slug"]] = {
                "train_family_ids": split_meta["train_family_ids"],
                "val_family_ids": split_meta["val_family_ids"],
                "test_family_ids": split_meta["test_family_ids"],
                "n_families": split_meta["n_families"],
            }
            print(
                f"  {domain['polarity']:<7} {title}: "
                f"n={split_meta['n_families']} train={len(train_fams)} "
                f"val={len(val_fams)} test={len(test_fams)}"
            )

        from steering.build_conditional_inlp_subspace import arrays_signature

        vec_meta = {
            "schema": "steering.xy_control_per_soc_vectors/v1",
            "scale": scale,
            "model_id": model_id,
            "map_config": cfg_path.name,
            "map_sha256": sha256_file(cfg_path),
            "layers": layers,
            "mapping": dict(MAPPING),
            "n_vectors": len(vectors),
            "arrays_sha256": arrays_signature(vectors) if vectors else "",
            "vectors": list(calib.values()),
            "domains": [
                {"slug": slug, **split, "vector_id": vector_id_for(slug)}
                for slug, split in domain_splits.items()
            ],
        }
        HERE.joinpath("vectors").mkdir(parents=True, exist_ok=True)
        np.savez(vectors_path, **vectors)
        vectors_path.with_suffix(".json").write_text(
            json.dumps(vec_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"  vectors → {vectors_path}")

    if model is None:
        _load_dotenv()
        print(f"\n[{scale}] load {model_id} for eval...")
        t_load = time.time()
        model, tokenizer = load_model(model_id, dtype=args.dtype, device=args.device)
        scorer = Scorer(model, tokenizer)
        print(f"  ready in {time.time() - t_load:.1f}s")

    ranking_all: list[dict] = []
    summaries: list[dict] = []
    t_all = time.time()

    for di, domain in enumerate(steer, 1):
        title = domain["soc_major_title"]
        slug = domain["slug"]
        split_meta = domain_splits.get(slug)
        if split_meta is None:
            print(f"\n[{scale} {di}/{len(steer)}] SKIP {title}: no vector")
            continue
        soc_items = filter_soc(soc_pool, title)
        soc_items = pick_items(soc_items, n_items)
        eval_ids = eval_ids_for_split(
            set(split_meta["train_family_ids"]),
            set(split_meta["val_family_ids"]),
            set(split_meta["test_family_ids"]),
            eval_split,
        )
        items = filter_items(soc_items, eval_ids)
        if not items:
            skip_rows.append(
                skip_row(scale=scale, title=title, reason="empty_eval", extra={"split": eval_split})
            )
            print(f"\n[{scale} {di}/{len(steer)}] SKIP {title}: empty eval")
            continue

        candidates = expand_soc_candidates(domain, cfg, smoke=args.smoke)
        missing = [
            c["id"]
            for c in candidates
            if f"{c['vector_id']}__L{c['layers'][0]}" not in vectors
        ]
        if missing:
            raise SystemExit(f"{title}: missing vectors for {missing[:3]}")

        configs: list[tuple[str, dict | None]] = [(BASELINE_ID, None)]
        configs += [(c["id"], c) for c in candidates]
        n_rows = sum(len(i["rows"]) for i in items)
        print(
            f"\n[{scale} {di}/{len(steer)}] {domain['polarity']}  {title}\n"
            f"  {len(configs)} configs × {n_rows} rows  "
            f"({len(items)} families, split={eval_split})  prior α {domain['hypothesized_alpha_sign']}"
        )

        soc_dir = out_root / slug
        soc_dir.mkdir(parents=True, exist_ok=True)
        ranking: list[dict] = []
        baseline_rows: list[dict] | None = None

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
            paired = None if baseline_rows is None else behavioral_metrics(
                baseline_rows, rows_eval, n_boot=1000, seed=0
            )
            if cfg_id == BASELINE_ID:
                baseline_rows = rows_eval

            metrics = {
                "config_id": cfg_id,
                "role": cand["role"] if cand else "baseline",
                "candidate": cand,
                "runtime_s": round(runtime, 1),
                "hook_check": hook_check(rows_eval),
                "mapping": dict(MAPPING),
                "eval_split": eval_split,
                "scale": scale,
                "soc_major_title": title,
                "slug": slug,
                "polarity": domain["polarity"],
                "hypothesized_alpha_sign": domain["hypothesized_alpha_sign"],
                "n_items": len(items),
                "n_rows": len(rows_eval),
                **gaps,
                "paired": paired,
            }
            cdir = soc_dir / cfg_id
            cdir.mkdir(parents=True, exist_ok=True)
            with (cdir / "per_item.jsonl").open("w", encoding="utf-8") as f:
                for r in rows_eval:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            (cdir / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"    GenderGap {gaps['gender_gap']:+.4f}  mean|Δ| {gaps['mean_abs_delta']:.4f}  "
                f"P(Δ>0) {gaps['p_delta_positive']:.3f}  {runtime:.1f}s"
            )
            ranking.append(
                {
                    "scale": scale,
                    "soc_major_title": title,
                    "slug": slug,
                    "polarity": domain["polarity"],
                    "gate": domain["gate"],
                    "h1_effect": domain["effect"],
                    "hypothesized_alpha_sign": domain["hypothesized_alpha_sign"],
                    "config_id": cfg_id,
                    "role": metrics["role"],
                    "family": cand["family"] if cand else "",
                    "vector_id": cand["vector_id"] if cand else "",
                    "layer": cand["layers"][0] if cand else "",
                    "intervention": cand["intervention"] if cand else "",
                    "alpha": cand.get("alpha") if cand else 0.0,
                    "alpha_matches_prior": cand.get("alpha_matches_prior") if cand else "",
                    "gender_gap": gaps["gender_gap"],
                    "mean_abs_delta": gaps["mean_abs_delta"],
                    "p_delta_positive": gaps["p_delta_positive"],
                    "slot_gap": gaps["slot_gap"],
                    "n_eval_families": len(items),
                    "n_train_families": len(split_meta["train_family_ids"]),
                }
            )

        base = next((r for r in ranking if r["config_id"] == BASELINE_ID), None)
        for r in ranking:
            if base is not None:
                r["gap_vs_baseline"] = r["gender_gap"] - base["gender_gap"]
                r["gap_abs_reduction"] = abs(base["gender_gap"]) - abs(r["gender_gap"])
            else:
                r["gap_vs_baseline"] = ""
                r["gap_abs_reduction"] = ""
        ranking.sort(key=lambda r: (r["role"] != "candidate", -float(r["gap_abs_reduction"] or 0)))
        _write_csv(soc_dir / "ranking.csv", ranking)
        ranking_all.extend(ranking)

        best = _best_of(ranking, prior_only=False)
        best_prior = _best_of(ranking, prior_only=True)
        summaries.append(
            {
                "scale": scale,
                "soc_major_title": title,
                "slug": slug,
                "polarity": domain["polarity"],
                "gate": domain["gate"],
                "h1_effect": domain["effect"],
                "hypothesized_alpha_sign": domain["hypothesized_alpha_sign"],
                "n_train_families": len(split_meta["train_family_ids"]),
                "n_eval_families": len(items),
                "baseline_gender_gap": base["gender_gap"] if base else "",
                "best_id": best["config_id"] if best else "",
                "best_layer": best["layer"] if best else "",
                "best_alpha": best["alpha"] if best else "",
                "best_gender_gap": best["gender_gap"] if best else "",
                "best_gap_abs_reduction": best["gap_abs_reduction"] if best else "",
                "best_matches_prior": best.get("alpha_matches_prior") if best else "",
                "best_prior_id": best_prior["config_id"] if best_prior else "",
                "best_prior_alpha": best_prior["alpha"] if best_prior else "",
                "best_prior_gap_abs_reduction": (
                    best_prior["gap_abs_reduction"] if best_prior else ""
                ),
            }
        )

    ranking_all.sort(
        key=lambda r: (
            r.get("soc_major_title") or "",
            r["role"] != "candidate",
            -float(r.get("gap_abs_reduction") or 0),
        )
    )
    _write_csv(out_root / "ranking.csv", ranking_all)
    _write_csv(out_root / "summary.csv", summaries)
    _write_csv(out_root / "skip.csv", skip_rows)
    (out_root / "domains.json").write_text(
        json.dumps(
            {"scale": scale, "steer": steer, "skip": skip_rows, "catalog": str(args.catalog)},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    meta = {
        "schema": "steering.xy_control_per_soc_run/v1",
        "hypothesis": "xy_control",
        "name": "per-SOC gender-conditioned activation steering",
        "tag": tag,
        "datetime": datetime.now().isoformat(),
        "scale": scale,
        "model": model_id,
        "device": args.device,
        "dtype": args.dtype,
        "mapping": dict(MAPPING),
        "eval_split": eval_split,
        "smoke": bool(args.smoke),
        "n_domains_steered": len(summaries),
        "n_domains_skipped": len(skip_rows),
        "vectors_file": str(vectors_path),
        "vectors_sha256": vec_meta.get("arrays_sha256"),
        "runtime_s": round(time.time() - t_all, 1),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    (out_root / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for src in (vectors_path, vectors_path.with_suffix(".json")):
        if src.is_file():
            shutil.copy2(src, out_root / src.name)
    print(f"\n=== per-SOC [{tag}] {len(summaries)} steered, {len(skip_rows)} skipped  {meta['runtime_s']:.0f}s ===")
    print(f"  {out_root}")
    for s in summaries:
        base_g = s["baseline_gender_gap"]
        red = s["best_gap_abs_reduction"]
        base_s = f"{base_g:+.4f}" if isinstance(base_g, float) else str(base_g)
        red_s = f"{red:.4f}" if isinstance(red, float) else str(red)
        print(f"  {s['polarity']:<7} {s['soc_major_title']}: base {base_s}  best {s['best_id']}  Δ|G| {red_s}")
    return out_root


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", choices=["2b", "4b", "both"], default="both")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--catalog", type=Path, default=CATALOG_JSON)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--limit-items", type=int, default=None)
    ap.add_argument("--only-soc", default=None)
    ap.add_argument("--max-domains", type=int, default=None)
    ap.add_argument("--eval-split", choices=["heldout", "val", "test", "train", "all"], default="val")
    ap.add_argument("--train-frac", type=float, default=None)
    ap.add_argument("--val-frac", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--from-capture", type=Path, default=None)
    ap.add_argument("--from-vectors", action="store_true")
    ap.add_argument("--vectors", type=Path, default=None)
    ap.add_argument("--vectors-tag", default=None)
    ap.add_argument("--no-save-capture", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "results" / "steering" / "xy_control")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--log-every", type=int, default=40)
    args = ap.parse_args(argv)

    scales = ["2b", "4b"] if args.scale == "both" else [args.scale]
    if args.from_capture is not None and args.scale == "both":
        raise SystemExit("--from-capture requires a single --scale")
    if args.model is not None and args.scale == "both":
        raise SystemExit("--model requires a single --scale")

    _load_dotenv()
    for scale in scales:
        run_scale(args, scale)
    return 0


if __name__ == "__main__":
    sys.exit(main())
