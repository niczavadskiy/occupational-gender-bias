"""Fit one unit v_raw per FDR-significant SOC.

Capture last-token HS on train families of each steered SOC (full
without_abstain pool), then for each significant domain independently:

    v_raw[soc] = mean(h_gender − h_XY)  on that SOC's train families.

    python -m steering.xy_control.build_vectors_per_soc --scale 2b --device cuda
    python -m steering.xy_control.build_vectors_per_soc --scale 4b --device cuda
    python -m steering.xy_control.build_vectors_per_soc --from-capture path.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from steering.build_conditional_inlp_subspace import arrays_signature
from steering.contrastive.build_vectors import load_probe_bank, needed_layers
from steering.intervene import Scorer, load_model
from steering.run_hs_recovery_auc import pick_items
from steering.xy_control.build_vectors import CONFIG_BY_SCALE, capture_pair, sha256_file
from steering.xy_control.capture_io import load_pair_capture, save_pair_capture
from steering.xy_control.domains import CATALOG_JSON, load_catalog
from steering.xy_control.mapping import MAPPING
from steering.xy_control.paired_sample import filter_soc, load_pooled_items
from steering.xy_control.per_soc import (
    fit_domain_vectors,
    plan_domain_splits,
    skip_row,
    train_items_from_plan,
    vector_id_for,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
STEERING_DIR = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def _probe_paths(scale: str, override: list[Path] | None) -> list[Path]:
    if override is not None:
        return list(override)
    if scale == "4b":
        return [
            REPO_ROOT
            / "experiments"
            / "qwen35-4b-base"
            / "steering"
            / "vectors"
            / "h1_vectors_v1.npz"
        ]
    return [
        STEERING_DIR / "vectors" / "h1_vectors_v1.npz",
        STEERING_DIR / "vectors" / "slot_vectors_v1.npz",
    ]


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", choices=["2b", "4b"], default="2b")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--catalog", type=Path, default=CATALOG_JSON)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--n-items", type=int, default=None, help="Cap families per steered SOC (smoke).")
    ap.add_argument("--only-soc", default=None, help="Comma-separated slugs.")
    ap.add_argument("--max-domains", type=int, default=None)
    ap.add_argument("--train-frac", type=float, default=None)
    ap.add_argument("--val-frac", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--probe-vectors", type=Path, nargs="*", default=None)
    ap.add_argument("--from-capture", type=Path, default=None)
    ap.add_argument("--save-capture", type=Path, default=None)
    ap.add_argument("--no-save-capture", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=HERE / "vectors")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args(argv)

    cfg_path = args.config or CONFIG_BY_SCALE[args.scale]
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    scale = str(cfg.get("scale", args.scale))
    model_id = args.model or cfg["source"]["model_id"]
    split_cfg = cfg["source"]["split"]
    train_frac = float(args.train_frac if args.train_frac is not None else split_cfg["train_frac"])
    val_frac = float(args.val_frac if args.val_frac is not None else split_cfg["val_frac"])
    seed = int(args.seed if args.seed is not None else split_cfg["seed"])
    layers = needed_layers(cfg)
    catalog = load_catalog(args.catalog)
    block = catalog["scales"][scale]
    wanted = {s.strip() for s in args.only_soc.split(",")} if args.only_soc else None
    steer = list(block["steer"])
    if wanted:
        steer = [d for d in steer if d["slug"] in wanted]
    if args.max_domains is not None:
        steer = steer[: args.max_domains]

    pooled = load_pooled_items()
    soc_pool: list[dict] = []
    seen_fids: set[int] = set()
    for domain in steer:
        soc_items = pick_items(filter_soc(pooled, domain["soc_major_title"]), args.n_items)
        for it in soc_items:
            fid = int(it["scenario_family_id"])
            if fid in seen_fids:
                continue
            seen_fids.add(fid)
            soc_pool.append(it)
    soc_pool.sort(key=lambda it: int(it["scenario_family_id"]))
    planned, too_few = plan_domain_splits(
        soc_pool, steer, seed=seed, train_frac=train_frac, val_frac=val_frac
    )
    capture_items = train_items_from_plan(soc_pool, planned)

    _load_dotenv()

    if args.from_capture is not None:
        print(f"\n[1] load capture {args.from_capture}")
        H_g, H_xy, rows, _cap_meta = load_pair_capture(args.from_capture)
        missing = [L for L in layers if L not in H_g]
        if missing:
            raise SystemExit(f"capture missing layers {missing}")
    else:
        if not capture_items:
            raise SystemExit("no families to capture — check catalog / --only-soc")
        print(f"\n[1] load {model_id}...")
        t0 = time.time()
        model, tokenizer = load_model(model_id, dtype=args.dtype, device=args.device)
        for p in model.parameters():
            p.requires_grad_(False)
        scorer = Scorer(model, tokenizer)
        print(f"  ready in {time.time() - t0:.1f}s")
        print(
            f"\n[2] capture paired last-token HS on layers {layers} "
            f"({len(capture_items)} families, {sum(len(i['rows']) for i in capture_items)} pairs)..."
        )
        rows, hs_g_lists, hs_xy_lists = capture_pair(
            model, scorer, capture_items, layers, log_every=args.log_every
        )
        H_g = {L: np.stack(vs, axis=0) for L, vs in hs_g_lists.items()}
        H_xy = {L: np.stack(vs, axis=0) for L, vs in hs_xy_lists.items()}
        if not args.no_save_capture:
            cap_path = args.save_capture or (
                HERE / "captures" / f"xy_control_{scale}_per_soc_capture_{args.tag or 'v1'}.npz"
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
                    "pooled_splits": ["stagea", "stageb", "test", "stagec"],
                    "mapping": dict(MAPPING),
                    "n_items": len(capture_items),
                },
            )
            print(f"  capture → {cap_path}")

    probe_bank = load_probe_bank(_probe_paths(scale, args.probe_vectors))

    print(f"\n[3] fit per-SOC v_raw on {len(steer)} FDR-significant domains")
    arrays: dict[str, np.ndarray] = {}
    entries: list[dict] = []
    domain_meta: list[dict] = []
    skipped: list[dict] = []
    for d in block["skip"]:
        skipped.append(skip_row(scale=scale, title=d["soc_major_title"], reason="not_significant"))
    if wanted:
        for d in block["steer"]:
            if d["slug"] not in wanted:
                skipped.append(
                    skip_row(scale=scale, title=d["soc_major_title"], reason="not_selected")
                )
    for domain, n_fams in too_few:
        skipped.append(
            skip_row(
                scale=scale,
                title=domain["soc_major_title"],
                reason="too_few_families",
                extra={"n_families": n_fams, "polarity": domain["polarity"]},
            )
        )
        print(f"  SKIP {domain['soc_major_title']}: {n_fams} families")

    for domain in steer:
        title = domain["soc_major_title"]
        split_meta = planned.get(domain["slug"])
        if split_meta is None:
            continue
        train_fams = set(split_meta["train_family_ids"])
        val_fams = set(split_meta["val_family_ids"])
        test_fams = set(split_meta["test_family_ids"])
        fams_n = split_meta["n_families"]
        try:
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
        except ValueError as exc:
            skipped.append(
                skip_row(scale=scale, title=title, reason="fit_failed", extra={"error": str(exc)})
            )
            print(f"  SKIP {title}: {exc}")
            continue
        arrays.update(arr)
        entries.extend(ent)
        domain_meta.append(
            {
                **{k: domain[k] for k in (
                    "soc_major_title",
                    "slug",
                    "polarity",
                    "gate",
                    "effect",
                    "hypothesized_alpha_sign",
                )},
                "vector_id": vector_id_for(domain["slug"]),
                "n_families": fams_n,
                "train_family_ids": split_meta["train_family_ids"],
                "val_family_ids": split_meta["val_family_ids"],
                "test_family_ids": split_meta["test_family_ids"],
                "n_train_families": len(train_fams),
                "n_val_families": len(val_fams),
                "n_test_families": len(test_fams),
            }
        )
        anchor = int(cfg["layers"]["anchor"])
        key = f"{vector_id_for(domain['slug'])}__L{anchor}"
        extra = next((e for e in ent if e["key"] == key), ent[0] if ent else {})
        print(
            f"  {domain['polarity']:<7} {title}: train={len(train_fams)} "
            f"val={len(val_fams)} test={len(test_fams)}  "
            f"||v|| {extra.get('norm_raw', float('nan')):.3f}"
        )

    ver = args.tag or f"per_soc_{cfg['version']}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.out_dir / f"xy_control_{scale}_per_soc_vectors_{ver}.npz"
    json_path = args.out_dir / f"xy_control_{scale}_per_soc_vectors_{ver}.json"
    meta = {
        "schema": "steering.xy_control_per_soc_vectors/v1",
        "hypothesis": "xy_control",
        "name": "per-SOC gender-conditioned activation direction",
        "scale": scale,
        "map_version": ver,
        "map_config": cfg_path.name,
        "map_sha256": sha256_file(cfg_path),
        "catalog": str(args.catalog),
        "datetime": datetime.now().isoformat(),
        "model_id": model_id,
        "d_model": int(cfg["source"]["d_model"]),
        "hidden_state_index": cfg["source"]["hidden_state_index"],
        "layers": layers,
        "mapping": dict(MAPPING),
        "n_capture_rows": len(rows),
        "train_frac": train_frac,
        "val_frac": val_frac,
        "seed": seed,
        "calibration": {
            "c": "unused for add",
            "sigma_train": "1.0 so shift β ≡ add α if needed",
            "v": "mean_i (h_gender − h_XY) on that SOC's train families, then unit-norm",
        },
        "n_vectors": len(arrays),
        "n_domains_fit": len(domain_meta),
        "arrays_sha256": arrays_signature(arrays) if arrays else "",
        "domains": domain_meta,
        "skipped": skipped,
        "vectors": entries,
    }
    np.savez(npz_path, **arrays)
    json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n=== per-SOC vectors {scale}: {len(domain_meta)} domains, {len(arrays)} arrays ===")
    print(f"  -> {npz_path}")
    print(f"  -> {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
