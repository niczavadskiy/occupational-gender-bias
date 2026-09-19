"""
Diagnostic: PCA of stratified live gender contrasts + V_k for Stage A.

D rows are per-SOC mean-diffs on train families (not per-row HS). Uncentered
PC1 should match v_G; unit-row centered PCA + y-shuffle null tests whether
preference is one direction.

Writes:
  steering/contrastive/rank/contrastive_gender_{scale}_rank_{tag}.json
  steering/contrastive/rank/contrastive_gender_{scale}_rank_{tag}.md
  steering/subspaces/contrastive_pca_{scale}_{tag}.npz  (INLP Stage A schema)

Examples:

    python -m steering.contrastive.analyze_rank --scale 2b --device cuda
    python -m steering.contrastive.analyze_rank --from-capture steering/contrastive/captures/....npz
    python -m steering.contrastive.analyze_rank --n-items 8 --n-null 8 --tag smoke
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

from steering.build_conditional_inlp_subspace import (
    arrays_signature,
    fit_direction_mean_diff,
    random_orthonormal,
    subspace_centers,
)
from steering.contrastive.build_vectors import (
    CONFIG_BY_SCALE,
    capture_live,
    load_probe_bank,
    maybe_cos_with_probe,
    needed_layers,
    resolve_sample,
    sha256_file,
)
from steering.contrastive.capture_io import load_capture, save_capture
from steering.contrastive.rank import layer_markdown, layer_report, report_to_jsonable
from steering.intervene import Scorer, load_model
from steering.run_hs_recovery_auc import family_split, pick_items, roc_auc
from steering.run_inlp_stagea import load_json

HERE = Path(__file__).resolve().parent
STEERING_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]


def capture_path_for(scale: str, ver: str) -> Path:
    return HERE / "captures" / f"contrastive_gender_{scale}_capture_{ver}.npz"


def default_probe_paths(scale: str) -> list[Path]:
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


def row_field(rows: list[dict], key: str, default: str = "") -> np.ndarray:
    return np.array([r.get(key, default) for r in rows])


def write_subspace_bank(
    path: Path,
    *,
    layers_fit: dict[int, dict],
    y: np.ndarray,
    train: np.ndarray,
    H_by_L: dict[int, np.ndarray],
    meta_extra: dict,
    seed: int,
) -> dict:
    arrays: dict[str, np.ndarray] = {}
    layers_detail: list[dict] = []
    rng = np.random.default_rng(seed)
    for L in sorted(layers_fit):
        fit = layers_fit[L]
        if "W" not in fit:
            continue
        W = np.asarray(fit["W"], dtype=np.float32)
        H_tr = np.asarray(H_by_L[L][train], dtype=np.float64)
        y_tr = y[train]
        c = subspace_centers(H_tr, y_tr, W.astype(np.float64)).astype(np.float32)
        k = int(W.shape[1])
        arrays[f"L{L}__W"] = W
        arrays[f"L{L}__centers"] = c
        Wr = random_orthonormal(W.shape[0], k, rng).astype(np.float32)
        cr = subspace_centers(H_tr, y_tr, Wr.astype(np.float64)).astype(np.float32)
        arrays[f"L{L}__W_random_s0"] = Wr
        arrays[f"L{L}__centers_random_s0"] = cr
        ranks = list(range(1, k + 1))
        layers_detail.append(
            {
                "layer": L,
                "k_found": k,
                "k_chance": None,
                "auc_curve": [float("nan")] * (k + 1),
                "steering_ranks": ranks,
                "random_control_ranks": ranks,
                "n_above_null": fit.get("n_above_null"),
                "verdict": (fit.get("interpret") or {}).get("verdict"),
                "cos_uncent_pc1_vg": fit.get("cos_uncent_pc1_vg"),
                "note": "auc_curve placeholder: PCA has no decoder AUC; use n_above_null",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema": "steering.inlp_subspace/v1",
        "hypothesis": "contrastive_pca",
        "target": "gender_choice",
        "datetime": datetime.now().isoformat(),
        "layers": [d["layer"] for d in layers_detail],
        "layers_detail": layers_detail,
        "arrays_sha256": arrays_signature(arrays),
        "centers_note": "c_j = mid-point of class projections on train families",
        "read": (
            "V_k = [v_G, unit-centered SOC PCs after Gram–Schmidt]. "
            "Use with run_inlp_stagea (center). k=1 is contrastive v_G."
        ),
        **meta_extra,
    }
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", choices=["2b", "4b"], default="2b")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--sample", type=Path, default=None)
    ap.add_argument("--n-items", type=int, default=None)
    ap.add_argument("--from-capture", type=Path, default=None)
    ap.add_argument("--recapture", action="store_true", help="игнорировать существующий capture, снять HS заново")
    ap.add_argument("--save-capture", type=Path, default=None)
    ap.add_argument("--no-save-capture", action="store_true")
    ap.add_argument("--probe-vectors", type=Path, nargs="*", default=None)
    ap.add_argument("--layers", default=None, help="comma layers; default = yaml belt")
    ap.add_argument("--n-null", type=int, default=48)
    ap.add_argument("--min-per-class", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--train-frac", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=HERE / "rank")
    ap.add_argument("--subspace-dir", type=Path, default=STEERING_DIR / "subspaces")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--log-every", type=int, default=40)
    args = ap.parse_args(argv)

    cfg_path = args.config or CONFIG_BY_SCALE[args.scale]
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    scale = str(cfg.get("scale", args.scale))
    ver = args.tag or cfg["version"]
    model_id = args.model or cfg["source"]["model_id"]
    train_frac = float(
        args.train_frac if args.train_frac is not None else cfg["source"]["split"]["train_frac"]
    )
    seed = int(args.seed if args.seed is not None else cfg["source"]["split"]["seed"])
    sample_path = resolve_sample(cfg, args.sample)

    if args.layers:
        layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    else:
        layers = needed_layers(cfg)

    cap_path = args.from_capture
    if cap_path is None and not args.recapture:
        for cand in (capture_path_for(scale, ver), capture_path_for(scale, cfg["version"])):
            if cand.is_file():
                cap_path = cand
                print(f"  using capture {cap_path}")
                break

    rows: list[dict]
    H_by_L: dict[int, np.ndarray]
    if cap_path is not None:
        H_all, rows, cap_meta = load_capture(cap_path)
        missing = [L for L in layers if L not in H_all]
        if missing:
            raise SystemExit(f"capture без слоёв {missing}; есть {sorted(H_all)}")
        H_by_L = {L: H_all[L] for L in layers}
        model_id = str(cap_meta.get("model_id") or model_id)
        print(f"[1] loaded capture {cap_path.name}: {len(rows)} rows, layers {layers}")
    else:
        try:
            from dotenv import load_dotenv

            load_dotenv(REPO_ROOT / ".env")
        except ImportError:
            pass
        if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

        sample = load_json(sample_path)
        items = pick_items(sample["items"], args.n_items)
        print(f"\n[1] load {model_id}...")
        t0 = time.time()
        model, tokenizer = load_model(model_id, dtype=args.dtype, device=args.device)
        for p in model.parameters():
            p.requires_grad_(False)
        scorer = Scorer(model, tokenizer)
        print(f"  ready in {time.time() - t0:.1f}s")
        print(f"\n[2] capture last-token HS on layers {layers} ({len(items)} items)...")
        rows, hs_lists = capture_live(model, scorer, items, layers, log_every=args.log_every)
        H_by_L = {L: np.stack(vs, axis=0) for L, vs in hs_lists.items()}
        if not args.no_save_capture:
            out_cap = args.save_capture or capture_path_for(scale, ver)
            save_capture(
                out_cap,
                H_by_L=H_by_L,
                rows=rows,
                meta={
                    "schema": "steering.contrastive_capture/v1",
                    "scale": scale,
                    "model_id": model_id,
                    "sample": str(sample_path),
                    "map_version": ver,
                },
            )
            print(f"  saved capture {out_cap}")

    if any("soc_major_title" not in r for r in rows):
        raise SystemExit("capture rows без soc_major_title — переснимите HS")
    if any("context_order" not in r for r in rows):
        print("  WARN: нет context_order в rows — layout-таблица будет пустой")

    y = np.array([r["y_live"] for r in rows], dtype=np.int32)
    families = np.array([r["scenario_family_id"] for r in rows], dtype=np.int64)
    soc = row_field(rows, "soc_major_title")
    context_order = row_field(rows, "context_order", "")
    position_variant = row_field(rows, "position_variant", "")
    train_fams, test_fams = family_split(list(families), seed=seed, train_frac=train_frac)
    train = np.array([int(fid) in train_fams for fid in families])
    min_pc = int(args.min_per_class)
    n_soc_train = len({s for s, t in zip(soc, train) if t})
    print(
        f"  rows={len(rows)}  train_fams={len(train_fams)}  test_fams={len(test_fams)}  "
        f"y pos={int(y.sum())}/{len(y)}  train SOC={n_soc_train}  min_per_class={min_pc}"
    )

    probe_paths = args.probe_vectors if args.probe_vectors is not None else default_probe_paths(scale)
    probe_bank = load_probe_bank(list(probe_paths))

    layers_fit: dict[int, dict] = {}
    print(f"\n[3] stratified PCA  n_null={args.n_null}  k_max={args.k_max}")
    for L in layers:
        x = H_by_L[L].astype(np.float64)
        v_g = fit_direction_mean_diff(x[train], y[train])
        probes: dict[str, np.ndarray] = {}
        for name in ("w_gender", "w_gender_perp", "w_slot", "w_slot_perp"):
            key = f"{name}__L{L}"
            if key in probe_bank:
                probes[name] = np.asarray(probe_bank[key], dtype=np.float64)
        fit = layer_report(
            x,
            y,
            train,
            soc=soc,
            context_order=context_order,
            position_variant=position_variant,
            v_g=v_g,
            n_null=args.n_null,
            seed=seed + 1000 * L,
            min_per_class=min_pc,
            k_max=args.k_max,
            probes=probes,
        )
        fit["v_G"] = v_g
        fit["auc_train"] = float(roc_auc(x[train] @ v_g, y[train]))
        test = ~train
        fit["auc_test"] = (
            float(roc_auc(x[test] @ v_g, y[test])) if test.any() else float("nan")
        )
        fit["cos_with_probe"] = maybe_cos_with_probe(v_g, L, probe_bank)
        layers_fit[L] = fit
        if "error" in fit:
            print(f"  L{L}: {fit['error']}")
            continue
        interp = fit["interpret"]
        print(
            f"  L{L}: strata={fit['n_soc_strata']}  PC1={fit['uncentered']['explained'][0]:.3f}  "
            f"cos(v_G)={fit['cos_uncent_pc1_vg']:+.3f}  above_null={fit['n_above_null']}  "
            f"verdict={interp['verdict']}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"contrastive_gender_{scale}_rank"
    json_path = args.out_dir / f"{prefix}_{ver}.json"
    md_path = args.out_dir / f"{prefix}_{ver}.md"
    sub_path = args.subspace_dir / f"contrastive_pca_{scale}_{ver}.npz"

    payload = {
        "schema": "steering.contrastive_rank/v1",
        "hypothesis": "contrastive_pca",
        "scale": scale,
        "map_version": ver,
        "map_config": cfg_path.name,
        "map_sha256": sha256_file(cfg_path),
        "datetime": datetime.now().isoformat(),
        "model_id": model_id,
        "sample": str(sample_path),
        "n_rows": len(rows),
        "train_frac": train_frac,
        "seed": seed,
        "n_null": args.n_null,
        "min_per_class": min_pc,
        "k_max": args.k_max,
        "n_train_families": len(train_fams),
        "n_test_families": len(test_fams),
        "construction": {
            "D": "per-SOC mean(h|y=M)−mean(h|y=F) on train families",
            "not": "per-row HS or random man−woman pairs",
            "primary_spectrum": "unit-row centered PCA vs within-stratum y-shuffle",
            "steering": "center h' = h − α V (Vᵀh − c); k=1 is v_G",
        },
        "layers": {str(L): report_to_jsonable(fit) for L, fit in layers_fit.items()},
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md_parts = [
        f"# Contrastive rank PCA — {scale} `{ver}`",
        "",
        f"Модель `{model_id}`. D = per-SOC mean-diff на train-семьях; "
        f"null = {args.n_null} within-stratum shuffle y.",
        "",
        "k=1 в `contrastive_pca_*.npz` — тот же \(v_G\), что contrastive mean-diff. "
        "Каузальный тест — Stage A по рангу, не спектр сам по себе.",
        "",
        "| layer | strata | PC1 share | cos(PC1, v_G) | above null | verdict |",
        "| ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for L in layers:
        fit = layers_fit[L]
        if "error" in fit:
            md_parts.append(f"| {L} | — | — | — | — | error |")
            continue
        un = fit["uncentered"]
        md_parts.append(
            f"| {L} | {fit['n_soc_strata']} | {un['explained'][0]:.3f} | "
            f"{fit['cos_uncent_pc1_vg']:+.3f} | {fit['n_above_null']} | "
            f"{fit['interpret']['verdict']} |"
        )
    md_parts.append("")
    for L in layers:
        md_parts.append(layer_markdown(L, layers_fit[L]))
    md_path.write_text("\n".join(md_parts) + "\n", encoding="utf-8")

    sub_meta = write_subspace_bank(
        sub_path,
        layers_fit=layers_fit,
        y=y,
        train=train,
        H_by_L=H_by_L,
        seed=seed,
        meta_extra={
            "map_version": ver,
            "scale": scale,
            "model_id": model_id,
            "sample": str(sample_path),
            "rank_report": str(json_path),
            "k_max": args.k_max,
        },
    )
    print(f"\n=== rank PCA {scale} {ver}: {len(sub_meta.get('layers_detail') or [])} layers ===")
    print(f"  -> {json_path}")
    print(f"  -> {md_path}")
    print(f"  -> {sub_path}")
    print(
        f"  Stage A: python -m steering.contrastive.run_pca_stagea --scale {scale} "
        f"--tag pca_{scale}_a_{ver} --device cuda"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
