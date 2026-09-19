"""
Paired last-token HS capture and gender-conditioned raw vector.

v_raw = mean_i (h_gender[i] − h_XY[i])  on train families, then unit-norm.

    python -m steering.xy_control.build_vectors
    python -m steering.xy_control.build_vectors --scale 4b --device cuda
    python -m steering.xy_control.build_vectors --n-items 3 --tag smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from steering.build_conditional_inlp_subspace import arrays_signature
from steering.contrastive.build_vectors import load_probe_bank, maybe_cos_with_probe, needed_layers, stable_rng
from steering.xy_control.capture_io import save_pair_capture
from steering.intervene import Scorer, capture_last_token, load_model
from steering.run_hs_recovery_auc import pick_items
from steering.run_inlp_stagea import load_json, row_margins
from steering.xy_control.algebra import family_split3, fit_v_raw
from steering.xy_control.mapping import MAPPING

REPO_ROOT = Path(__file__).resolve().parents[2]
STEERING_DIR = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CONFIG_BY_SCALE = {
    "2b": HERE / "configs" / "xy_control_2b_v1.yaml",
    "4b": HERE / "configs" / "xy_control_4b_v1.yaml",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_sample(cfg: dict, override: Path | None, key: str) -> Path:
    if override is not None:
        return override
    rel = cfg["source"].get(key, "data/xy_pairs_stagea_v1.json")
    p = Path(rel)
    if p.is_file():
        return p
    return HERE / rel


    return train, val, test


def gender_prompt_of(row: dict) -> str:
    return row.get("gender_prompt") or row["prompt"]


def xy_prompt_of(row: dict) -> str:
    if not row.get("xy_prompt"):
        raise ValueError(f"{row.get('id')}: missing xy_prompt — rebuild the paired dataset")
    return row["xy_prompt"]


def capture_pair(
    model,
    scorer: Scorer,
    items: list[dict],
    layers: list[int],
    *,
    log_every: int,
) -> tuple[list[dict], dict[int, list[np.ndarray]], dict[int, list[np.ndarray]]]:
    hs_g: dict[int, list[np.ndarray]] = {L: [] for L in layers}
    hs_xy: dict[int, list[np.ndarray]] = {L: [] for L in layers}
    rows: list[dict] = []
    total = sum(len(i["rows"]) for i in items)
    t0 = time.time()
    done = 0
    for item in items:
        for src in item["rows"]:
            g_prompt = gender_prompt_of(src)
            xy_prompt = xy_prompt_of(src)
            captured: dict[str, dict] = {}
            for tag, prompt in (("gender", g_prompt), ("xy", xy_prompt)):
                with capture_last_token(model, layers) as store:
                    out = scorer.score(prompt, list(src["valid_labels"]))
                    captured[tag] = {"out": out, "hs": {L: store[L][0, -1].float().detach().cpu().numpy().astype(np.float32) for L in layers}}
            for L in layers:
                hs_g[L].append(captured["gender"]["hs"][L])
                hs_xy[L].append(captured["xy"]["hs"][L])
            labels = src["labels"]
            margins = row_margins(captured["gender"]["out"], labels)
            g_out = captured["gender"]["out"]
            xy_out = captured["xy"]["out"]
            rows.append(
                {
                    "id": src["id"],
                    "scenario_family_id": item["scenario_family_id"],
                    "soc_major_title": item["soc_major_title"],
                    "context_order": src.get("context_order"),
                    "position_variant": src.get("position_variant"),
                    "labels": labels,
                    "xy_labels": src.get("xy_labels"),
                    "mapping": src.get("mapping", dict(MAPPING)),
                    "n_prompt_tokens_gender": g_out["n_prompt_tokens"],
                    "n_prompt_tokens_xy": xy_out["n_prompt_tokens"],
                    "choice_gender": g_out["choice"],
                    **margins,
                }
            )
            done += 1
            if log_every and done % log_every == 0:
                rate = done / max(time.time() - t0, 1e-6)
                eta = (total - done) / max(rate, 1e-9) / 60
                print(f"    {done}/{total} pairs  {rate:.2f} pair/s  ETA {eta:.1f} min", flush=True)
    return rows, hs_g, hs_xy


def build_from_capture(
    cfg: dict,
    H_g: dict[int, np.ndarray],
    H_xy: dict[int, np.ndarray],
    families: np.ndarray,
    train_fams: set[int],
    val_fams: set[int],
    test_fams: set[int],
    *,
    probe_bank: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], list[dict]]:
    layers = needed_layers(cfg)
    train = np.array([int(fid) in train_fams for fid in families])
    if not train.any():
        raise SystemExit("train split is empty")
    d_model = int(cfg["source"]["d_model"])
    arrays: dict[str, np.ndarray] = {}
    entries: list[dict] = []

    for vec in cfg["vectors"]:
        vid, kind = vec["id"], vec["kind"]
        seeds = vec.get("seeds", [None]) if kind == "random_unit" else [None]
        for seed in seeds:
            for layer in layers:
                if layer not in H_g:
                    continue
                g = H_g[layer].astype(np.float64)
                x = H_xy[layer].astype(np.float64)
                if g.shape != x.shape:
                    raise SystemExit(f"L{layer}: gender {g.shape} vs XY {x.shape}")
                if g.shape[1] != d_model:
                    raise SystemExit(f"d_model mismatch: config {d_model}, HS L{layer} {g.shape[1]}")
                extra: dict = {}
                if kind == "random_unit":
                    rng = stable_rng("xy_control", cfg["version"], vid, seed, layer)
                    w = rng.normal(size=d_model).astype(np.float64)
                    w = w / float(np.linalg.norm(w))
                    calib = {
                        "c": 0.0,
                        "sigma_train": 1.0,
                        "norm_raw": 1.0,
                    }
                elif kind == "paired_mean_diff":
                    D = g - x
                    w, norm_raw = fit_v_raw(D[train])
                    D_tr = D[train]
                    extra = {
                        "norm_raw": norm_raw,
                        "mean_abs_d": float(np.mean(np.linalg.norm(D_tr, axis=1))),
                        "n_train_pairs": int(train.sum()),
                        "n_val_pairs": int(sum(int(fid) in val_fams for fid in families)),
                        "n_test_pairs": int(sum(int(fid) in test_fams for fid in families)),
                        **maybe_cos_with_probe(w, layer, probe_bank),
                    }
                    calib = {"c": 0.0, "sigma_train": 1.0, "norm_raw": norm_raw}
                else:
                    raise ValueError(f"unknown vector kind {kind!r}")

                key = f"{vid}__L{layer}" if seed is None else f"{vid}__L{layer}__s{seed}"
                arrays[key] = w.astype(np.float32)
                entry = {
                    "key": key,
                    "vector_id": vid,
                    "kind": kind,
                    "layer": layer,
                    "label_for_c": "paired_xy_control",
                    **calib,
                    **extra,
                }
                if seed is not None:
                    entry["random_seed"] = int(seed)
                entries.append(entry)
    return arrays, entries


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", choices=["2b", "4b"], default="2b")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--sample", type=Path, default=None)
    ap.add_argument("--n-items", type=int, default=None)
    ap.add_argument("--train-frac", type=float, default=None)
    ap.add_argument("--val-frac", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--probe-vectors", type=Path, nargs="*", default=None)
    ap.add_argument("--out-dir", type=Path, default=HERE / "vectors")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--save-capture", type=Path, default=None)
    ap.add_argument("--no-save-capture", action="store_true")
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
    sample_path = resolve_sample(cfg, args.sample, "fit_sample")

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

    print(f"\n[2] capture paired last-token HS on layers {layers} ({len(items)} items)...")
    rows, hs_g_lists, hs_xy_lists = capture_pair(model, scorer, items, layers, log_every=args.log_every)
    H_g = {L: np.stack(vs, axis=0) for L, vs in hs_g_lists.items()}
    H_xy = {L: np.stack(vs, axis=0) for L, vs in hs_xy_lists.items()}
    families = np.array([r["scenario_family_id"] for r in rows], dtype=np.int64)
    train_fams, val_fams, test_fams = family_split3(
        list(families), seed=seed, train_frac=train_frac, val_frac=val_frac
    )
    print(
        f"  pairs={len(rows)}  train_fams={len(train_fams)}  val_fams={len(val_fams)}  "
        f"test_fams={len(test_fams)}"
    )

    if not args.no_save_capture:
        cap_path = args.save_capture or (
            HERE / "captures" / f"xy_control_{scale}_capture_{args.tag or cfg['version']}.npz"
        )
        save_pair_capture(
            cap_path,
            H_gender=H_g,
            H_xy=H_xy,
            rows=rows,
            meta={
                "schema": "steering.xy_control_capture/v1",
                "scale": scale,
                "model_id": model_id,
                "sample": str(sample_path),
                "mapping": dict(MAPPING),
                "map_version": args.tag or cfg["version"],
            },
        )
        print(f"  capture → {cap_path}")

    probe_paths = args.probe_vectors
    if probe_paths is None:
        probe_paths = [
            STEERING_DIR / "vectors" / "h1_vectors_v1.npz",
            STEERING_DIR / "vectors" / "slot_vectors_v1.npz",
        ]
        exp4 = (
            REPO_ROOT
            / "experiments"
            / "qwen35-4b-base"
            / "steering"
            / "vectors"
            / "h1_vectors_v1.npz"
        )
        if scale == "4b":
            probe_paths = [exp4]

    probe_bank = load_probe_bank(list(probe_paths))

    print("\n[3] fit v_raw = mean(h_gender − h_XY) on train families")
    arrays, entries = build_from_capture(
        cfg, H_g, H_xy, families, train_fams, val_fams, test_fams, probe_bank=probe_bank
    )

    ver = args.tag or cfg["version"]
    prefix = f"xy_control_{scale}_vectors"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.out_dir / f"{prefix}_{ver}.npz"
    json_path = args.out_dir / f"{prefix}_{ver}.json"
    meta = {
        "schema": "steering.xy_control_vectors/v1",
        "hypothesis": "xy_control",
        "name": "gender-conditioned activation direction",
        "scale": scale,
        "map_version": ver,
        "map_config": cfg_path.name,
        "map_sha256": sha256_file(cfg_path),
        "datetime": datetime.now().isoformat(),
        "model_id": model_id,
        "d_model": int(cfg["source"]["d_model"]),
        "hidden_state_index": cfg["source"]["hidden_state_index"],
        "layers": layers,
        "mapping": dict(MAPPING),
        "sample": str(sample_path),
        "n_rows": len(rows),
        "n_items": len(items),
        "train_frac": train_frac,
        "val_frac": val_frac,
        "seed": seed,
        "train_family_ids": sorted(train_fams),
        "val_family_ids": sorted(val_fams),
        "test_family_ids": sorted(test_fams),
        "n_train_families": len(train_fams),
        "n_val_families": len(val_fams),
        "n_test_families": len(test_fams),
        "calibration": {
            "c": "unused for add (kept 0 for H1 vector-bank schema)",
            "sigma_train": "1.0 so shift β ≡ add α if needed",
            "v": "mean_i (h_gender − h_XY) on train, then unit-norm",
        },
        "n_vectors": len(arrays),
        "arrays_sha256": arrays_signature(arrays),
        "vectors": entries,
    }
    np.savez(npz_path, **arrays)
    json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    anchor = int(cfg["layers"]["anchor"])
    print(f"\n=== xy_control vectors {scale} {ver}: {len(arrays)} ===")
    for e in entries:
        if e["layer"] != anchor or e["kind"] != "paired_mean_diff":
            continue
        extra = f"  ||v_raw|| {e['norm_raw']:.4f}  mean||d_i|| {e['mean_abs_d']:.4f}"
        if "cos_with_w_gender" in e:
            extra += f"  cos(w_gender) {e['cos_with_w_gender']:+.3f}"
        if "cos_with_w_gender_perp" in e:
            extra += f"  cos(w_gender_perp) {e['cos_with_w_gender_perp']:+.3f}"
        print(f"  {e['key']:<24}{extra}")
    print(f"  -> {npz_path}")
    print(f"  -> {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
