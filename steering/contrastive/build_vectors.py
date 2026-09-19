"""
Live-capture last-token HS and fit contrastive mean-diff vectors.

v = mean(h | y=M) − mean(h | y=F)  on train families, then unit-norm.
c = mid-point of class mean projections on the same train rows.

y is the model's live gender choice (argmax constrained P(man|x) vs P(woman|x)),
not occupation gold and not frozen baseline_choice.

Output (H1 vector-bank schema, so run_h1_stagea / intervene work as-is):
  steering/contrastive/vectors/contrastive_gender_{scale}_vectors_v1.npz
  steering/contrastive/vectors/contrastive_gender_{scale}_vectors_v1.json

Examples:

    python -m steering.contrastive.build_vectors
    python -m steering.contrastive.build_vectors --scale 4b --model Qwen/Qwen3.5-4B-Base
    python -m steering.contrastive.build_vectors --n-items 3 --tag smoke
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

from steering.build_conditional_inlp_subspace import arrays_signature, fit_direction_mean_diff
from steering.contrastive.capture_io import save_capture
from steering.intervene import Scorer, capture_last_token, load_model
from steering.run_hs_recovery_auc import family_split, gender_choice_y, pick_items, roc_auc, row_margins
from steering.run_inlp_stagea import load_json

REPO_ROOT = Path(__file__).resolve().parents[2]
STEERING_DIR = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CONFIG_BY_SCALE = {
    "2b": HERE / "configs" / "contrastive_gender_2b_v1.yaml",
    "4b": HERE / "configs" / "contrastive_gender_4b_v1.yaml",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_rng(*parts) -> np.random.Generator:
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))


def needed_layers(cfg: dict) -> list[int]:
    named = cfg["layers"]
    out: set[int] = set()
    for fam in cfg["families"]:
        if not fam.get("enabled", True):
            continue
        spec = fam["layers"]
        vals = named[spec] if isinstance(spec, str) else spec
        out.update(int(v) for v in vals)
    return sorted(out)


def class_calibration(s: np.ndarray, y: np.ndarray, train: np.ndarray) -> dict[str, float]:
    s_tr, y_tr = s[train], y[train].astype(int)
    pos, neg = s_tr[y_tr == 1], s_tr[y_tr == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("train не содержит обоих классов")
    m1, m0 = float(pos.mean()), float(neg.mean())
    return {
        "c": 0.5 * (m1 + m0),
        "mean_positive": m1,
        "mean_negative": m0,
        "separation": m1 - m0,
        "sigma_train": float(s_tr.std(ddof=1)),
        "mean_train": float(s_tr.mean()),
        "n_train_positive": int((y_tr == 1).sum()),
        "n_train_negative": int((y_tr == 0).sum()),
    }


def resolve_sample(cfg: dict, override: Path | None) -> Path:
    if override is not None:
        return override
    rel = cfg["source"].get("sample", "samples/h1_stagea_sample_v1.json")
    p = Path(rel)
    if p.is_file():
        return p
    return STEERING_DIR / rel


def capture_live(
    model,
    scorer: Scorer,
    items: list[dict],
    layers: list[int],
    *,
    log_every: int,
) -> tuple[list[dict], dict[int, list[np.ndarray]]]:
    hs: dict[int, list[np.ndarray]] = {L: [] for L in layers}
    rows: list[dict] = []
    total = sum(len(i["rows"]) for i in items)
    t0 = time.time()
    done = 0
    for item in items:
        for src in item["rows"]:
            with capture_last_token(model, layers) as store:
                out = scorer.score(src["prompt"], list(src["valid_labels"]))
                for L in layers:
                    h = store[L][0, -1].float().detach().cpu().numpy().astype(np.float32)
                    hs[L].append(h)
            margins = row_margins(out, src["labels"])
            chosen = src["labels"].get(out["choice"])
            y_live = 1 if chosen == "man" else 0
            try:
                y_frozen = gender_choice_y(src)
            except SystemExit:
                y_frozen = None
            rows.append(
                {
                    "id": src["id"],
                    "scenario_family_id": item["scenario_family_id"],
                    "soc_major_title": item["soc_major_title"],
                    "context_order": src.get("context_order"),
                    "position_variant": src.get("position_variant"),
                    "y_live": y_live,
                    "y_frozen": y_frozen,
                    **margins,
                }
            )
            done += 1
            if log_every and done % log_every == 0:
                rate = done / max(time.time() - t0, 1e-6)
                eta = (total - done) / max(rate, 1e-9) / 60
                print(f"    {done}/{total}  {rate:.2f} row/s  ETA {eta:.1f} min", flush=True)
    return rows, hs


def maybe_cos_with_probe(
    w: np.ndarray,
    layer: int,
    probe_bank: dict[str, np.ndarray],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in ("w_gender", "w_gender_perp"):
        key = f"{name}__L{layer}"
        if key not in probe_bank:
            continue
        u = np.asarray(probe_bank[key], dtype=np.float64)
        nu = float(np.linalg.norm(u))
        if nu < 1e-12:
            continue
        out[f"cos_with_{name}"] = float(w @ (u / nu))
    return out


def load_probe_bank(paths: list[Path]) -> dict[str, np.ndarray]:
    bank: dict[str, np.ndarray] = {}
    for path in paths:
        if not path.is_file():
            continue
        with np.load(path) as z:
            for k in z.files:
                bank[k] = np.asarray(z[k], dtype=np.float32)
    return bank


def build_from_capture(
    cfg: dict,
    H_by_L: dict[int, np.ndarray],
    y: np.ndarray,
    families: np.ndarray,
    train_fams: set[int],
    test_fams: set[int],
    *,
    probe_bank: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], list[dict]]:
    layers = needed_layers(cfg)
    train = np.array([int(fid) in train_fams for fid in families])
    test = np.array([int(fid) in test_fams for fid in families])
    d_model = int(cfg["source"]["d_model"])
    arrays: dict[str, np.ndarray] = {}
    entries: list[dict] = []

    for vec in cfg["vectors"]:
        vid, kind = vec["id"], vec["kind"]
        seeds = vec.get("seeds", [None]) if kind == "random_unit" else [None]
        for seed in seeds:
            for layer in layers:
                if layer not in H_by_L:
                    continue
                x = H_by_L[layer].astype(np.float64)
                if x.shape[1] != d_model:
                    raise SystemExit(
                        f"d_model mismatch: config {d_model}, HS L{layer} {x.shape[1]}"
                    )
                extra: dict = {}
                if kind == "random_unit":
                    rng = stable_rng("contrastive", cfg["version"], vid, seed, layer)
                    w = rng.normal(size=d_model).astype(np.float64)
                    n = float(np.linalg.norm(w))
                    w = w / n
                    calib = class_calibration(x @ w, y, train)
                elif kind == "mean_diff":
                    w = fit_direction_mean_diff(x[train], y[train])
                    calib = class_calibration(x @ w, y, train)
                    extra = {
                        "auc_train": roc_auc(x[train] @ w, y[train]),
                        "auc_test": roc_auc(x[test] @ w, y[test]) if test.any() else float("nan"),
                        "norm_raw_mean_diff": float(
                            np.linalg.norm(
                                x[train][y[train] == 1].mean(0) - x[train][y[train] == 0].mean(0)
                            )
                        ),
                        **maybe_cos_with_probe(w, layer, probe_bank),
                    }
                else:
                    raise ValueError(f"unknown vector kind {kind!r}")

                key = f"{vid}__L{layer}" if seed is None else f"{vid}__L{layer}__s{seed}"
                arrays[key] = w.astype(np.float32)
                entry = {
                    "key": key,
                    "vector_id": vid,
                    "kind": kind,
                    "layer": layer,
                    "label_for_c": "live_gender_choice",
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
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--probe-vectors", type=Path, nargs="*", default=None)
    ap.add_argument("--out-dir", type=Path, default=HERE / "vectors")
    ap.add_argument("--tag", default=None, help="suffix instead of version (default: yaml version)")
    ap.add_argument(
        "--save-capture",
        type=Path,
        default=None,
        help="npz last-token HS (default: contrastive/captures/…_capture_{ver}.npz)",
    )
    ap.add_argument("--no-save-capture", action="store_true")
    ap.add_argument("--log-every", type=int, default=40)
    args = ap.parse_args(argv)

    cfg_path = args.config or CONFIG_BY_SCALE[args.scale]
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    scale = str(cfg.get("scale", args.scale))
    model_id = args.model or cfg["source"]["model_id"]
    train_frac = float(
        args.train_frac if args.train_frac is not None else cfg["source"]["split"]["train_frac"]
    )
    seed = int(args.seed if args.seed is not None else cfg["source"]["split"]["seed"])
    layers = needed_layers(cfg)
    sample_path = resolve_sample(cfg, args.sample)

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
    families = np.array([r["scenario_family_id"] for r in rows], dtype=np.int64)
    y = np.array([r["y_live"] for r in rows], dtype=np.int32)
    train_fams, test_fams = family_split(list(families), seed=seed, train_frac=train_frac)

    agree = [r for r in rows if r["y_frozen"] is not None]
    agree_rate = (
        float(np.mean([r["y_live"] == r["y_frozen"] for r in agree])) if agree else float("nan")
    )
    print(
        f"  rows={len(rows)}  train_fams={len(train_fams)}  test_fams={len(test_fams)}  "
        f"y pos={int(y.sum())}/{len(y)}  live≡frozen={agree_rate:.3f}"
    )

    if not args.no_save_capture:
        cap_path = args.save_capture or (
            HERE / "captures" / f"contrastive_gender_{scale}_capture_{args.tag or cfg['version']}.npz"
        )
        save_capture(
            cap_path,
            H_by_L=H_by_L,
            rows=rows,
            meta={
                "schema": "steering.contrastive_capture/v1",
                "scale": scale,
                "model_id": model_id,
                "sample": str(sample_path),
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

    print("\n[3] fit mean-diff + calibrate c on train families")
    arrays, entries = build_from_capture(
        cfg, H_by_L, y, families, train_fams, test_fams, probe_bank=probe_bank
    )

    ver = args.tag or cfg["version"]
    prefix = f"contrastive_gender_{scale}_vectors"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.out_dir / f"{prefix}_{ver}.npz"
    json_path = args.out_dir / f"{prefix}_{ver}.json"
    meta = {
        "schema": "steering.contrastive_vectors/v1",
        "hypothesis": "contrastive",
        "scale": scale,
        "map_version": ver,
        "map_config": cfg_path.name,
        "map_sha256": sha256_file(cfg_path),
        "datetime": datetime.now().isoformat(),
        "model_id": model_id,
        "d_model": int(cfg["source"]["d_model"]),
        "hidden_state_index": cfg["source"]["hidden_state_index"],
        "layers": layers,
        "label": "live_gender_choice",
        "sample": str(sample_path),
        "n_rows": len(rows),
        "n_items": len(items),
        "train_frac": train_frac,
        "seed": seed,
        "n_train_families": len(train_fams),
        "n_test_families": len(test_fams),
        "y_balance": {"n_man": int(y.sum()), "n_woman": int(len(y) - y.sum())},
        "live_vs_frozen_agreement": agree_rate,
        "calibration": {
            "c": "mid-point средних проекций классов на train-семьях",
            "sigma_train": "std проекций на train (единица масштаба для shift)",
            "v": "mean(h|y=1) − mean(h|y=0), затем unit-norm",
        },
        "n_vectors": len(arrays),
        "arrays_sha256": arrays_signature(arrays),
        "vectors": entries,
    }
    np.savez(npz_path, **arrays)
    json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    anchor = int(cfg["layers"]["anchor"])
    print(f"\n=== contrastive vectors {scale} {ver}: {len(arrays)} ===")
    for e in entries:
        if e["layer"] != anchor:
            continue
        extra = ""
        if e.get("auc_test") is not None:
            extra = f"  auc_tr {e['auc_train']:.3f} auc_te {e['auc_test']:.3f}"
        if "cos_with_w_gender" in e:
            extra += f"  cos(w_gender) {e['cos_with_w_gender']:+.3f}"
        print(f"  {e['key']:<28} c {e['c']:+.3f}  sep {e['separation']:+.3f}{extra}")
    print(f"  -> {npz_path}")
    print(f"  -> {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
