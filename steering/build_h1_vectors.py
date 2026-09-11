"""
Материализация направлений и калибровок для H1 steering.

Для каждого слоя пояса и каждого направления считает:
  - ŵ            — unit-norm вектор в сыром HS-пространстве
  - c            — mid-point средних проекций классов, ТОЛЬКО train-строки пробы
  - sigma_train  — std проекций на train (единица масштаба для shift)

Направления gender / narrative / slot берутся готовыми из стадии
compare_directions (ключ w_raw формы (25, 2048)); w_gender_perp считается
ортогонализацией gender к span{narrative, slot} на том же слое.

Вход:
  - steering/configs/h1_steering_candidates_v1.yaml
  - results/<run>/hidden_states.npz + per_item.jsonl
  - results/<run>/probes/v1_rep_prob/<slug>/compare_directions/*_weights.npz

Выход:
  - steering/vectors/h1_vectors_v1.npz    — сами векторы
  - steering/vectors/h1_vectors_v1.json   — c, sigma, диагностика

Запуск из корня репозитория:
    python -m steering.build_h1_vectors
    python -m steering.build_h1_vectors --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from probes.core import load_split_for_run
from probes.v1_rep import load_v1_rep_bundle
from probes.v1_rep_orthogonal import _orthogonalize, _orthonormal_basis, _unit

REPO_ROOT = Path(__file__).resolve().parents[1]
STEERING_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = STEERING_DIR / "configs" / "h1_steering_candidates_v1.yaml"
DEFAULT_OUT_DIR = STEERING_DIR / "vectors"

# Ось → каким лейблом делятся классы при расчёте mid-point c.
LABEL_FOR_BASE = {"gender": "gender_choice", "narrative": "narrative_choice", "slot": "slot_choice"}


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


def load_axis_weights(run_dir: Path, cfg: dict) -> dict[str, np.ndarray]:
    """base → (n_layers, d_model) unit-norm directions из compare_directions."""
    src = cfg["source"]
    root = run_dir / src["probe_root"]
    out: dict[str, np.ndarray] = {}
    for base, slug in src["slugs"].items():
        target = f"{base}_choice"
        rel = src["weights_npz"].format(target=target)
        path = root / slug / rel
        if not path.exists():
            raise SystemExit(f"нет весов направлений: {path}")
        with np.load(path, allow_pickle=True) as z:
            w = np.asarray(z[src["weights_key"]], dtype=np.float64)
        out[base] = w
    return out


def class_calibration(s: np.ndarray, y: np.ndarray, train: np.ndarray) -> dict[str, float]:
    """c = mid-point средних проекций классов на train; sigma = std на train."""
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


def build(cfg: dict, run_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    layers = needed_layers(cfg)
    _rd, parent_meta, batch, x_layers = load_v1_rep_bundle(run_dir.name)
    split = load_split_for_run(
        run_dir, batch, seed=0, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, force=False
    )
    train = split.train_mask
    axis_w = load_axis_weights(run_dir, cfg)
    d_model = int(cfg["source"]["d_model"])

    labels = {
        "gender_choice": batch.gender_choice.astype(int),
        "narrative_choice": batch.narrative_choice.astype(int),
        "slot_choice": batch.slot_choice.astype(int),
    }

    arrays: dict[str, np.ndarray] = {}
    entries: list[dict] = []

    for vec in cfg["vectors"]:
        vid, kind = vec["id"], vec["kind"]
        seeds = vec.get("seeds", [None]) if kind == "random_unit" else [None]
        for seed in seeds:
            for layer in layers:
                x = x_layers[:, layer, :].astype(np.float64)

                if kind == "random_unit":
                    rng = stable_rng("h1_vectors", cfg["version"], vid, seed, layer)
                    w = _unit(rng.normal(size=d_model))
                    label_key = None
                elif kind == "probe_direction":
                    w = _unit(axis_w[vec["base"]][layer])
                    label_key = LABEL_FOR_BASE[vec["base"]]
                elif kind == "probe_direction_orthogonalized":
                    basis = _orthonormal_basis(
                        *[axis_w[b][layer] for b in vec["orthogonalize_against"]]
                    )
                    w = _orthogonalize(axis_w[vec["base"]][layer], basis)
                    label_key = LABEL_FOR_BASE[vec["base"]]
                else:
                    raise ValueError(f"unknown vector kind {kind!r}")

                if not np.isfinite(w).all():
                    raise ValueError(f"{vid} L{layer}: невалидный вектор")

                s = x @ w
                if label_key is None:
                    s_tr = s[train]
                    calib = {
                        "c": float(s_tr.mean()),
                        "mean_positive": None,
                        "mean_negative": None,
                        "separation": None,
                        "sigma_train": float(s_tr.std(ddof=1)),
                        "mean_train": float(s_tr.mean()),
                        "n_train_positive": None,
                        "n_train_negative": None,
                    }
                else:
                    calib = class_calibration(s, labels[label_key], train)

                key = f"{vid}__L{layer}" if seed is None else f"{vid}__L{layer}__s{seed}"
                arrays[key] = w.astype(np.float32)

                entry = {
                    "key": key,
                    "vector_id": vid,
                    "kind": kind,
                    "base": vec.get("base"),
                    "layer": layer,
                    "label_for_c": label_key,
                    **calib,
                }
                if seed is not None:
                    entry["random_seed"] = int(seed)
                if kind == "probe_direction_orthogonalized":
                    wg = _unit(axis_w[vec["base"]][layer])
                    entry["cos_with_base"] = float(w @ wg)
                    for other in vec["orthogonalize_against"]:
                        entry[f"cos_with_{other}"] = float(w @ _unit(axis_w[other][layer]))
                entries.append(entry)

    hyp = cfg.get("hypothesis", "h1")
    meta = {
        "schema": f"steering.{hyp}_vectors/v1",
        "hypothesis": hyp,
        "map_version": cfg["version"],
        "map_config": None,  # заполняется в main()
        "map_sha256": None,
        "run": cfg["source"]["run"],
        "model_id": parent_meta.get("model_id"),
        "d_model": d_model,
        "hidden_state_index": cfg["source"]["hidden_state_index"],
        "layers": layers,
        "calibration": {
            "c": "mid-point средних проекций классов на train-строках пробы",
            "sigma_train": "std проекций на train (единица масштаба для shift)",
            "split": {"seed": 0, "train_families": int(split.train_mask.sum() // 4)},
        },
        "n_vectors": len(arrays),
        "vectors": entries,
    }
    return arrays, meta


def arrays_signature(arrays: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for key in sorted(arrays):
        h.update(key.encode("utf-8"))
        h.update(np.ascontiguousarray(arrays[key], dtype=np.float32).tobytes())
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--out-prefix",
        default=None,
        help="Префикс файлов (default: {hypothesis}_vectors)",
    )
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_dir = args.results_root / cfg["source"]["run"]
    arrays, meta = build(cfg, run_dir)
    meta["map_config"] = args.config.name
    meta["map_sha256"] = sha256_file(args.config)
    meta["arrays_sha256"] = arrays_signature(arrays)

    ver = cfg["version"]
    hyp = cfg.get("hypothesis", "h1")
    prefix = args.out_prefix or f"{hyp}_vectors"
    npz_path = args.out_dir / f"{prefix}_{ver}.npz"
    json_path = args.out_dir / f"{prefix}_{ver}.json"
    text = json.dumps(meta, ensure_ascii=False, indent=2) + "\n"

    if args.verify:
        if not json_path.exists() or json_path.read_text(encoding="utf-8") != text:
            print(f"MISMATCH: {json_path.name}")
            return 1
        with np.load(npz_path) as z:
            on_disk = {k: np.asarray(z[k], dtype=np.float32) for k in z.files}
        if arrays_signature(on_disk) != meta["arrays_sha256"]:
            print(f"MISMATCH: {npz_path.name}")
            return 1
        print("OK — векторы совпадают с пересборкой")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **arrays)
    json_path.write_text(text, encoding="utf-8")

    print(f"{hyp} vectors {ver}: {len(arrays)} векторов, слои {meta['layers']}")
    for e in meta["vectors"]:
        if e["layer"] == cfg["layers"]["anchor"]:
            sep = e["separation"]
            sep_s = f"sep {sep:+.3f}" if sep is not None else "sep —"
            print(
                f"  {e['key']:<28} c {e['c']:+.3f}  sigma {e['sigma_train']:.3f}  {sep_s}"
            )
    print(f"  -> {npz_path.name}")
    print(f"  -> {json_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
