"""
Эксп.0: unit-test геометрии пробы в raw hidden-state space (CPU, без GPU).

Проба — Pipeline(StandardScaler → LogisticRegression). coef_ живёт в
standardized space. Steering использует inverse-scaler направление.
Этот скрипт проверяет тождество:

    q1 = pipeline.decision_function(H)
    q2 = H @ w_raw + b_raw

где w_raw,j = coef_j / σ_j,  b_raw = intercept − w_raw · μ.

Это не «center сажает probe на 0»: отдельно печатаются t_probe (порог
гиперплоскости после unit-norm) и class-midpoint c, которым пользуется
steering.

Запуск из корня репо (нужен hidden_states.npz исходного прогона):

    python -m steering.check_probe_geometry
    python -m steering.check_probe_geometry --layers 16,23 --targets gender_choice,slot_choice

Выход: results/steering/geometry_check/<tag>/geometry_check.json
Код 0 — все тождества в допуске; 1 — провал.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from probes.core import (
    direction_w_from_probe,
    fit_probe,
    load_split_for_run,
    raw_space_logit_params,
)
from probes.paths import DEFAULT_RUN_NAME
from probes.v1_rep import (
    V1_PROBE_ROOT,
    load_v1_rep_bundle,
    target_vector,
    v1_pipeline_run_slug,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STEERING_DIR = Path(__file__).resolve().parent

DEFAULT_RUN = "run_2026-06-09_12-50-33_Qwen3.5-2B-Base_v1_full_pos_shuffle"
VECTOR_FILE_FOR_TARGET = {
    "gender_choice": STEERING_DIR / "vectors" / "h1_vectors_v1.npz",
    "narrative_choice": STEERING_DIR / "vectors" / "h1_vectors_v1.npz",
    "slot_choice": STEERING_DIR / "vectors" / "slot_vectors_v1.npz",
}
VECTOR_KEY_FOR_TARGET = {
    "gender_choice": "w_gender__L{layer}",
    "narrative_choice": "w_narrative__L{layer}",
    "slot_choice": "w_slot__L{layer}",
}


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_targets(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def class_midpoint(s: np.ndarray, y: np.ndarray, train: np.ndarray) -> dict[str, float]:
    s_tr, y_tr = s[train], y[train].astype(int)
    pos, neg = s_tr[y_tr == 1], s_tr[y_tr == 0]
    m1, m0 = float(pos.mean()), float(neg.mean())
    return {
        "c": 0.5 * (m1 + m0),
        "mean_positive": m1,
        "mean_negative": m0,
        "separation": m1 - m0,
        "sigma_train": float(s_tr.std(ddof=1)),
    }


def err_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return {
        "max_abs": float(np.max(np.abs(d))),
        "mean_abs": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d * d))),
        "corr": float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 else float("nan"),
    }


def load_frozen_unit_w(target: str, layer: int) -> np.ndarray | None:
    path = VECTOR_FILE_FOR_TARGET.get(target)
    key_tmpl = VECTOR_KEY_FOR_TARGET.get(target)
    if path is None or key_tmpl is None or not path.is_file():
        return None
    key = key_tmpl.format(layer=layer)
    with np.load(path) as z:
        if key not in z.files:
            return None
        w = np.asarray(z[key], dtype=np.float64).ravel()
    n = float(np.linalg.norm(w))
    return w / n if n > 0 else w


def load_compare_directions_w(run_dir: Path, target: str, layer: int) -> np.ndarray | None:
    slug = v1_pipeline_run_slug(target)
    path = (
        run_dir
        / "probes"
        / V1_PROBE_ROOT
        / slug
        / "compare_directions"
        / f"compare_directions_{target}_weights.npz"
    )
    if not path.is_file():
        return None
    with np.load(path) as z:
        w = np.asarray(z["w_raw"], dtype=np.float64)
    if w.ndim != 2 or layer >= w.shape[0]:
        return None
    v = w[layer]
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float((a @ b) / (na * nb))


def check_one(
    *,
    X: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    target: str,
    layer: int,
    run_dir: Path,
    clf_C: float,
    max_iter: int,
    max_abs_err: float,
    min_cos_frozen: float,
) -> dict:
    model = fit_probe(
        X,
        y,
        train,
        target=target,
        ridge_alpha=1.0,
        clf_C=clf_C,
        max_iter=max_iter,
    )
    w_raw, b_raw = raw_space_logit_params(model, target=target)
    w_unit = direction_w_from_probe(model, target=target)
    q1 = np.asarray(model.decision_function(X), dtype=np.float64).ravel()
    q2 = X.astype(np.float64) @ w_raw + b_raw
    logit_err = err_stats(q1, q2)

    s_unit = X.astype(np.float64) @ w_unit
    calib = class_midpoint(s_unit, y, train)
    norm = float(np.linalg.norm(w_raw))
    t_probe = -b_raw / norm if norm > 0 else float("nan")

    frozen = load_frozen_unit_w(target, layer)
    compare_w = load_compare_directions_w(run_dir, target, layer)
    cos_frozen = cosine(w_unit, frozen) if frozen is not None else None
    cos_compare = cosine(w_unit, compare_w) if compare_w is not None else None

    logit_ok = logit_err["max_abs"] <= max_abs_err
    frozen_ok = True if cos_frozen is None else abs(cos_frozen) >= min_cos_frozen
    compare_ok = True if cos_compare is None else abs(cos_compare) >= min_cos_frozen
    passed = bool(logit_ok and frozen_ok and compare_ok)

    return {
        "target": target,
        "layer": layer,
        "n_rows": int(X.shape[0]),
        "n_train": int(train.sum()),
        "passed": passed,
        "logit_identity": {
            "ok": logit_ok,
            "threshold_max_abs": max_abs_err,
            **logit_err,
            "note": "q1=decision_function(H), q2=H@w_raw+b_raw",
        },
        "raw_hyperplane": {
            "||w_raw||": norm,
            "b_raw": float(b_raw),
            "t_probe": t_probe,
            "t_probe_note": "unit-projection threshold: w_unit·h + t_probe = 0  iff  q=0",
        },
        "class_midpoint": {
            **calib,
            "c_minus_t_probe": float(calib["c"] - t_probe),
            "note": "steering center uses c, not t_probe",
        },
        "frozen_steering_vector": {
            "file": str(VECTOR_FILE_FOR_TARGET.get(target, "")),
            "key": VECTOR_KEY_FOR_TARGET.get(target, "").format(layer=layer),
            "present": frozen is not None,
            "cosine_with_refit": cos_frozen,
            "ok": frozen_ok,
        },
        "compare_directions_w_raw": {
            "present": compare_w is not None,
            "cosine_with_refit": cos_compare,
            "ok": compare_ok,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=DEFAULT_RUN, help=f"results/<run>/ (default: v1 full; not {DEFAULT_RUN_NAME})")
    ap.add_argument("--layers", default="16,23")
    ap.add_argument("--targets", default="gender_choice,slot_choice")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--max-iter", type=int, default=10_000)
    ap.add_argument("--max-abs-err", type=float, default=1e-4, help="допуск |q1−q2|")
    ap.add_argument("--min-cos-frozen", type=float, default=0.999, help="допуск cos с замороженным ŵ")
    ap.add_argument("--tag", default="exp0")
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "results" / "steering")
    args = ap.parse_args(argv)

    layers = parse_int_list(args.layers)
    targets = parse_targets(args.targets)

    print(f"[1] Загрузка HS: {args.run}")
    run_dir, parent_meta, batch, x_layers = load_v1_rep_bundle(args.run)
    split = load_split_for_run(
        run_dir, batch, seed=0, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, force=False
    )
    train = split.train_mask
    print(f"    {x_layers.shape[0]} строк × {x_layers.shape[1]} слоёв × d={x_layers.shape[2]}  "
          f"train={int(train.sum())}")

    checks: list[dict] = []
    for target in targets:
        y = target_vector(batch, target)
        for layer in layers:
            print(f"\n[2] fit {target} L{layer} ...")
            X = np.asarray(x_layers[:, layer, :], dtype=np.float64)
            row = check_one(
                X=X,
                y=y,
                train=train,
                target=target,
                layer=layer,
                run_dir=run_dir,
                clf_C=args.C,
                max_iter=args.max_iter,
                max_abs_err=args.max_abs_err,
                min_cos_frozen=args.min_cos_frozen,
            )
            checks.append(row)
            le = row["logit_identity"]
            cm = row["class_midpoint"]
            rh = row["raw_hyperplane"]
            status = "OK" if row["passed"] else "FAIL"
            print(
                f"    [{status}] |q1−q2|_max={le['max_abs']:.3e}  "
                f"c={cm['c']:+.4f}  t_probe={rh['t_probe']:+.4f}  "
                f"c−t={cm['c_minus_t_probe']:+.4f}"
            )
            fz = row["frozen_steering_vector"]
            if fz["present"]:
                print(f"    frozen {fz['key']}: cos={fz['cosine_with_refit']:.6f}")
            else:
                print(f"    frozen {fz['key']}: нет файла/ключа — пропуск")
            cd = row["compare_directions_w_raw"]
            if cd["present"]:
                print(f"    compare_directions w_raw: cos={cd['cosine_with_refit']:.6f}")

    all_ok = all(c["passed"] for c in checks)
    out_dir = args.out_root / "geometry_check" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema": "steering.geometry_check/v1",
        "tag": args.tag,
        "datetime": datetime.now().isoformat(),
        "run": str(run_dir.name),
        "model_id": parent_meta.get("model_id"),
        "layers": layers,
        "targets": targets,
        "thresholds": {"max_abs_err": args.max_abs_err, "min_cos_frozen": args.min_cos_frozen},
        "passed": all_ok,
        "checks": checks,
    }
    out_path = out_dir / "geometry_check.json"
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n=== geometry check {'PASSED' if all_ok else 'FAILED'} → {out_path} ===")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
