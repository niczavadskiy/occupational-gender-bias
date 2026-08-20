"""
Phase A: INLP — минимальная размерность линейно декодируемого preference-subspace.

Для каждого слоя-кандидата итеративно:
    probe = Pipeline(StandardScaler → LogisticRegression).fit(H_train, y_train)
    AUC(j) на held-out val
    w_raw = coef_scaled / sigma            (обратный scaler)
    w_j   = unit(w_raw − W W^T w_raw)      (ортогонализация против найденных)
    H    ←  H − (H w_j) w_j^T              (train и val одновременно)

Останов: AUC(k) <= 0.55 И 95% cluster-bootstrap CI накрывает 0.5 (тогда
k_chance = k), либо k_max итераций (тогда k_chance = null — полное линейное
стирание не заявляется).

Centers для rank-k center считаются на СЫРЫХ train-проекциях:
    c_j = ½(mean[h·w_j | y=0] + mean[h·w_j | y=1])

Дополнительно кладётся случайное ортонормированное подпространство того же
максимального ранга (negative control §31): первые k столбцов — валидный
random rank-k базис.

Вход:
  - steering/configs/inlp_gender_v1.yaml
  - results/<run>/hidden_states.npz
  - results/<run>/probes/_shared/group_split_v1_scenario.json

Выход:
  - steering/subspaces/inlp_<target>_<ver>.npz   — W, centers, диагностика
  - steering/subspaces/inlp_<target>_<ver>.json  — AUC(k), k_chance, схема §28
  - steering/subspaces/inlp_<target>_<ver>.md    — таблица AUC(k) по слоям

Запуск из корня репозитория (CPU, GPU не нужен):
    python -m steering.build_inlp_subspace
    python -m steering.build_inlp_subspace --layers 16 --k-max 4 --tag smoke
    python -m steering.build_inlp_subspace --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from probes.core import load_split_for_run, make_classifier, raw_space_logit_params
from probes.v1_rep import load_v1_rep_bundle, target_vector

REPO_ROOT = Path(__file__).resolve().parents[1]
STEERING_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = STEERING_DIR / "configs" / "inlp_gender_v1.yaml"
DEFAULT_OUT_DIR = STEERING_DIR / "subspaces"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_rng(*parts) -> np.random.Generator:
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))


def arrays_signature(arrays: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for key in sorted(arrays):
        h.update(key.encode("utf-8"))
        h.update(np.ascontiguousarray(arrays[key], dtype=np.float32).tobytes())
    return h.hexdigest()


def cluster_bootstrap_auc(
    scores: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    ci: float = 0.95,
) -> dict[str, float]:
    """95% CI для AUC с ресемплом по семьям (позиционные варианты не разлучаются)."""
    uniq = np.unique(groups)
    per_group = [np.flatnonzero(groups == g) for g in uniq]
    sizes = {len(idx) for idx in per_group}
    fast = len(sizes) == 1
    idx_matrix = np.stack(per_group) if fast else None

    rng = np.random.default_rng(seed)
    boot: list[float] = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = idx_matrix[pick].ravel() if fast else np.concatenate([per_group[p] for p in pick])
        y_b = y[idx]
        if len(np.unique(y_b)) < 2:
            continue
        boot.append(float(roc_auc_score(y_b, scores[idx])))

    if not boot:
        return {"ci_lo": float("nan"), "ci_hi": float("nan"), "boot_mean": float("nan"), "n_boot": 0}
    arr = np.array(boot)
    lo, hi = np.quantile(arr, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "boot_mean": float(arr.mean()),
        "n_boot": len(boot),
    }


def is_chance(auc: float, ci_lo: float, ci_hi: float, *, threshold: float, level: float) -> bool:
    if not np.isfinite([auc, ci_lo, ci_hi]).all():
        return False
    return bool(auc <= threshold and ci_lo <= level <= ci_hi)


def residualize(H: np.ndarray, w: np.ndarray) -> np.ndarray:
    return H - np.outer(H @ w, w)


def orthogonality_diag(W: np.ndarray) -> dict[str, float]:
    """max |WᵀW − I| по диагонали и вне неё; считается в float32 (как в хуке)."""
    if W.shape[1] == 0:
        return {"max_diag_error": 0.0, "max_offdiag_abs": 0.0, "max_abs_error": 0.0}
    W32 = np.asarray(W, dtype=np.float32)
    G = W32.T @ W32
    off_diagonal = G - np.diag(np.diag(G))
    return {
        "max_diag_error": float(np.max(np.abs(np.diag(G) - 1.0))),
        "max_offdiag_abs": float(np.max(np.abs(off_diagonal))) if G.shape[0] > 1 else 0.0,
        "max_abs_error": float(np.max(np.abs(G - np.eye(G.shape[0], dtype=np.float32)))),
    }


def subspace_centers(
    H_raw: np.ndarray, y: np.ndarray, W: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """c, mu_0, mu_1, sigma по столбцам W на сырых train-проекциях."""
    s = H_raw @ W
    y = y.astype(int)
    mu0 = s[y == 0].mean(axis=0)
    mu1 = s[y == 1].mean(axis=0)
    return 0.5 * (mu0 + mu1), mu0, mu1, s.std(axis=0, ddof=1)


def random_orthonormal(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Столбцы ортонормированы; первые k' < k столбцов — валидный random базис."""
    Q, _ = np.linalg.qr(rng.normal(size=(d, k)))
    return np.asarray(Q[:, :k], dtype=np.float64)


def inlp_layer(
    X_layer: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    groups: np.ndarray,
    *,
    target: str,
    k_max: int,
    clf_C: float,
    max_iter: int,
    chance_cfg: dict,
    verbose: bool = True,
) -> dict:
    """Итерации INLP на одном слое. Возвращает W, centers, auc_curve, схему §28."""
    H_tr_raw = np.asarray(X_layer[train], dtype=np.float64)
    H_va_raw = np.asarray(X_layer[val], dtype=np.float64)
    y_tr = y[train].astype(int)
    y_va = y[val].astype(int)
    g_va = groups[val]

    H_tr, H_va = H_tr_raw.copy(), H_va_raw.copy()
    directions: list[np.ndarray] = []
    iterations: list[dict] = []
    diag_arrays: dict[str, list[np.ndarray]] = {
        "coef_scaled": [],
        "scaler_mean": [],
        "scaler_scale": [],
        "w_raw": [],
    }
    k_chance: int | None = None

    for j in range(k_max + 1):
        t0 = time.time()
        probe = make_classifier(clf_C, max_iter)
        probe.fit(H_tr, y_tr)
        s_va = np.asarray(probe.decision_function(H_va), dtype=np.float64).ravel()
        s_tr = np.asarray(probe.decision_function(H_tr), dtype=np.float64).ravel()
        auc_va = float(roc_auc_score(y_va, s_va))
        auc_tr = float(roc_auc_score(y_tr, s_tr))
        boot = cluster_bootstrap_auc(
            s_va,
            y_va,
            g_va,
            n_boot=int(chance_cfg["n_bootstrap"]),
            seed=int(chance_cfg.get("bootstrap_seed", 20260820)) + j,
        )
        reached = is_chance(
            auc_va,
            boot["ci_lo"],
            boot["ci_hi"],
            threshold=float(chance_cfg["auc_threshold"]),
            level=float(chance_cfg["require_ci_contains"]),
        )

        entry: dict = {
            "iteration": j,
            "rank_removed": j,
            "train_n": int(train.sum()),
            "validation_n": int(val.sum()),
            "train_auc": auc_tr,
            "validation_auc": auc_va,
            "validation_auc_ci": [boot["ci_lo"], boot["ci_hi"]],
            "validation_auc_boot_mean": boot["boot_mean"],
            "n_bootstrap_effective": boot["n_boot"],
            "is_chance": reached,
            "residual_norm_train_mean": float(np.linalg.norm(H_tr, axis=1).mean()),
            "residual_norm_val_mean": float(np.linalg.norm(H_va, axis=1).mean()),
            "fit_seconds": round(time.time() - t0, 2),
        }

        if reached:
            k_chance = j
            entry["direction_index"] = None
            iterations.append(entry)
            if verbose:
                print(
                    f"    k={j:>2}  AUC {auc_va:.4f}  CI [{boot['ci_lo']:.3f},{boot['ci_hi']:.3f}]"
                    f"  → chance, стоп"
                )
            break

        if j == k_max:
            entry["direction_index"] = None
            entry["note"] = "k_max достигнут, направление не извлекалось"
            iterations.append(entry)
            if verbose:
                print(
                    f"    k={j:>2}  AUC {auc_va:.4f}  CI [{boot['ci_lo']:.3f},{boot['ci_hi']:.3f}]"
                    f"  → k_max, chance не достигнут"
                )
            break

        w_raw, _ = raw_space_logit_params(probe, target=target)
        w = w_raw.copy()
        if directions:
            W_prev = np.column_stack(directions)
            w = w - W_prev @ (W_prev.T @ w)
            resid_share = float(np.linalg.norm(w) / max(np.linalg.norm(w_raw), 1e-30))
            ortho_err = float(np.max(np.abs(W_prev.T @ (w / np.linalg.norm(w)))))
        else:
            resid_share = 1.0
            ortho_err = 0.0
        nrm = float(np.linalg.norm(w))
        if nrm < 1e-10:
            entry["direction_index"] = None
            entry["note"] = "w_raw целиком лежит в уже удалённом подпространстве"
            iterations.append(entry)
            if verbose:
                print(f"    k={j:>2}  AUC {auc_va:.4f}  → выродившееся направление, стоп")
            break
        w = w / nrm

        scaler = probe.named_steps["scaler"]
        coef = np.asarray(probe.named_steps["clf"].coef_, dtype=np.float64).ravel()
        diag_arrays["coef_scaled"].append(coef)
        diag_arrays["scaler_mean"].append(np.asarray(scaler.mean_, dtype=np.float64))
        diag_arrays["scaler_scale"].append(np.asarray(scaler.scale_, dtype=np.float64))
        diag_arrays["w_raw"].append(w_raw)

        entry.update(
            {
                "direction_index": j + 1,
                "w_raw_norm": float(np.linalg.norm(w_raw)),
                "cos_w_raw_orthogonalized": float(
                    w_raw @ w / max(np.linalg.norm(w_raw), 1e-30)
                ),
                "residual_share_after_orthogonalization": resid_share,
                "orthogonality_error": ortho_err,
            }
        )
        iterations.append(entry)
        if verbose:
            print(
                f"    k={j:>2}  AUC {auc_va:.4f}  CI [{boot['ci_lo']:.3f},{boot['ci_hi']:.3f}]"
                f"  train {auc_tr:.4f}  cos(w_raw,w_⊥) {entry['cos_w_raw_orthogonalized']:+.3f}"
                f"  {entry['fit_seconds']:.1f}s"
            )

        directions.append(w)
        H_tr = residualize(H_tr, w)
        H_va = residualize(H_va, w)

    W = np.column_stack(directions) if directions else np.zeros((X_layer.shape[1], 0))
    centers, mu0, mu1, sigma = subspace_centers(H_tr_raw, y_tr, W)

    # Class means вписываются в схему §28 по итерациям.
    for e in iterations:
        idx = e.get("direction_index")
        if idx is None:
            continue
        e["class_mean_0"] = float(mu0[idx - 1])
        e["class_mean_1"] = float(mu1[idx - 1])
        e["center"] = float(centers[idx - 1])
        e["sigma_train"] = float(sigma[idx - 1])

    return {
        "W": W,
        "centers": centers,
        "class_mean_0": mu0,
        "class_mean_1": mu1,
        "sigma_train": sigma,
        "auc_curve": np.array([e["validation_auc"] for e in iterations], dtype=np.float64),
        "k_chance": k_chance,
        "k_found": int(W.shape[1]),
        "iterations": iterations,
        "orthogonality": orthogonality_diag(W),
        "diag_arrays": {k: (np.stack(v) if v else np.zeros((0, X_layer.shape[1]))) for k, v in diag_arrays.items()},
    }


def steering_ranks(k_chance: int | None, k_found: int, base_ranks: list[int]) -> list[int]:
    """§14: base + k_chance; если chance не достигнут — все степени двойки до k_found."""
    ranks = set(int(r) for r in base_ranks)
    if k_chance is not None and k_chance > 0:
        ranks.add(int(k_chance))
    else:
        power = 1
        while power <= k_found:
            ranks.add(power)
            power *= 2
        ranks.add(k_found)
    return sorted(r for r in ranks if 1 <= r <= k_found)


def build(
    cfg: dict,
    run_dir: Path,
    *,
    layers: list[int],
    k_max: int,
    clf_C: float | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    target = cfg["target"]
    split_cfg = cfg["split"]
    probe_cfg = dict(cfg["probe"])
    if clf_C is not None:
        probe_cfg["C"] = clf_C
    inlp_cfg = cfg["inlp"]

    print(f"[1] Загрузка HS: {run_dir.name}")
    _rd, parent_meta, batch, x_layers = load_v1_rep_bundle(run_dir.name)
    split = load_split_for_run(
        run_dir,
        batch,
        seed=int(split_cfg["seed"]),
        train_ratio=float(split_cfg["ratios"]["train"]),
        val_ratio=float(split_cfg["ratios"]["val"]),
        test_ratio=float(split_cfg["ratios"]["test"]),
        force=False,
    )
    y = target_vector(batch, target)
    train, val = split.train_mask, split.val_mask
    groups = batch.scenario_family_id
    d_model = int(x_layers.shape[2])
    print(
        f"    {x_layers.shape[0]} строк × {x_layers.shape[1]} слоёв × d={d_model}  "
        f"train={int(train.sum())} ({len(np.unique(groups[train]))} семей)  "
        f"val={int(val.sum())} ({len(np.unique(groups[val]))} семей)"
    )
    print(f"    target={target}  баланс train: {int(y[train].sum())} / {int((1 - y[train]).sum())}")

    arrays: dict[str, np.ndarray] = {}
    layer_meta: list[dict] = []
    rand_cfg = cfg["steering"]["random_control"]

    for layer in layers:
        print(f"\n[2] INLP L{layer} (k_max={k_max})")
        res = inlp_layer(
            x_layers[:, layer, :],
            y,
            train,
            val,
            groups,
            target=target,
            k_max=k_max,
            clf_C=float(probe_cfg["C"]),
            max_iter=int(probe_cfg["max_iter"]),
            chance_cfg=inlp_cfg["chance"],
        )
        k_found = res["k_found"]
        ranks = steering_ranks(res["k_chance"], k_found, cfg["steering"]["ranks"])
        prefix = f"L{layer}__"
        arrays[prefix + "W"] = res["W"].astype(np.float32)
        arrays[prefix + "centers"] = res["centers"].astype(np.float32)
        arrays[prefix + "auc_curve"] = res["auc_curve"].astype(np.float32)
        for name, arr in res["diag_arrays"].items():
            arrays[f"{prefix}iter_{name}"] = arr.astype(np.float32)

        k_rand = max(k_found, 1)
        random_ranks = steering_ranks(res["k_chance"], k_rand, rand_cfg["ranks"])
        H_train_raw = np.asarray(x_layers[:, layer, :][train], dtype=np.float64)
        for seed in rand_cfg["seeds"]:
            rng = stable_rng("inlp_random", cfg["version"], target, layer, seed)
            Q = random_orthonormal(d_model, k_rand, rng)
            arrays[f"{prefix}W_random_s{seed}"] = Q.astype(np.float32)
            arrays[f"{prefix}centers_random_s{seed}"] = (H_train_raw @ Q).mean(axis=0).astype(np.float32)

        ortho = res["orthogonality"]
        tol = float(inlp_cfg["orthogonality_tolerance"])
        layer_meta.append(
            {
                "layer": layer,
                "model": parent_meta.get("model_id"),
                "target": target,
                "k_max": k_max,
                "k_found": k_found,
                "k_chance": res["k_chance"],
                "auc_curve": [float(a) for a in res["auc_curve"]],
                "auc_0": float(res["auc_curve"][0]),
                "auc_last": float(res["auc_curve"][-1]),
                "steering_ranks": ranks,
                "random_control_ranks": random_ranks,
                "centers": [float(c) for c in res["centers"]],
                "sigma_train": [float(s) for s in res["sigma_train"]],
                "orthogonality": ortho,
                "orthogonality_ok": bool(ortho["max_abs_error"] <= tol),
                "iterations": res["iterations"],
            }
        )
        status = "OK" if layer_meta[-1]["orthogonality_ok"] else "FAIL"
        print(
            f"    итог: k_found={k_found}  k_chance={res['k_chance']}  "
            f"AUC {res['auc_curve'][0]:.4f} → {res['auc_curve'][-1]:.4f}  "
            f"ranks={ranks}  |WᵀW−I|={ortho['max_abs_error']:.2e} [{status}]"
        )

    meta = {
        "schema": "steering.inlp_subspace/v1",
        "hypothesis": cfg.get("hypothesis", "inlp"),
        "map_version": cfg["version"],
        "map_config": None,
        "map_sha256": None,
        "run": run_dir.name,
        "model_id": parent_meta.get("model_id"),
        "target": target,
        "d_model": d_model,
        "hidden_state_index": cfg["source"]["hidden_state_index"],
        "layers": layers,
        "k_max": k_max,
        "probe": dict(probe_cfg),
        "chance_criterion": dict(inlp_cfg["chance"]),
        "split": {
            "seed": int(split_cfg["seed"]),
            "train_families": int(len(np.unique(groups[train]))),
            "val_families": int(len(np.unique(groups[val]))),
            "inlp_train": split_cfg["inlp_train"],
            "inlp_validation": split_cfg["inlp_validation"],
        },
        "centers_note": "c_j = ½(mean[h·w_j|y=0] + mean[h·w_j|y=1]) на СЫРЫХ train-проекциях",
        "random_control_note": (
            "W_random_s* — ортонормированный базис ранга k_found; первые k столбцов "
            "образуют валидное random rank-k подпространство. centers_random = "
            "средняя train-проекция (классовой структуры нет)."
        ),
        "layers_detail": layer_meta,
    }
    return arrays, meta


def summary_markdown(meta: dict) -> str:
    lines = [
        f"# INLP subspace — `{meta['target']}` / `{meta['map_version']}`",
        "",
        f"- Run: `{meta['run']}`, модель `{meta['model_id']}`, d={meta['d_model']}",
        f"- INLP train = probe train ({meta['split']['train_families']} семей), "
        f"held-out AUC = val ({meta['split']['val_families']} семей)",
        f"- Критерий chance: AUC ≤ {meta['chance_criterion']['auc_threshold']} "
        f"и 95% cluster-bootstrap CI накрывает {meta['chance_criterion']['require_ci_contains']}",
        "",
        "## AUC(k) — held-out val",
        "",
    ]
    # Опорные ранги вместо всех k: при k_max=64 полная таблица нечитаема,
    # вся кривая лежит в auc_curve соседнего JSON.
    longest = max((len(L["auc_curve"]) for L in meta["layers_detail"]), default=1)
    marks = [k for k in (0, 1, 2, 4, 8, 16, 32, 64, 96, 128) if k < longest]
    if longest - 1 not in marks:
        marks.append(longest - 1)
    header = "| Layer | " + " | ".join(f"k={k}" for k in marks) + " | k_chance |"
    lines += [header, "| :--- | " + " | ".join("---:" for _ in marks) + " | ---: |"]
    for L in meta["layers_detail"]:
        curve = L["auc_curve"]
        cells = [f"{curve[k]:.3f}" if k < len(curve) else "" for k in marks]
        kc = L["k_chance"] if L["k_chance"] is not None else "не достигнут"
        lines.append(f"| {L['layer']} | " + " | ".join(cells) + f" | {kc} |")

    lines += ["", "## Ранги для Phase B", "", "| Layer | k_found | steering ranks | random control | max \\|WᵀW−I\\| |", "| :--- | ---: | :--- | :--- | ---: |"]
    for L in meta["layers_detail"]:
        lines.append(
            f"| {L['layer']} | {L['k_found']} | {L['steering_ranks']} | "
            f"{L['random_control_ranks']} | {L['orthogonality']['max_abs_error']:.1e} |"
        )
    lines += [
        "",
        "Если `k_chance` не достигнут, rank-k intervention НЕ представляет полного "
        "линейного стирания — это фиксируется в выводах.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--layers", default=None, help="Переопределить слои, через запятую")
    ap.add_argument("--k-max", type=int, default=None)
    ap.add_argument(
        "--clf-C",
        type=float,
        default=None,
        help="Переопределить регуляризацию пробы (диагностика: при C=1 train AUC=1.0, n≈d)",
    )
    ap.add_argument("--tag", default=None, help="Суффикс файлов вместо version (для smoke)")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_dir = args.results_root / cfg["source"]["run"]
    if not (run_dir / "hidden_states.npz").exists():
        raise SystemExit(f"нет hidden_states.npz в {run_dir}")

    layers = (
        [int(x) for x in args.layers.split(",") if x.strip()]
        if args.layers
        else [int(x) for x in cfg["layers"]["candidates"]]
    )
    k_max = args.k_max if args.k_max is not None else int(cfg["inlp"]["k_max"])

    t0 = time.time()
    arrays, meta = build(cfg, run_dir, layers=layers, k_max=k_max, clf_C=args.clf_C)
    meta["map_config"] = args.config.name
    meta["map_sha256"] = sha256_file(args.config)
    meta["arrays_sha256"] = arrays_signature(arrays)
    meta["build_seconds"] = round(time.time() - t0, 1)

    ver = args.tag or cfg["version"]
    stem = f"inlp_{cfg['target']}_{ver}"
    npz_path = args.out_dir / f"{stem}.npz"
    json_path = args.out_dir / f"{stem}.json"
    md_path = args.out_dir / f"{stem}.md"
    md = summary_markdown(meta)

    if args.verify:
        text_on_disk = json_path.read_text(encoding="utf-8") if json_path.exists() else ""
        on_disk_meta = json.loads(text_on_disk) if text_on_disk else {}
        if on_disk_meta.get("arrays_sha256") != meta["arrays_sha256"]:
            print(f"MISMATCH: {npz_path.name} (arrays_sha256)")
            return 1
        print("OK — подпространства совпадают с пересборкой")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **arrays)
    json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(md, encoding="utf-8")

    print(f"\n=== INLP {cfg['target']} {ver} готово за {meta['build_seconds']:.0f}s ===")
    for L in meta["layers_detail"]:
        curve = " ".join(f"{a:.3f}" for a in L["auc_curve"])
        print(f"  L{L['layer']}: AUC(k) = {curve}")
        print(f"          k_chance={L['k_chance']}  ranks={L['steering_ranks']}")
    print(f"  -> {npz_path.name}")
    print(f"  -> {json_path.name}")
    print(f"  -> {md_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
