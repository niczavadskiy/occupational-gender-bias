"""
Phase B: rank-k center steering и behavioral effect vs rank.

Для каждого (layer, k) применяется h' = h − α W_k(Wᵀh − c) в last-token
residual и сравнивается с baseline на той же замороженной val-выборке.

Метрика. Скоринг — z_A vs z_B, пол задаётся лейаутом строки, поэтому
построчно |d_gender| ≡ |d_slot|: построчное падение |d| неотличимо от падения
решительности A/B. Гендерная ось выделяется усреднением по 4 лейаутам семьи
ДО взятия модуля (p0/p1 меняют слот, mf/wf — порядок упоминания):

    D_i = mean_layouts (z_man − z_woman)      гендерная ось
    S_i = mean_layouts (z_A − z_B)            позиционная ось (guardrail)
    R_i = |D_i^before| − |D_i^after|          primary, R>0 = preference ослабла

95% CI — cluster bootstrap по scenario_family_id (семья и есть кластер).

Обязательный hook-check: при α=1 должно быть Wᵀh' = c; сохраняются
max/mean projection error и относительное возмущение ||Δh||/||h||.

Вход:
  - steering/subspaces/inlp_<target>_<ver>.npz + .json   (Phase A)
  - steering/samples/h1_stagea_sample_v1.json

Выход:
  results/steering/inlp_stage_a/<tag>/<config_id>/per_row.jsonl
  results/steering/inlp_stage_a/<tag>/<config_id>/per_family.jsonl
  results/steering/inlp_stage_a/<tag>/<config_id>/metrics.json
  results/steering/inlp_stage_a/<tag>/ranking.csv
  results/steering/inlp_stage_a/<tag>/auc_vs_behavior.csv     ← главная таблица §37
  results/steering/inlp_stage_a/<tag>/run_meta.json

Примеры:
    # smoke на CPU: baseline + rank-1/2 на L16, 3 семьи
    python -m steering.run_inlp_stagea --model steering/.cache/model --device cpu \\
        --dtype bfloat16 --layers 16 --ranks 1,2 --limit-items 3 --tag smoke

    # полный прогон
    python -m steering.run_inlp_stagea --model Qwen/Qwen3.5-2B-Base --device cuda \\
        --dtype float32 --tag inlp_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from steering.intervene import (
    Scorer,
    SubspaceSpec,
    SubspaceTrace,
    load_model,
    num_hidden_layers,
    steered,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STEERING_DIR = Path(__file__).resolve().parent
BASELINE_ID = "baseline"
DEFAULT_SUBSPACES = STEERING_DIR / "subspaces" / "inlp_gender_choice_v1.npz"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_int_list(s: str | None) -> list[int] | None:
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str | None) -> list[float] | None:
    if not s:
        return None
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def slot_for(labels: dict[str, str], semantic: str) -> str:
    for slot, sem in labels.items():
        if sem == semantic:
            return slot
    raise KeyError(f"{semantic} нет в labels {labels}")


def row_margins(out: dict, labels: dict[str, str]) -> dict[str, float]:
    """d_gender / d_slot / p_man из constrained-скоринга одной строки."""
    man, woman = slot_for(labels, "man"), slot_for(labels, "woman")
    z_man = float(out[f"logit_{man}"])
    z_woman = float(out[f"logit_{woman}"])
    p_man = float(out.get(f"prob_constrained_{man}", float("nan")))
    p_woman = float(out.get(f"prob_constrained_{woman}", float("nan")))
    denom = p_man + p_woman
    return {
        "d_gender": z_man - z_woman,
        "d_slot": float(out["logit_A"]) - float(out["logit_B"]),
        "p_man": p_man,
        "p_man_norm": p_man / denom if denom > 0 else float("nan"),
        "prefers_man": labels.get(out["choice"]) == "man",
        "prefers_woman": labels.get(out["choice"]) == "woman",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Конфигурации
# ─────────────────────────────────────────────────────────────────────────────
def build_configs(
    meta: dict,
    available: set[str],
    *,
    layers: list[int] | None,
    ranks: list[int] | None,
    alphas: list[float],
    random_seeds: list[int],
    include_random: bool,
) -> list[dict]:
    by_layer = {int(L["layer"]): L for L in meta["layers_detail"]}
    want_layers = layers if layers is not None else sorted(by_layer)
    out: list[dict] = []
    for layer in want_layers:
        if layer not in by_layer:
            raise SystemExit(f"в подпространствах нет L{layer}; есть {sorted(by_layer)}")
        detail = by_layer[layer]
        layer_ranks = ranks if ranks is not None else detail["steering_ranks"]
        for k in layer_ranks:
            if k > detail["k_found"]:
                print(f"  пропуск L{layer} k={k}: найдено только {detail['k_found']} направлений")
                continue
            for alpha in alphas:
                out.append(
                    {
                        "id": f"inlp__L{layer}__k{k}__a{alpha:g}".replace(".", "p"),
                        "role": "candidate",
                        "basis": "inlp",
                        "layer": layer,
                        "rank": k,
                        "alpha": alpha,
                        "W_key": f"L{layer}__W",
                        "c_key": f"L{layer}__centers",
                        "auc_after_removal": (
                            detail["auc_curve"][k] if k < len(detail["auc_curve"]) else None
                        ),
                        "k_chance": detail["k_chance"],
                    }
                )
        if not include_random:
            continue
        for k in detail["random_control_ranks"]:
            for seed in random_seeds:
                if f"L{layer}__W_random_s{seed}" not in available:
                    continue
                for alpha in alphas:
                    out.append(
                        {
                            "id": f"rand{seed}__L{layer}__k{k}__a{alpha:g}".replace(".", "p"),
                            "role": "control",
                            "basis": f"random_s{seed}",
                            "layer": layer,
                            "rank": k,
                            "alpha": alpha,
                            "W_key": f"L{layer}__W_random_s{seed}",
                            "c_key": f"L{layer}__centers_random_s{seed}",
                            "auc_after_removal": None,
                            "k_chance": detail["k_chance"],
                        }
                    )
    return out


def spec_for(cfg: dict, arrays: dict[str, np.ndarray]) -> SubspaceSpec:
    W = arrays[cfg["W_key"]]
    c = arrays[cfg["c_key"]]
    k = int(cfg["rank"])
    if W.shape[1] < k:
        raise SystemExit(f"{cfg['id']}: rank {k} > доступных {W.shape[1]} в {cfg['W_key']}")
    return SubspaceSpec(
        layer=int(cfg["layer"]),
        W=np.ascontiguousarray(W[:, :k], dtype=np.float32),
        c=np.ascontiguousarray(c[:k], dtype=np.float32),
        kind="center",
        alpha=float(cfg["alpha"]),
        label=cfg["id"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Прогон
# ─────────────────────────────────────────────────────────────────────────────
def run_config(
    scorer: Scorer,
    model,
    items: list[dict],
    spec: SubspaceSpec | None,
    *,
    label: str,
    log_every: int,
) -> list[dict]:
    trace = SubspaceTrace() if spec is not None else None
    rows: list[dict] = []
    total = sum(len(i["rows"]) for i in items)
    t0 = time.time()

    for item in items:
        for src in item["rows"]:
            if trace is not None:
                trace.clear()
            with steered(model, [spec] if spec else [], trace):
                out = scorer.score(src["prompt"], list(src["valid_labels"]))

            labels = src["labels"]
            row = {
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
                "baseline_choice_recorded": src["baseline_choice"],
            }
            if spec is not None and trace is not None:
                s_b = trace.before.get(spec.layer)
                s_a = trace.after.get(spec.layer)
                h_norm = trace.h_norm.get(spec.layer)
                d_norm = trace.delta_norm.get(spec.layer)
                if s_b is not None and s_a is not None:
                    err = np.abs(s_a - spec.expected_s_after(np.asarray(s_b, dtype=np.float64)))
                    row.update(
                        {
                            "projection_error_max": float(err.max()),
                            "projection_error_mean": float(err.mean()),
                            "hidden_norm_before": h_norm,
                            "delta_hidden_norm": d_norm,
                            "relative_delta_norm": (
                                d_norm / h_norm if h_norm and h_norm > 0 else float("nan")
                            ),
                            "s_before_norm": float(np.linalg.norm(s_b)),
                            "s_after_norm": float(np.linalg.norm(s_a)),
                        }
                    )
            rows.append(row)

            if log_every and len(rows) % log_every == 0:
                rate = len(rows) / max(time.time() - t0, 1e-9)
                eta = (total - len(rows)) / max(rate, 1e-9) / 60
                print(f"    [{label}] {len(rows)}/{total}  {rate:.2f} row/s  ETA {eta:.1f} min", flush=True)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Агрегация
# ─────────────────────────────────────────────────────────────────────────────
def family_aggregate(rows: list[dict]) -> dict[int, dict]:
    """Усреднение по лейаутам семьи ДО взятия модуля — так выделяется гендерная ось."""
    by_fam: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_fam[r["scenario_family_id"]].append(r)
    out: dict[int, dict] = {}
    for fam, rs in by_fam.items():
        n_man = sum(1 for r in rs if r["prefers_man"])
        n_woman = sum(1 for r in rs if r["prefers_woman"])
        out[fam] = {
            "scenario_family_id": fam,
            "soc_major_title": rs[0]["soc_major_title"],
            "n_rows": len(rs),
            "D_gender": float(np.mean([r["d_gender"] for r in rs])),
            "S_slot": float(np.mean([r["d_slot"] for r in rs])),
            "theta_prob": float(np.mean([r["p_man_norm"] for r in rs])),
            "theta_choice": n_man / (n_man + n_woman) if (n_man + n_woman) else float("nan"),
            "mean_abs_d_row": float(np.mean([abs(r["d_gender"]) for r in rs])),
        }
    return out


def bootstrap_mean_ci(values: np.ndarray, *, n_boot: int, seed: int, ci: float = 0.95) -> dict:
    """Ресемпл единиц кластеризации (семей) с возвращением."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[draws].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "fraction_positive": float(np.mean(values > 0)),
        "n": int(len(values)),
    }


def paired_block(before: np.ndarray, after: np.ndarray, *, n_boot: int, seed: int) -> dict:
    """|before| − |after| (нейтрализация) и signed изменение."""
    reduction = np.abs(before) - np.abs(after)
    signed = after - before
    return {
        "reduction": bootstrap_mean_ci(reduction, n_boot=n_boot, seed=seed),
        "mean_signed_delta": float(np.mean(signed)),
        "mean_abs_before": float(np.mean(np.abs(before))),
        "mean_abs_after": float(np.mean(np.abs(after))),
    }


def behavioral_metrics(
    base_rows: list[dict],
    after_rows: list[dict],
    *,
    n_boot: int,
    seed: int,
) -> dict:
    base_fam = family_aggregate(base_rows)
    after_fam = family_aggregate(after_rows)
    fams = sorted(set(base_fam) & set(after_fam))

    def col(src: dict[int, dict], key: str) -> np.ndarray:
        return np.array([src[f][key] for f in fams], dtype=np.float64)

    gender = paired_block(col(base_fam, "D_gender"), col(after_fam, "D_gender"), n_boot=n_boot, seed=seed)
    slot = paired_block(col(base_fam, "S_slot"), col(after_fam, "S_slot"), n_boot=n_boot, seed=seed + 1)
    theta = paired_block(
        col(base_fam, "theta_prob") - 0.5,
        col(after_fam, "theta_prob") - 0.5,
        n_boot=n_boot,
        seed=seed + 2,
    )

    by_id = {r["id"]: r for r in base_rows}
    d_before, d_after, flips = [], [], []
    for r in after_rows:
        b = by_id.get(r["id"])
        if b is None:
            continue
        d_before.append(b["d_gender"])
        d_after.append(r["d_gender"])
        flips.append(b["choice"] != r["choice"])
    row_level = paired_block(
        np.array(d_before), np.array(d_after), n_boot=n_boot, seed=seed + 3
    )

    per_family = []
    for f in fams:
        b, a = base_fam[f], after_fam[f]
        per_family.append(
            {
                **{k: b[k] for k in ("scenario_family_id", "soc_major_title", "n_rows")},
                "D_before": b["D_gender"],
                "D_after": a["D_gender"],
                "R_gender": abs(b["D_gender"]) - abs(a["D_gender"]),
                "S_before": b["S_slot"],
                "S_after": a["S_slot"],
                "R_slot": abs(b["S_slot"]) - abs(a["S_slot"]),
                "theta_prob_before": b["theta_prob"],
                "theta_prob_after": a["theta_prob"],
                "theta_choice_before": b["theta_choice"],
                "theta_choice_after": a["theta_choice"],
            }
        )

    return {
        "n_families": len(fams),
        "n_rows": len(after_rows),
        "primary": {
            "id": "mean_R_family",
            "formula": "R_i = |mean_layouts d_gender|_before − |...|_after",
            **gender["reduction"],
        },
        "gender_axis": gender,
        "slot_axis": slot,
        "theta_prob_axis": theta,
        "row_level_degenerate": {
            **row_level,
            "note": "построчно |d_gender| ≡ |d_slot| — метрика не разделяет гендер и решительность A/B",
        },
        "flip_rate": float(np.mean(flips)) if flips else float("nan"),
        "per_family": per_family,
    }


def hook_check(rows: list[dict], spec: SubspaceSpec) -> dict | None:
    errs_max = [r["projection_error_max"] for r in rows if "projection_error_max" in r]
    if not errs_max:
        return None
    errs_mean = [r["projection_error_mean"] for r in rows]
    rel = [r["relative_delta_norm"] for r in rows]
    dn = [r["delta_hidden_norm"] for r in rows]
    return {
        "rank": spec.rank,
        "alpha": spec.alpha,
        "max_projection_error": float(np.max(errs_max)),
        "mean_projection_error": float(np.mean(errs_mean)),
        "mean_relative_perturbation": float(np.mean(rel)),
        "max_relative_perturbation": float(np.max(rel)),
        "mean_delta_hidden_norm": float(np.mean(dn)),
        "mean_hidden_norm": float(np.mean([r["hidden_norm_before"] for r in rows])),
        "note": "ε = |Wᵀh' − expected|; при α=1 expected = c. Остаток = квантование dtype модели",
    }


# ─────────────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--subspaces", type=Path, default=DEFAULT_SUBSPACES)
    ap.add_argument("--sample", type=Path, default=STEERING_DIR / "samples" / "h1_stagea_sample_v1.json")
    ap.add_argument("--layers", default=None, help="Слои через запятую (default: все из подпространств)")
    ap.add_argument("--ranks", default=None, help="Ранги через запятую (default: steering_ranks слоя)")
    ap.add_argument("--alphas", default="1.0", help="alpha через запятую; до появления эффекта только 1.0")
    ap.add_argument("--no-random-control", action="store_true")
    ap.add_argument("--random-seeds", default="0")
    ap.add_argument("--limit-items", type=int, default=None)
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260820)
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "results" / "steering")
    ap.add_argument("--tag", default="inlp_v1")
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    sub_meta = load_json(args.subspaces.with_suffix(".json"))
    with np.load(args.subspaces) as z:
        arrays = {k: np.asarray(z[k], dtype=np.float32) for k in z.files}

    sample = load_json(args.sample)
    items = sample["items"][: args.limit_items] if args.limit_items else sample["items"]
    n_rows = sum(len(i["rows"]) for i in items)

    configs = build_configs(
        sub_meta,
        set(arrays),
        layers=parse_int_list(args.layers),
        ranks=parse_int_list(args.ranks),
        alphas=parse_float_list(args.alphas) or [1.0],
        random_seeds=parse_int_list(args.random_seeds) or [0],
        include_random=not args.no_random_control,
    )
    if not configs:
        raise SystemExit("пустой план конфигураций")

    out_dir = args.out_root / "inlp_stage_a" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"INLP Stage A [{args.tag}]: 1 baseline + {len(configs)} конфигураций × {n_rows} строк "
        f"({len(items)} семей) = {(len(configs) + 1) * n_rows} forward"
    )
    print(f"  подпространства: {args.subspaces.name}, target={sub_meta['target']}")
    for L in sub_meta["layers_detail"]:
        print(
            f"    L{L['layer']}: AUC(0)={L['auc_curve'][0]:.3f} → AUC({L['k_found']})="
            f"{L['auc_curve'][-1]:.3f}, k_chance={L['k_chance']}"
        )
    if args.dtype != "float32":
        print(f"  ВНИМАНИЕ: dtype={args.dtype} — часть шага съедается квантованием, см. hook_check")

    print(f"\n[1] Загрузка {args.model} ({args.dtype}, {args.device})...")
    t_load = time.time()
    model, tokenizer = load_model(args.model, dtype=args.dtype, device=args.device)
    scorer = Scorer(model, tokenizer)
    print(f"  готово за {time.time() - t_load:.1f}s, {num_hidden_layers(model.config)} блоков")

    t_all = time.time()
    print(f"\n[2/{len(configs) + 2}] {BASELINE_ID}")
    base_rows = run_config(scorer, model, items, None, label=BASELINE_ID, log_every=args.log_every)
    base_fam = family_aggregate(base_rows)
    base_D = np.array([v["D_gender"] for v in base_fam.values()])
    base_S = np.array([v["S_slot"] for v in base_fam.values()])
    print(
        f"  baseline: mean|D_gender| {np.mean(np.abs(base_D)):.4f}  "
        f"mean|S_slot| {np.mean(np.abs(base_S)):.4f}  "
        f"mean θ_prob {np.mean([v['theta_prob'] for v in base_fam.values()]):.4f}  "
        f"agree with recorded {np.mean([r['choice'] == r['baseline_choice_recorded'] for r in base_rows]):.3f}"
    )

    bdir = out_dir / BASELINE_ID
    bdir.mkdir(parents=True, exist_ok=True)
    with (bdir / "per_row.jsonl").open("w", encoding="utf-8") as f:
        for r in base_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (bdir / "metrics.json").write_text(
        json.dumps(
            {
                "config_id": BASELINE_ID,
                "role": "baseline",
                "n_families": len(base_fam),
                "n_rows": len(base_rows),
                "mean_abs_D_gender": float(np.mean(np.abs(base_D))),
                "mean_abs_S_slot": float(np.mean(np.abs(base_S))),
                "mean_theta_prob": float(np.mean([v["theta_prob"] for v in base_fam.values()])),
                "per_family": list(base_fam.values()),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    ranking: list[dict] = []
    for idx, cfg in enumerate(configs, start=3):
        spec = spec_for(cfg, arrays)
        print(f"\n[{idx}/{len(configs) + 2}] {cfg['id']}  (rank {spec.rank}, α={spec.alpha:g}, L{spec.layer})")
        t0 = time.time()
        rows = run_config(scorer, model, items, spec, label=cfg["id"], log_every=args.log_every)
        runtime = time.time() - t0

        metrics = behavioral_metrics(
            base_rows, rows, n_boot=args.n_bootstrap, seed=args.bootstrap_seed
        )
        hc = hook_check(rows, spec)
        metrics.update(
            {
                "config_id": cfg["id"],
                "role": cfg["role"],
                "config": cfg,
                "runtime_s": round(runtime, 1),
                "hook_check": hc,
            }
        )

        cdir = out_dir / cfg["id"]
        cdir.mkdir(parents=True, exist_ok=True)
        with (cdir / "per_row.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        per_family = metrics.pop("per_family")
        with (cdir / "per_family.jsonl").open("w", encoding="utf-8") as f:
            for r in per_family:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (cdir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        pr = metrics["primary"]
        print(
            f"  R_gender {pr['mean']:+.4f}  CI [{pr['ci_lo']:+.4f}, {pr['ci_hi']:+.4f}]  "
            f"frac(R>0) {pr['fraction_positive']:.2f}  |D| {gender_arrow(metrics)}  "
            f"flip {metrics['flip_rate']:.1%}  {runtime:.1f}s"
        )
        print(
            f"  R_slot {metrics['slot_axis']['reduction']['mean']:+.4f}  "
            f"Δθ_dev {metrics['theta_prob_axis']['reduction']['mean']:+.4f}"
        )
        if hc:
            print(
                f"  hook: ε_max {hc['max_projection_error']:.2e}  ε_mean {hc['mean_projection_error']:.2e}  "
                f"||Δh||/||h|| {hc['mean_relative_perturbation']:.4f}"
            )

        ranking.append(
            {
                "config_id": cfg["id"],
                "role": cfg["role"],
                "basis": cfg["basis"],
                "layer": cfg["layer"],
                "rank": cfg["rank"],
                "alpha": cfg["alpha"],
                "auc_after_removal": cfg["auc_after_removal"],
                "k_chance": cfg["k_chance"],
                "mean_R_gender": pr["mean"],
                "ci_lo": pr["ci_lo"],
                "ci_hi": pr["ci_hi"],
                "median_R_gender": pr["median"],
                "fraction_R_positive": pr["fraction_positive"],
                "mean_signed_delta_D": metrics["gender_axis"]["mean_signed_delta"],
                "mean_abs_D_before": metrics["gender_axis"]["mean_abs_before"],
                "mean_abs_D_after": metrics["gender_axis"]["mean_abs_after"],
                "mean_R_slot": metrics["slot_axis"]["reduction"]["mean"],
                "mean_theta_dev_reduction": metrics["theta_prob_axis"]["reduction"]["mean"],
                "mean_R_row_degenerate": metrics["row_level_degenerate"]["reduction"]["mean"],
                "flip_rate": metrics["flip_rate"],
                "mean_relative_perturbation": hc["mean_relative_perturbation"] if hc else "",
                "mean_delta_hidden_norm": hc["mean_delta_hidden_norm"] if hc else "",
                "max_projection_error": hc["max_projection_error"] if hc else "",
            }
        )

    write_csv(out_dir / "ranking.csv", sorted(ranking, key=lambda r: (r["layer"], r["role"], r["rank"])))
    write_csv(
        out_dir / "auc_vs_behavior.csv",
        auc_vs_behavior(sub_meta, ranking),
    )

    meta = {
        "schema": "steering.inlp_stage_a/v1",
        "tag": args.tag,
        "datetime": datetime.now().isoformat(),
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "subspaces_file": args.subspaces.name,
        "subspaces_sha256": sub_meta.get("arrays_sha256"),
        "target": sub_meta["target"],
        "sample": {
            "file": args.sample.name,
            "source_split": sample.get("source_split"),
            "n_families": len(items),
            "n_rows": n_rows,
            "limited": args.limit_items is not None,
        },
        "alphas": parse_float_list(args.alphas),
        "n_configs": len(configs),
        "config_ids": [BASELINE_ID] + [c["id"] for c in configs],
        "bootstrap": {"n": args.n_bootstrap, "seed": args.bootstrap_seed, "cluster_by": "scenario_family_id"},
        "primary_metric": "mean_R_family = mean_i (|D_i^before| − |D_i^after|), D_i = mean_layouts d_gender",
        "runtime_s": round(time.time() - t_all, 1),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\n=== INLP Stage A [{args.tag}] готово за {meta['runtime_s']:.0f}s ===")
    print(f"  {out_dir}")
    print("\n  layer  rank   AUC(k)   mean R_gender        CI            flip")
    for r in sorted(ranking, key=lambda x: (x["layer"], x["role"] != "candidate", x["rank"])):
        auc = f"{r['auc_after_removal']:.3f}" if r["auc_after_removal"] is not None else "  —  "
        print(
            f"  {r['layer']:>5}  {r['rank']:>4}   {auc}   {r['mean_R_gender']:+.4f}   "
            f"[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]   {r['flip_rate']:.1%}  {r['role']}"
        )
    return 0


def gender_arrow(metrics: dict) -> str:
    g = metrics["gender_axis"]
    return f"{g['mean_abs_before']:.3f}→{g['mean_abs_after']:.3f}"


def auc_vs_behavior(sub_meta: dict, ranking: list[dict]) -> list[dict]:
    """Главная таблица §37: AUC(k) из Phase A × behavioral effect из Phase B."""
    out: list[dict] = []
    by_layer = {int(L["layer"]): L for L in sub_meta["layers_detail"]}
    for layer in sorted({r["layer"] for r in ranking}):
        detail = by_layer[layer]
        out.append(
            {
                "layer": layer,
                "rank": 0,
                "basis": "baseline",
                "auc_after_removal": detail["auc_curve"][0],
                "mean_R_gender": "",
                "ci": "",
                "flip_rate": "",
                "mean_relative_perturbation": "",
            }
        )
        for r in sorted((x for x in ranking if x["layer"] == layer), key=lambda x: (x["basis"], x["rank"])):
            out.append(
                {
                    "layer": layer,
                    "rank": r["rank"],
                    "basis": r["basis"],
                    "auc_after_removal": r["auc_after_removal"] if r["auc_after_removal"] is not None else "",
                    "mean_R_gender": round(r["mean_R_gender"], 5),
                    "ci": f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]",
                    "flip_rate": round(r["flip_rate"], 4),
                    "mean_relative_perturbation": r["mean_relative_perturbation"],
                }
            )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    sys.exit(main())
