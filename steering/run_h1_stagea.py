"""
Stage A: скрининг H1 steering-кандидатов на замороженной выборке val.

Для baseline и каждого кандидата прогоняет 380 строк (95 base items × 4) и
считает нейтральность preference:
    θ_i = M_i/(M_i+W_i) по choice внутри base item
    primary = mean_i |θ_i − 0.5|            (минимизируем)
Параллельно ведётся prob-версия θ (среднее p_man по строкам item) — на 4 строках
choice-θ грубая, вероятностная метрика чувствительнее.

Выход (по одному каталогу на конфигурацию):
    results/steering/stage_a/<tag>/<candidate_id>/per_item.jsonl
    results/steering/stage_a/<tag>/<candidate_id>/metrics.json
    results/steering/stage_a/<tag>/run_meta.json
    results/steering/stage_a/<tag>/ranking.csv

Примеры:
    # локальный smoke: baseline + 2 кандидата на 3 items
    python -m steering.run_h1_stagea --model steering/.cache/model --device cpu \
        --dtype bfloat16 --limit-items 3 --tag smoke \
        --candidates main_center_core__wgperp__L16__center__a1p0,causality_shift__wgperp__L16__shift__b1

    # полный скрининг
    python -m steering.run_h1_stagea --model Qwen/Qwen3.5-2B-Base --device cuda --tag full
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from steering.intervene import (
    ProjectionTrace,
    Scorer,
    load_model,
    num_hidden_layers,
    specs_for_candidate,
    steered,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STEERING_DIR = Path(__file__).resolve().parent
BASELINE_ID = "baseline"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def row_gender_probs(row_out: dict, labels: dict[str, str]) -> tuple[float, float]:
    """p_man / p_woman: сумма constrained-вероятностей по слотам с нужным лейблом."""
    p = {"man": 0.0, "woman": 0.0}
    for slot, semantic in labels.items():
        if semantic in p:
            p[semantic] += float(row_out.get(f"prob_constrained_{slot}", 0.0))
    return p["man"], p["woman"]


def summarize(rows: list[dict], items: list[dict]) -> dict:
    by_family: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_family[r["scenario_family_id"]].append(r)
    soc_of = {i["scenario_family_id"]: i["soc_major_title"] for i in items}

    per_item = []
    for fam, rs in sorted(by_family.items()):
        m = sum(1 for r in rs if r["prefers_man"])
        w = sum(1 for r in rs if r["prefers_woman"])
        theta = m / (m + w) if (m + w) else float("nan")
        theta_prob = float(np.mean([r["p_man_norm"] for r in rs]))
        per_item.append(
            {
                "scenario_family_id": fam,
                "soc_major_title": soc_of[fam],
                "n_rows": len(rs),
                "n_man": m,
                "n_woman": w,
                "theta_choice": theta,
                "theta_prob": theta_prob,
            }
        )

    tc = np.array([p["theta_choice"] for p in per_item], dtype=float)
    tp = np.array([p["theta_prob"] for p in per_item], dtype=float)
    valid = np.isfinite(tc)

    def block(arr: np.ndarray) -> dict:
        return {
            "mean_abs_dev": float(np.mean(np.abs(arr - 0.5))),
            "abs_mean_dev": float(abs(arr.mean() - 0.5)),
            "mean_theta": float(arr.mean()),
            "n_items": int(len(arr)),
        }

    per_soc = []
    for soc in sorted({p["soc_major_title"] for p in per_item}):
        sub = [p for p in per_item if p["soc_major_title"] == soc]
        a_tc = np.array([p["theta_choice"] for p in sub], dtype=float)
        a_tp = np.array([p["theta_prob"] for p in sub], dtype=float)
        per_soc.append(
            {
                "soc_major_title": soc,
                "n_items": len(sub),
                "mean_theta_choice": float(np.nanmean(a_tc)),
                "mean_theta_prob": float(a_tp.mean()),
                "abs_dev_theta_prob": float(abs(a_tp.mean() - 0.5)),
            }
        )

    return {
        "n_items": len(per_item),
        "n_rows": len(rows),
        "theta_choice": block(tc[valid]),
        "theta_prob": block(tp),
        "rates": {
            "slot_a_rate": float(np.mean([r["choice"] == "A" for r in rows])),
            "man_choice_rate": float(np.mean([r["prefers_man"] for r in rows])),
            "narrative_first_rate": float(np.mean([r["prefers_first_mentioned"] for r in rows])),
        },
        "per_soc": per_soc,
        "per_item": per_item,
    }


def hook_check(rows: list[dict]) -> dict | None:
    errs = [r["s_error"] for r in rows if r.get("s_error") is not None]
    if not errs:
        return None
    deltas = [abs(r["s_after"] - r["s_before"]) for r in rows if r.get("s_after") is not None]
    return {
        "mean_abs_error": float(np.mean(np.abs(errs))),
        "max_abs_error": float(np.max(np.abs(errs))),
        "mean_abs_delta_s": float(np.mean(deltas)),
        "relative_error": float(np.mean(np.abs(errs)) / max(float(np.mean(deltas)), 1e-12)),
        "note": "s_after против аналитической формулы; ошибка = квантование dtype модели",
    }


def run_config(
    scorer: Scorer,
    model,
    items: list[dict],
    specs: list,
    *,
    label: str,
    log_every: int,
) -> list[dict]:
    trace = ProjectionTrace() if specs else None
    rows: list[dict] = []
    total = sum(len(i["rows"]) for i in items)
    t0 = time.time()

    for item in items:
        for src in item["rows"]:
            if trace is not None:
                trace.clear()
            with steered(model, specs, trace):
                out = scorer.score(src["prompt"], list(src["valid_labels"]))

            labels = src["labels"]
            p_man, p_woman = row_gender_probs(out, labels)
            denom = p_man + p_woman
            chosen = labels.get(out["choice"])
            first = "man" if src["context_order"] == "man_first" else "woman"

            row = {
                "id": src["id"],
                "scenario_family_id": item["scenario_family_id"],
                "soc_major_title": item["soc_major_title"],
                "position_variant": src["position_variant"],
                "context_order": src["context_order"],
                "labels": labels,
                **out,
                "p_man": p_man,
                "p_woman": p_woman,
                "p_man_norm": p_man / denom if denom > 0 else float("nan"),
                "prefers_man": chosen == "man",
                "prefers_woman": chosen == "woman",
                "prefers_first_mentioned": chosen == first,
                "baseline_choice_recorded": src["baseline_choice"],
                "s_before": None,
                "s_after": None,
                "s_error": None,
            }
            if specs and trace is not None:
                spec = specs[-1]
                s_b = trace.before.get(spec.layer)
                s_a = trace.after.get(spec.layer)
                if s_b is not None:
                    expected = float(spec.expected_s_after(np.array([s_b]))[0])
                    row.update({"s_before": s_b, "s_after": s_a, "s_error": s_a - expected})
            rows.append(row)

            if log_every and len(rows) % log_every == 0:
                rate = len(rows) / (time.time() - t0)
                eta = (total - len(rows)) / max(rate, 1e-9) / 60
                print(f"    [{label}] {len(rows)}/{total}  {rate:.2f} row/s  ETA {eta:.1f} min", flush=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--sample", type=Path, default=STEERING_DIR / "samples" / "h1_stagea_sample_v1.json")
    ap.add_argument("--candidates-file", type=Path, default=STEERING_DIR / "candidates" / "h1_candidates_v1.json")
    ap.add_argument("--vectors", type=Path, default=STEERING_DIR / "vectors" / "h1_vectors_v1.npz")
    ap.add_argument("--candidates", default=None, help="Список id через запятую; по умолчанию все")
    ap.add_argument("--roles", default=None, help="Фильтр по role: candidate,control")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--limit-items", type=int, default=None)
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "results" / "steering")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args(argv)

    sample = load_json(args.sample)
    cand_doc = load_json(args.candidates_file)
    vec_meta = load_json(args.vectors.with_suffix(".json"))
    with np.load(args.vectors) as z:
        vectors = {k: np.asarray(z[k], dtype=np.float32) for k in z.files}
    calib = {e["key"]: e for e in vec_meta["vectors"]}

    items = sample["items"][: args.limit_items] if args.limit_items else sample["items"]

    selected = cand_doc["candidates"]
    if args.roles:
        roles = {r.strip() for r in args.roles.split(",")}
        selected = [c for c in selected if c["role"] in roles]
    if args.candidates:
        wanted = [c.strip() for c in args.candidates.split(",") if c.strip()]
        by_id = {c["id"]: c for c in cand_doc["candidates"]}
        missing = [w for w in wanted if w not in by_id and w != BASELINE_ID]
        if missing:
            raise SystemExit(f"неизвестные кандидаты: {missing}")
        selected = [by_id[w] for w in wanted if w != BASELINE_ID]

    configs: list[tuple[str, dict | None]] = []
    if not args.skip_baseline:
        configs.append((BASELINE_ID, None))
    configs += [(c["id"], c) for c in selected]

    out_dir = args.out_root / "stage_a" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    n_rows = sum(len(i["rows"]) for i in items)
    print(f"Stage A [{args.tag}]: {len(configs)} конфигураций × {n_rows} строк "
          f"({len(items)} base items) = {len(configs) * n_rows} forward")
    if args.dtype != "float32":
        print(f"  ВНИМАНИЕ: dtype={args.dtype}; |Δh| на компоненту ~1e-3 при ||h||≈12 — "
              f"часть шага съедается квантованием, см. hook_check в metrics.json")

    print(f"\n[1] Загрузка {args.model} ({args.dtype}, {args.device})...")
    t_load = time.time()
    model, tokenizer = load_model(args.model, dtype=args.dtype, device=args.device)
    scorer = Scorer(model, tokenizer)
    print(f"  готово за {time.time() - t_load:.1f}s, "
          f"{num_hidden_layers(model.config)} блоков, токены {scorer.tok_ids}")

    t_all = time.time()
    ranking = []
    for idx, (cfg_id, cand) in enumerate(configs, 1):
        specs = specs_for_candidate(cand, vectors, calib) if cand else []
        print(f"\n[{idx}/{len(configs)}] {cfg_id}")
        t0 = time.time()
        rows = run_config(scorer, model, items, specs, label=cfg_id, log_every=args.log_every)
        runtime = time.time() - t0

        metrics = summarize(rows, items)
        metrics.update(
            {
                "config_id": cfg_id,
                "role": cand["role"] if cand else "baseline",
                "candidate": cand,
                "runtime_s": round(runtime, 1),
                "hook_check": hook_check(rows),
                "agreement_with_recorded_run": float(
                    np.mean([r["choice"] == r["baseline_choice_recorded"] for r in rows])
                ),
            }
        )

        cdir = out_dir / cfg_id
        cdir.mkdir(parents=True, exist_ok=True)
        with (cdir / "per_item.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (cdir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        hc = metrics["hook_check"]
        print(
            f"  mean|θ−.5| choice {metrics['theta_choice']['mean_abs_dev']:.4f} | "
            f"prob {metrics['theta_prob']['mean_abs_dev']:.4f} | "
            f"θ̄_prob {metrics['theta_prob']['mean_theta']:.4f} | "
            f"agree with recorded {metrics['agreement_with_recorded_run']:.3f} | {runtime:.1f}s"
        )
        if hc:
            print(
                f"  hook: |Δs| {hc['mean_abs_delta_s']:.5f}, "
                f"ошибка формулы {hc['mean_abs_error']:.5f} ({hc['relative_error']:.1%})"
            )
        ranking.append(
            {
                "config_id": cfg_id,
                "role": metrics["role"],
                "family": cand["family"] if cand else "",
                "vector_id": cand["vector_id"] if cand else "",
                "layer": cand["layers"][0] if cand else "",
                "intervention": cand["intervention"] if cand else "",
                "alpha": (cand.get("alpha") if cand else ""),
                "beta": (cand.get("beta") if cand else ""),
                "mean_abs_theta_dev": metrics["theta_choice"]["mean_abs_dev"],
                "mean_abs_theta_prob_dev": metrics["theta_prob"]["mean_abs_dev"],
                "mean_theta_prob": metrics["theta_prob"]["mean_theta"],
                "slot_a_rate": metrics["rates"]["slot_a_rate"],
                "narrative_first_rate": metrics["rates"]["narrative_first_rate"],
            }
        )

    base = next((r for r in ranking if r["config_id"] == BASELINE_ID), None)
    for r in ranking:
        r["reduction_vs_baseline"] = (
            base["mean_abs_theta_dev"] - r["mean_abs_theta_dev"] if base else ""
        )
        r["prob_reduction_vs_baseline"] = (
            base["mean_abs_theta_prob_dev"] - r["mean_abs_theta_prob_dev"] if base else ""
        )
    ranking.sort(key=lambda r: (r["role"] != "candidate", r["mean_abs_theta_prob_dev"]))

    with (out_dir / "ranking.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ranking[0].keys()))
        writer.writeheader()
        writer.writerows(ranking)

    meta = {
        "schema": "steering.stage_a_run/v1",
        "tag": args.tag,
        "datetime": datetime.now().isoformat(),
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "sample": {
            "file": args.sample.name,
            "n_base_items": len(items),
            "n_rows": n_rows,
            "limited": args.limit_items is not None,
        },
        "candidates_file": args.candidates_file.name,
        "vectors_file": args.vectors.name,
        "vectors_sha256": vec_meta["arrays_sha256"],
        "n_configs": len(configs),
        "config_ids": [c for c, _ in configs],
        "runtime_s": round(time.time() - t_all, 1),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\n=== Stage A [{args.tag}] готово за {meta['runtime_s']:.0f}s ===")
    print(f"  {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
