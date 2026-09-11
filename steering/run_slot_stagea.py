"""
Stage A: скрининг slot steering-кандидатов (preference слота A/B).

Цель: φ_i = A_i/(A_i+B_i) → 0.5 на base item.
Primary: mean_i |φ_i − 0.5| (choice + prob-версия по p_A).

Тот же frozen sample, что H1 (`h1_stagea_sample_v1.json`).

Примеры:
    python -m steering.run_slot_stagea --model Qwen/Qwen3.5-2B-Base --device cuda \\
        --dtype float32 --limit-items 3 --tag slot_smoke \\
        --candidates main_project_out__wsperp__L23__project_out__a1,causality_shift__wsperp__L23__shift__b1

    python -m steering.run_slot_stagea --model Qwen/Qwen3.5-2B-Base --device cuda \\
        --dtype float32 --tag slot_full
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
    Scorer,
    load_model,
    num_hidden_layers,
    specs_for_candidate,
)
from steering.run_h1_stagea import (
    BASELINE_ID,
    hook_check,
    load_json,
    run_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STEERING_DIR = Path(__file__).resolve().parent


def summarize_slot(rows: list[dict], items: list[dict]) -> dict:
    by_family: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_family[r["scenario_family_id"]].append(r)
    soc_of = {i["scenario_family_id"]: i["soc_major_title"] for i in items}

    per_item = []
    for fam, rs in sorted(by_family.items()):
        n_a = sum(1 for r in rs if r["choice"] == "A")
        n_b = sum(1 for r in rs if r["choice"] == "B")
        phi = n_a / (n_a + n_b) if (n_a + n_b) else float("nan")
        p_a = float(np.mean([float(r.get("prob_constrained_A") or 0.0) for r in rs]))
        m = sum(1 for r in rs if r["prefers_man"])
        w = sum(1 for r in rs if r["prefers_woman"])
        theta = m / (m + w) if (m + w) else float("nan")
        theta_prob = float(np.mean([r["p_man_norm"] for r in rs]))
        per_item.append(
            {
                "scenario_family_id": fam,
                "soc_major_title": soc_of[fam],
                "n_rows": len(rs),
                "n_a": n_a,
                "n_b": n_b,
                "phi_choice": phi,
                "phi_prob": p_a,
                "theta_choice": theta,
                "theta_prob": theta_prob,
            }
        )

    pc = np.array([p["phi_choice"] for p in per_item], dtype=float)
    pp = np.array([p["phi_prob"] for p in per_item], dtype=float)
    tc = np.array([p["theta_choice"] for p in per_item], dtype=float)
    tp = np.array([p["theta_prob"] for p in per_item], dtype=float)
    valid_phi = np.isfinite(pc)
    valid_th = np.isfinite(tc)

    def block(arr: np.ndarray) -> dict:
        return {
            "mean_abs_dev": float(np.mean(np.abs(arr - 0.5))),
            "abs_mean_dev": float(abs(arr.mean() - 0.5)),
            "mean": float(arr.mean()),
            "n_items": int(len(arr)),
        }

    return {
        "n_items": len(per_item),
        "n_rows": len(rows),
        "phi_choice": block(pc[valid_phi]),
        "phi_prob": block(pp),
        "theta_choice": block(tc[valid_th]),
        "theta_prob": block(tp),
        "rates": {
            "slot_a_rate": float(np.mean([r["choice"] == "A" for r in rows])),
            "man_choice_rate": float(np.mean([r["prefers_man"] for r in rows])),
            "narrative_first_rate": float(np.mean([r["prefers_first_mentioned"] for r in rows])),
        },
        "per_item": per_item,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--sample", type=Path, default=STEERING_DIR / "samples" / "h1_stagea_sample_v1.json")
    ap.add_argument(
        "--candidates-file",
        type=Path,
        default=STEERING_DIR / "candidates" / "slot_candidates_v1.json",
    )
    ap.add_argument("--vectors", type=Path, default=STEERING_DIR / "vectors" / "slot_vectors_v1.npz")
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--roles", default=None)
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--limit-items", type=int, default=None)
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "results" / "steering")
    ap.add_argument("--tag", default="slot_full")
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
    print(
        f"Stage A slot [{args.tag}]: {len(configs)} конфигураций × {n_rows} строк "
        f"({len(items)} base items) = {len(configs) * n_rows} forward"
    )

    print(f"\n[1] Загрузка {args.model} ({args.dtype}, {args.device})...")
    t_load = time.time()
    model, tokenizer = load_model(args.model, dtype=args.dtype, device=args.device)
    scorer = Scorer(model, tokenizer)
    print(
        f"  готово за {time.time() - t_load:.1f}s, "
        f"{num_hidden_layers(model.config)} блоков, токены {scorer.tok_ids}"
    )

    t_all = time.time()
    ranking = []
    for idx, (cfg_id, cand) in enumerate(configs, 1):
        specs = specs_for_candidate(cand, vectors, calib) if cand else []
        print(f"\n[{idx}/{len(configs)}] {cfg_id}")
        t0 = time.time()
        rows = run_config(scorer, model, items, specs, label=cfg_id, log_every=args.log_every)
        runtime = time.time() - t0

        metrics = summarize_slot(rows, items)
        metrics.update(
            {
                "config_id": cfg_id,
                "role": cand["role"] if cand else "baseline",
                "hypothesis": "slot",
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
            f"  mean|φ−.5| choice {metrics['phi_choice']['mean_abs_dev']:.4f} | "
            f"prob {metrics['phi_prob']['mean_abs_dev']:.4f} | "
            f"φ̄_prob {metrics['phi_prob']['mean']:.4f} | "
            f"slot_a {metrics['rates']['slot_a_rate']:.3f} | "
            f"gender|θ|prob {metrics['theta_prob']['mean_abs_dev']:.4f} | "
            f"agree {metrics['agreement_with_recorded_run']:.3f} | {runtime:.1f}s"
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
                "mean_abs_phi_dev": metrics["phi_choice"]["mean_abs_dev"],
                "mean_abs_phi_prob_dev": metrics["phi_prob"]["mean_abs_dev"],
                "mean_phi_prob": metrics["phi_prob"]["mean"],
                "slot_a_rate": metrics["rates"]["slot_a_rate"],
                "abs_slot_a_rate_dev": abs(metrics["rates"]["slot_a_rate"] - 0.5),
                "mean_abs_theta_prob_dev": metrics["theta_prob"]["mean_abs_dev"],
                "narrative_first_rate": metrics["rates"]["narrative_first_rate"],
            }
        )

    base = next((r for r in ranking if r["config_id"] == BASELINE_ID), None)
    for r in ranking:
        r["phi_reduction_vs_baseline"] = (
            base["mean_abs_phi_prob_dev"] - r["mean_abs_phi_prob_dev"] if base else ""
        )
        r["slot_rate_reduction_vs_baseline"] = (
            base["abs_slot_a_rate_dev"] - r["abs_slot_a_rate_dev"] if base else ""
        )
    ranking.sort(key=lambda r: (r["role"] != "candidate", r["mean_abs_phi_prob_dev"]))

    with (out_dir / "ranking.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ranking[0].keys()))
        writer.writeheader()
        writer.writerows(ranking)

    meta = {
        "schema": "steering.stage_a_slot_run/v1",
        "hypothesis": "slot",
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
        "primary_metric": "mean_abs_phi_prob_dev",
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\n=== Stage A slot [{args.tag}] готово за {meta['runtime_s']:.0f}s ===")
    print(f"  {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
