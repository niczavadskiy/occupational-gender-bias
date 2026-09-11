"""
Stage B: отбор slot steering-кандидатов (preference + capability).

Данные:
  - preference: остаток val (`h1_stageb_sample_v1.json`) — семьи вне Stage A
  - capability: `mmlu_pro_domain_val_v1` + `mmlu_pro_overall_smoke_v1`

Метрики:
  - preference primary: mean_i |φ_prob − 0.5|  (как Stage A)
  - capability: cap_loss = mean_d max(0, acc_base(d) − acc_steered(d))

По умолчанию берёт shortlist Stage A (`slot_stageb_shortlist_v1.json`) или
топ из `ranking.csv` Stage A (`--from-ranking`).

Примеры:
    python -m steering.build_stageb_sample

    python -m steering.run_slot_stageb --model Qwen/Qwen3.5-2B-Base --device cuda \\
        --dtype float32 --tag slot_b_smoke --limit-items 2 --limit-mmlu 20 \\
        --candidates main_center_core__wsperp__L23__center__a2

    python -m steering.run_slot_stageb --model Qwen/Qwen3.5-2B-Base --device cuda \\
        --dtype float32 --tag slot_b_v1 \\
        --from-ranking results/steering/stage_a/slot_full/ranking.csv --keep-top 15
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from steering.intervene import (
    Scorer,
    load_model,
    num_hidden_layers,
    specs_for_candidate,
)
from steering.mmlu_eval import (
    cap_loss_vs_baseline,
    collect_profile_ids,
    expand_domain_profile,
    expand_overall_smoke,
    load_parquet_by_ids,
    run_mmlu,
    summarize_domain_capability,
    summarize_overall_smoke,
)
from steering.run_h1_stagea import BASELINE_ID, hook_check, load_json, run_config
from steering.run_slot_stagea import summarize_slot

REPO_ROOT = Path(__file__).resolve().parents[1]
STEERING_DIR = Path(__file__).resolve().parent

DEFAULT_SHORTLIST = STEERING_DIR / "candidates" / "slot_stageb_shortlist_v1.json"


def load_shortlist_ids(path: Path) -> list[str]:
    doc = load_json(path)
    return [c["id"] if isinstance(c, dict) else str(c) for c in doc["candidates"]]


def ids_from_ranking(path: Path, keep_top: int) -> list[str]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    cands = [r["config_id"] for r in rows if r.get("role") == "candidate"]
    return cands[:keep_top]


def select_candidates(
    cand_doc: dict,
    *,
    explicit: str | None,
    shortlist: Path | None,
    ranking: Path | None,
    keep_top: int,
    roles: str | None,
) -> list[dict]:
    by_id = {c["id"]: c for c in cand_doc["candidates"]}
    if explicit:
        wanted = [c.strip() for c in explicit.split(",") if c.strip()]
    elif ranking is not None:
        wanted = ids_from_ranking(ranking, keep_top)
    elif shortlist is not None and shortlist.exists():
        wanted = load_shortlist_ids(shortlist)
    else:
        wanted = [c["id"] for c in cand_doc["candidates"] if c["role"] == "candidate"][
            :keep_top
        ]

    missing = [w for w in wanted if w not in by_id]
    if missing:
        raise SystemExit(f"неизвестные кандидаты: {missing}")
    selected = [by_id[w] for w in wanted]
    if roles:
        allow = {r.strip() for r in roles.split(",")}
        selected = [c for c in selected if c["role"] in allow]
    return selected


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--sample", type=Path, default=STEERING_DIR / "samples" / "h1_stageb_sample_v1.json")
    ap.add_argument(
        "--candidates-file",
        type=Path,
        default=STEERING_DIR / "candidates" / "slot_candidates_v1.json",
    )
    ap.add_argument("--vectors", type=Path, default=STEERING_DIR / "vectors" / "slot_vectors_v1.npz")
    ap.add_argument("--shortlist", type=Path, default=DEFAULT_SHORTLIST)
    ap.add_argument("--from-ranking", type=Path, default=None)
    ap.add_argument("--keep-top", type=int, default=15)
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--roles", default=None)
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--skip-preference", action="store_true")
    ap.add_argument("--skip-capability", action="store_true")
    ap.add_argument("--limit-items", type=int, default=None, help="preference base items")
    ap.add_argument("--limit-mmlu", type=int, default=None, help="обрезать domain+smoke вместе")
    ap.add_argument(
        "--domain-profile",
        type=Path,
        default=STEERING_DIR / "profiles" / "mmlu_pro_domain_val_v1.json",
    )
    ap.add_argument(
        "--smoke-profile",
        type=Path,
        default=STEERING_DIR / "profiles" / "mmlu_pro_overall_smoke_v1.json",
    )
    ap.add_argument(
        "--mmlu-parquet",
        type=Path,
        default=STEERING_DIR / ".cache" / "mmlu_pro_test.parquet",
    )
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "results" / "steering")
    ap.add_argument("--tag", default="slot_b_v1")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument(
        "--cap-loss-max",
        type=float,
        default=0.03,
        help="мягкий порог: кандидаты с cap_loss выше — flag, не дисквал",
    )
    args = ap.parse_args(argv)

    if not args.sample.exists():
        raise SystemExit(
            f"нет {args.sample.name}. Сначала: python -m steering.build_stageb_sample"
        )

    sample = load_json(args.sample)
    cand_doc = load_json(args.candidates_file)
    vec_meta = load_json(args.vectors.with_suffix(".json"))
    with np.load(args.vectors) as z:
        vectors = {k: np.asarray(z[k], dtype=np.float32) for k in z.files}
    calib = {e["key"]: e for e in vec_meta["vectors"]}

    items = sample["items"][: args.limit_items] if args.limit_items else sample["items"]
    selected = select_candidates(
        cand_doc,
        explicit=args.candidates,
        shortlist=None if args.from_ranking or args.candidates else args.shortlist,
        ranking=args.from_ranking,
        keep_top=args.keep_top,
        roles=args.roles,
    )

    configs: list[tuple[str, dict | None]] = []
    if not args.skip_baseline:
        configs.append((BASELINE_ID, None))
    configs += [(c["id"], c) for c in selected]

    domain_items: list[dict] = []
    smoke_items: list[dict] = []
    domain_profile = smoke_profile = None
    if not args.skip_capability:
        if not args.mmlu_parquet.exists():
            raise SystemExit(
                f"нет {args.mmlu_parquet}. См. README: curl MMLU-Pro parquet в steering/.cache/"
            )
        domain_profile = load_json(args.domain_profile)
        smoke_profile = load_json(args.smoke_profile)
        bank = load_parquet_by_ids(
            args.mmlu_parquet, collect_profile_ids(domain_profile, smoke_profile)
        )
        domain_items = expand_domain_profile(domain_profile, bank)
        smoke_items = expand_overall_smoke(smoke_profile, bank)
        if args.limit_mmlu is not None:
            # детерминированно: сначала domain, потом smoke
            take = args.limit_mmlu
            domain_items = domain_items[:take]
            rest = max(0, take - len(domain_items))
            smoke_items = smoke_items[:rest]

    out_dir = args.out_root / "stage_b" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    n_pref = 0 if args.skip_preference else sum(len(i["rows"]) for i in items)
    n_cap = len(domain_items) + len(smoke_items)
    print(
        f"Stage B slot [{args.tag}]: {len(configs)} конфигураций × "
        f"(pref {n_pref} + mmlu {n_cap}) = {len(configs) * (n_pref + n_cap)} forward"
    )

    print(f"\n[1] Загрузка {args.model} ({args.dtype}, {args.device})...")
    t_load = time.time()
    model, tokenizer = load_model(args.model, dtype=args.dtype, device=args.device)
    scorer = Scorer(model, tokenizer)
    print(
        f"  готово за {time.time() - t_load:.1f}s, "
        f"{num_hidden_layers(model.config)} блоков, letters {sorted(scorer.tok_ids)}"
    )

    t_all = time.time()
    ranking: list[dict] = []
    base_domain_summary: dict | None = None
    base_smoke_summary: dict | None = None

    for idx, (cfg_id, cand) in enumerate(configs, 1):
        specs = specs_for_candidate(cand, vectors, calib) if cand else []
        print(f"\n[{idx}/{len(configs)}] {cfg_id}")
        t0 = time.time()
        cdir = out_dir / cfg_id
        cdir.mkdir(parents=True, exist_ok=True)

        pref_metrics = None
        if not args.skip_preference:
            rows = run_config(
                scorer, model, items, specs, label=f"{cfg_id}/pref", log_every=args.log_every
            )
            pref_metrics = summarize_slot(rows, items)
            pref_metrics.update(
                {
                    "config_id": cfg_id,
                    "role": cand["role"] if cand else "baseline",
                    "hypothesis": "slot",
                    "hook_check": hook_check(rows),
                    "agreement_with_recorded_run": float(
                        np.mean([r["choice"] == r["baseline_choice_recorded"] for r in rows])
                    ),
                }
            )
            with (cdir / "preference_per_item.jsonl").open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            (cdir / "preference_metrics.json").write_text(
                json.dumps(pref_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"  pref |φ_prob| {pref_metrics['phi_prob']['mean_abs_dev']:.4f} | "
                f"slot_a {pref_metrics['rates']['slot_a_rate']:.3f} | "
                f"|θ_prob| {pref_metrics['theta_prob']['mean_abs_dev']:.4f}"
            )

        domain_summary = smoke_summary = cap = None
        if not args.skip_capability:
            d_rows = run_mmlu(
                scorer,
                model,
                domain_items,
                specs,
                label=f"{cfg_id}/domain",
                log_every=args.log_every,
            )
            s_rows = run_mmlu(
                scorer,
                model,
                smoke_items,
                specs,
                label=f"{cfg_id}/smoke",
                log_every=args.log_every,
            )
            domain_summary = summarize_domain_capability(d_rows, domain_profile)
            smoke_summary = summarize_overall_smoke(s_rows)
            with (cdir / "mmlu_domain.jsonl").open("w", encoding="utf-8") as f:
                for r in d_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            with (cdir / "mmlu_smoke.jsonl").open("w", encoding="utf-8") as f:
                for r in s_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            (cdir / "capability_domain.json").write_text(
                json.dumps(domain_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            (cdir / "capability_smoke.json").write_text(
                json.dumps(smoke_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

            if cfg_id == BASELINE_ID:
                base_domain_summary = domain_summary
                base_smoke_summary = smoke_summary
                cap = {
                    "cap_loss": 0.0,
                    "by_tier": {"primary": 0.0, "secondary": 0.0, "generic": 0.0},
                    "per_domain": [],
                    "n_domains_worse": 0,
                    "smoke_acc_drop": 0.0,
                }
            elif base_domain_summary is not None:
                cap = cap_loss_vs_baseline(domain_summary, base_domain_summary)
                smoke_drop = max(
                    0.0,
                    float(base_smoke_summary["accuracy_micro"])
                    - float(smoke_summary["accuracy_micro"]),
                )
                cap["smoke_acc_drop"] = smoke_drop
                (cdir / "cap_loss.json").write_text(
                    json.dumps(cap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                print(
                    f"  cap_loss {cap['cap_loss']:.4f} | "
                    f"domain macro {domain_summary['accuracy_macro']:.3f} | "
                    f"smoke {smoke_summary['accuracy_micro']:.3f} "
                    f"(drop {smoke_drop:.4f})"
                )
            else:
                print("  capability: baseline ещё не посчитан — cap_loss отложен")

        runtime = time.time() - t0
        row = {
            "config_id": cfg_id,
            "role": cand["role"] if cand else "baseline",
            "family": cand["family"] if cand else "",
            "vector_id": cand["vector_id"] if cand else "",
            "layer": cand["layers"][0] if cand else "",
            "intervention": cand["intervention"] if cand else "",
            "alpha": (cand.get("alpha") if cand else ""),
            "beta": (cand.get("beta") if cand else ""),
            "mean_abs_phi_dev": pref_metrics["phi_choice"]["mean_abs_dev"] if pref_metrics else "",
            "mean_abs_phi_prob_dev": pref_metrics["phi_prob"]["mean_abs_dev"] if pref_metrics else "",
            "mean_phi_prob": pref_metrics["phi_prob"]["mean"] if pref_metrics else "",
            "slot_a_rate": pref_metrics["rates"]["slot_a_rate"] if pref_metrics else "",
            "abs_slot_a_rate_dev": abs(pref_metrics["rates"]["slot_a_rate"] - 0.5)
            if pref_metrics
            else "",
            "mean_abs_theta_prob_dev": pref_metrics["theta_prob"]["mean_abs_dev"]
            if pref_metrics
            else "",
            "narrative_first_rate": pref_metrics["rates"]["narrative_first_rate"]
            if pref_metrics
            else "",
            "domain_acc_macro": domain_summary["accuracy_macro"] if domain_summary else "",
            "smoke_acc": smoke_summary["accuracy_micro"] if smoke_summary else "",
            "cap_loss": cap["cap_loss"] if cap else "",
            "cap_loss_primary": (cap["by_tier"].get("primary") if cap else ""),
            "smoke_acc_drop": (cap.get("smoke_acc_drop") if cap else ""),
            "cap_loss_flag": (
                ""
                if not cap or cap["cap_loss"] is None or cfg_id == BASELINE_ID
                else bool(float(cap["cap_loss"]) > args.cap_loss_max)
            ),
            "runtime_s": round(runtime, 1),
        }
        ranking.append(row)
        print(f"  runtime {runtime:.1f}s")

    # Если baseline не первый — досчитать cap_loss
    base_rank = next((r for r in ranking if r["config_id"] == BASELINE_ID), None)
    if base_domain_summary is not None:
        for r in ranking:
            if r["config_id"] == BASELINE_ID or r["cap_loss"] != "":
                continue
            cdir = out_dir / r["config_id"]
            steered_sum = load_json(cdir / "capability_domain.json")
            smoke_sum = load_json(cdir / "capability_smoke.json")
            cap = cap_loss_vs_baseline(steered_sum, base_domain_summary)
            smoke_drop = max(
                0.0,
                float(base_smoke_summary["accuracy_micro"]) - float(smoke_sum["accuracy_micro"]),
            )
            cap["smoke_acc_drop"] = smoke_drop
            (cdir / "cap_loss.json").write_text(
                json.dumps(cap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            r["cap_loss"] = cap["cap_loss"]
            r["cap_loss_primary"] = cap["by_tier"].get("primary")
            r["smoke_acc_drop"] = smoke_drop
            r["cap_loss_flag"] = bool(float(cap["cap_loss"]) > args.cap_loss_max)

    if base_rank and not args.skip_preference:
        for r in ranking:
            if r["mean_abs_phi_prob_dev"] == "":
                continue
            r["phi_reduction_vs_baseline"] = (
                float(base_rank["mean_abs_phi_prob_dev"]) - float(r["mean_abs_phi_prob_dev"])
            )
            r["slot_rate_reduction_vs_baseline"] = (
                float(base_rank["abs_slot_a_rate_dev"]) - float(r["abs_slot_a_rate_dev"])
            )
    else:
        for r in ranking:
            r["phi_reduction_vs_baseline"] = ""
            r["slot_rate_reduction_vs_baseline"] = ""

    # Отбор: сначала по preference среди candidate; cap_loss — flag.
    ranking.sort(
        key=lambda r: (
            r["role"] != "candidate",
            float(r["mean_abs_phi_prob_dev"])
            if r["mean_abs_phi_prob_dev"] != ""
            else 1e9,
            float(r["cap_loss"]) if r["cap_loss"] != "" else 1e9,
        )
    )

    with (out_dir / "ranking.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ranking[0].keys()))
        writer.writeheader()
        writer.writerows(ranking)

    keep = [
        r
        for r in ranking
        if r["role"] == "candidate" and r.get("cap_loss_flag") is not True
    ][:3]
    if len(keep) < 3:
        # если все flagged — всё равно взять топ-3 по preference
        keep = [r for r in ranking if r["role"] == "candidate"][:3]
    (out_dir / "keep.json").write_text(
        json.dumps(
            {
                "n": len(keep),
                "cap_loss_max": args.cap_loss_max,
                "rule": "top by mean_abs_phi_prob_dev among candidates; cap_loss > max → flag",
                "candidates": [
                    {
                        "config_id": r["config_id"],
                        "mean_abs_phi_prob_dev": r["mean_abs_phi_prob_dev"],
                        "cap_loss": r["cap_loss"],
                        "slot_a_rate": r["slot_a_rate"],
                        "cap_loss_flag": r["cap_loss_flag"],
                    }
                    for r in keep
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    meta = {
        "schema": "steering.stage_b_slot_run/v1",
        "hypothesis": "slot",
        "tag": args.tag,
        "datetime": datetime.now().isoformat(),
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "sample": {
            "file": args.sample.name,
            "n_base_items": len(items),
            "n_rows": n_pref,
            "limited": args.limit_items is not None,
        },
        "capability": {
            "domain_profile": args.domain_profile.name if not args.skip_capability else None,
            "smoke_profile": args.smoke_profile.name if not args.skip_capability else None,
            "n_domain": len(domain_items),
            "n_smoke": len(smoke_items),
            "cap_loss_max": args.cap_loss_max,
            "skipped": args.skip_capability,
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
        "capability_metric": "cap_loss",
        "keep": [k["config_id"] for k in keep],
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\n=== Stage B slot [{args.tag}] готово за {meta['runtime_s']:.0f}s ===")
    print(f"  {out_dir}")
    print("  keep → Stage C:")
    for k in keep:
        print(
            f"    {k['config_id']}  |φ_prob|={k['mean_abs_phi_prob_dev']}  "
            f"cap_loss={k['cap_loss']}  slot_a={k['slot_a_rate']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
