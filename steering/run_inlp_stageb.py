"""
INLP Stage B: capability (MMLU-Pro) + optional preference re-score.

После confirmatory на test и (опционально) α-sweep проверяем non-inferiority
capability: cap_loss vs baseline на domain_val + overall_smoke.

Preference по умолчанию выключен (уже закрыт Stage A + confirmatory). Включить:
`--with-preference` на остатке val (`h1_stageb_sample_v1.json`) или другом sample.

Метрики:
  - capability: cap_loss = mean_d max(0, acc_base(d) − acc_steered(d))
  - preference (optional): mean R_gender как в run_inlp_stagea

Примеры:
    # smoke: 1 кандидат, 20 MMLU
    python -m steering.run_inlp_stageb --model Qwen/Qwen3.5-2B-Base --device cuda \\
        --dtype float32 --tag inlp_b_smoke --limit-mmlu 20 \\
        --candidates inlp__L15__k8__a1

    # полный capability shortlist
    python -m steering.run_inlp_stageb --model Qwen/Qwen3.5-2B-Base --device cuda \\
        --dtype float32 --tag inlp_b_v1

    # после α-sweep: явные победители
    python -m steering.run_inlp_stageb --device cuda --dtype float32 --tag inlp_b_alpha \\
        --candidates inlp__L15__k8__a0p75,inlp__L15__k16__a1 \\
        --layers 15 --ranks 8,16 --alphas 0.75,1.0 --no-random-control
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from steering.intervene import Scorer, load_model, num_hidden_layers
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
from steering.run_inlp_stagea import (
    BASELINE_ID,
    DEFAULT_SUBSPACES,
    behavioral_metrics,
    build_configs,
    family_aggregate,
    hook_check,
    load_json,
    parse_float_list,
    parse_int_list,
    run_config,
    spec_for,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STEERING_DIR = Path(__file__).resolve().parent
DEFAULT_SHORTLIST = STEERING_DIR / "candidates" / "inlp_stageb_shortlist_v1.json"


def load_shortlist(path: Path) -> list[dict]:
    doc = load_json(path)
    out = []
    for c in doc["candidates"]:
        if isinstance(c, str):
            out.append({"id": c, "role": "candidate"})
        else:
            out.append(
                {
                    "id": c["id"],
                    "role": c.get("role", "candidate"),
                }
            )
    return out


def select_configs(
    all_cfgs: list[dict],
    *,
    explicit: str | None,
    shortlist: Path | None,
    roles: str | None,
) -> list[dict]:
    by_id = {c["id"]: c for c in all_cfgs}
    if explicit:
        wanted = [x.strip() for x in explicit.split(",") if x.strip()]
        source_roles: dict[str, str] = {}
    elif shortlist is not None and shortlist.exists():
        entries = load_shortlist(shortlist)
        wanted = [e["id"] for e in entries]
        source_roles = {e["id"]: e["role"] for e in entries}
    else:
        wanted = [c["id"] for c in all_cfgs if c["role"] == "candidate"]
        source_roles = {}

    missing = [w for w in wanted if w not in by_id]
    if missing:
        raise SystemExit(
            "неизвестные config_id (проверьте --layers/--ranks/--alphas и subspaces):\n  "
            + "\n  ".join(missing)
            + f"\nдоступно {len(by_id)} конфигов, пример: {next(iter(by_id), '—')}"
        )

    selected = []
    for wid in wanted:
        cfg = dict(by_id[wid])
        if wid in source_roles:
            cfg["role"] = source_roles[wid]
        selected.append(cfg)

    if roles:
        allow = {r.strip() for r in roles.split(",")}
        selected = [c for c in selected if c["role"] in allow]
    return selected


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--subspaces", type=Path, default=DEFAULT_SUBSPACES)
    ap.add_argument(
        "--sample",
        type=Path,
        default=STEERING_DIR / "samples" / "h1_stageb_sample_v1.json",
        help="только при --with-preference",
    )
    ap.add_argument("--shortlist", type=Path, default=DEFAULT_SHORTLIST)
    ap.add_argument("--candidates", default=None, help="config_id через запятую")
    ap.add_argument("--roles", default=None, help="фильтр role: candidate,control")
    ap.add_argument("--layers", default="15")
    ap.add_argument("--ranks", default="8,16")
    ap.add_argument("--alphas", default="1.0")
    ap.add_argument("--no-random-control", action="store_true")
    ap.add_argument("--random-seeds", default="0")
    ap.add_argument("--with-preference", action="store_true")
    ap.add_argument("--skip-capability", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true", help="только domain profile, без overall_smoke")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--limit-items", type=int, default=None)
    ap.add_argument("--limit-mmlu", type=int, default=None)
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
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260820)
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "results" / "steering")
    ap.add_argument("--tag", default="inlp_b_v1")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument(
        "--cap-loss-max",
        type=float,
        default=0.03,
        help="мягкий порог: cap_loss выше → flag, не дисквал",
    )
    args = ap.parse_args(argv)
    return run_from_args(
        args,
        stage_label="INLP Stage B",
        out_subdir="inlp_stage_b",
        schema="steering.stage_b_inlp_run/v1",
        reselect_keep=True,
    )


def run_from_args(
    args: argparse.Namespace,
    *,
    stage_label: str,
    out_subdir: str,
    schema: str,
    reselect_keep: bool,
) -> int:
    with_preference = bool(getattr(args, "with_preference", False))
    skip_smoke = bool(getattr(args, "skip_smoke", False))

    if args.skip_capability and not with_preference:
        raise SystemExit("нужен хотя бы capability или preference")
    if with_preference and args.skip_baseline:
        raise SystemExit("preference требует baseline (уберите --skip-baseline)")

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

    all_cfgs = build_configs(
        sub_meta,
        set(arrays),
        layers=parse_int_list(args.layers),
        ranks=parse_int_list(args.ranks),
        alphas=parse_float_list(args.alphas) or [1.0],
        random_seeds=parse_int_list(args.random_seeds) or [0],
        include_random=not args.no_random_control,
    )
    selected = select_configs(
        all_cfgs,
        explicit=args.candidates,
        shortlist=None if args.candidates else args.shortlist,
        roles=args.roles,
    )
    if not selected:
        raise SystemExit("пустой список кандидатов")

    items: list[dict] = []
    if with_preference:
        if not args.sample.exists():
            raise SystemExit(f"нет {args.sample}")
        sample = load_json(args.sample)
        items = sample["items"][: args.limit_items] if args.limit_items else sample["items"]

    domain_items: list[dict] = []
    smoke_items: list[dict] = []
    domain_profile = smoke_profile = None
    if not args.skip_capability:
        if not args.mmlu_parquet.exists():
            raise SystemExit(
                f"нет {args.mmlu_parquet}. См. README: curl MMLU-Pro parquet в steering/.cache/"
            )
        domain_profile = load_json(args.domain_profile)
        profiles = [domain_profile]
        if not skip_smoke:
            smoke_profile = load_json(args.smoke_profile)
            profiles.append(smoke_profile)
        bank = load_parquet_by_ids(args.mmlu_parquet, collect_profile_ids(*profiles))
        domain_items = expand_domain_profile(domain_profile, bank)
        if not skip_smoke:
            smoke_items = expand_overall_smoke(smoke_profile, bank)
        if args.limit_mmlu is not None:
            take = args.limit_mmlu
            domain_items = domain_items[:take]
            rest = max(0, take - len(domain_items))
            smoke_items = smoke_items[:rest]

    configs: list[tuple[str, dict | None]] = []
    if not args.skip_baseline:
        configs.append((BASELINE_ID, None))
    configs += [(c["id"], c) for c in selected]

    out_dir = args.out_root / out_subdir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    n_pref = sum(len(i["rows"]) for i in items) if with_preference else 0
    n_cap = len(domain_items) + len(smoke_items)
    print(
        f"{stage_label} [{args.tag}]: {len(configs)} конфигураций × "
        f"(pref {n_pref} + mmlu {n_cap}) = {len(configs) * (n_pref + n_cap)} forward"
    )
    print(f"  subspaces={args.subspaces.name}  candidates={[c for c, _ in configs if c != BASELINE_ID]}")

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
    base_rows: list[dict] | None = None
    base_domain_summary: dict | None = None
    base_smoke_summary: dict | None = None

    for idx, (cfg_id, cand) in enumerate(configs, 1):
        spec = spec_for(cand, arrays) if cand else None
        specs = [spec] if spec else []
        print(f"\n[{idx}/{len(configs)}] {cfg_id}")
        t0 = time.time()
        cdir = out_dir / cfg_id
        cdir.mkdir(parents=True, exist_ok=True)

        pref_metrics = None
        if with_preference:
            rows = run_config(
                scorer, model, items, spec, label=f"{cfg_id}/pref", log_every=args.log_every
            )
            if cfg_id == BASELINE_ID:
                base_rows = rows
                fam = family_aggregate(rows)
                pref_metrics = {
                    "config_id": cfg_id,
                    "role": "baseline",
                    "n_families": len(fam),
                    "n_rows": len(rows),
                    "mean_abs_D_gender": float(
                        np.mean([abs(v["D_gender"]) for v in fam.values()])
                    ),
                    "mean_R_gender": 0.0,
                    "flip_rate": 0.0,
                }
            else:
                assert base_rows is not None, "baseline должен идти первым для preference"
                pref_metrics = behavioral_metrics(
                    base_rows, rows, n_boot=args.n_bootstrap, seed=args.bootstrap_seed
                )
                pref_metrics.update(
                    {
                        "config_id": cfg_id,
                        "role": cand["role"] if cand else "baseline",
                        "config": cand,
                        "hook_check": hook_check(rows, spec) if spec else None,
                    }
                )
            with (cdir / "preference_per_row.jsonl").open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            (cdir / "preference_metrics.json").write_text(
                json.dumps(pref_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            if cfg_id == BASELINE_ID:
                print(f"  pref baseline |D| {pref_metrics['mean_abs_D_gender']:.4f}")
            else:
                prim = pref_metrics["primary"]
                print(
                    f"  pref R_gender {prim['mean']:+.4f} "
                    f"[{prim['ci_lo']:+.4f},{prim['ci_hi']:+.4f}]  "
                    f"flip {pref_metrics['flip_rate']:.1%}"
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
            domain_summary = summarize_domain_capability(d_rows, domain_profile)
            with (cdir / "mmlu_domain.jsonl").open("w", encoding="utf-8") as f:
                for r in d_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            (cdir / "capability_domain.json").write_text(
                json.dumps(domain_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

            if smoke_items:
                s_rows = run_mmlu(
                    scorer,
                    model,
                    smoke_items,
                    specs,
                    label=f"{cfg_id}/smoke",
                    log_every=args.log_every,
                )
                smoke_summary = summarize_overall_smoke(s_rows)
                with (cdir / "mmlu_smoke.jsonl").open("w", encoding="utf-8") as f:
                    for r in s_rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
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
                smoke_txt = (
                    f" | smoke {smoke_summary['accuracy_micro']:.3f}" if smoke_summary else ""
                )
                print(f"  cap baseline domain macro {domain_summary['accuracy_macro']:.3f}{smoke_txt}")
            elif base_domain_summary is not None:
                cap = cap_loss_vs_baseline(domain_summary, base_domain_summary)
                if smoke_summary is not None and base_smoke_summary is not None:
                    smoke_drop = max(
                        0.0,
                        float(base_smoke_summary["accuracy_micro"])
                        - float(smoke_summary["accuracy_micro"]),
                    )
                else:
                    smoke_drop = 0.0
                cap["smoke_acc_drop"] = smoke_drop
                (cdir / "cap_loss.json").write_text(
                    json.dumps(cap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                smoke_txt = (
                    f" | smoke {smoke_summary['accuracy_micro']:.3f} (drop {smoke_drop:.4f})"
                    if smoke_summary
                    else ""
                )
                print(
                    f"  cap_loss {cap['cap_loss']:.4f} | "
                    f"domain macro {domain_summary['accuracy_macro']:.3f}{smoke_txt}"
                )
            else:
                print("  capability: baseline ещё не посчитан — cap_loss отложен")

        runtime = time.time() - t0
        mean_r = ""
        if pref_metrics and cfg_id != BASELINE_ID and "primary" in pref_metrics:
            mean_r = pref_metrics["primary"]["mean"]
        elif pref_metrics and cfg_id == BASELINE_ID:
            mean_r = 0.0

        row = {
            "config_id": cfg_id,
            "role": cand["role"] if cand else "baseline",
            "basis": cand["basis"] if cand else "",
            "layer": cand["layer"] if cand else "",
            "rank": cand["rank"] if cand else "",
            "alpha": cand["alpha"] if cand else "",
            "mean_R_gender": mean_r,
            "flip_rate": (
                pref_metrics["flip_rate"]
                if pref_metrics and "flip_rate" in pref_metrics
                else ""
            ),
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

    if base_domain_summary is not None:
        for r in ranking:
            if r["config_id"] == BASELINE_ID or r["cap_loss"] != "":
                continue
            cdir = out_dir / r["config_id"]
            steered_sum = load_json(cdir / "capability_domain.json")
            cap = cap_loss_vs_baseline(steered_sum, base_domain_summary)
            smoke_path = cdir / "capability_smoke.json"
            if smoke_path.exists() and base_smoke_summary is not None:
                smoke_sum = load_json(smoke_path)
                smoke_drop = max(
                    0.0,
                    float(base_smoke_summary["accuracy_micro"]) - float(smoke_sum["accuracy_micro"]),
                )
            else:
                smoke_drop = 0.0
            cap["smoke_acc_drop"] = smoke_drop
            (cdir / "cap_loss.json").write_text(
                json.dumps(cap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            r["cap_loss"] = cap["cap_loss"]
            r["cap_loss_primary"] = cap["by_tier"].get("primary")
            r["smoke_acc_drop"] = smoke_drop
            r["cap_loss_flag"] = bool(float(cap["cap_loss"]) > args.cap_loss_max)

    if reselect_keep:
        ranking.sort(
            key=lambda r: (
                r["role"] != "candidate",
                float(r["cap_loss"]) if r["cap_loss"] != "" else 1e9,
                -(float(r["mean_R_gender"]) if r["mean_R_gender"] != "" else 0.0),
            )
        )
        keep = [
            r
            for r in ranking
            if r["role"] == "candidate" and r.get("cap_loss_flag") is not True
        ][:3]
        if len(keep) < 1:
            keep = [r for r in ranking if r["role"] == "candidate"][:3]
        keep_rule = "candidates with cap_loss ≤ max; sort by cap_loss then −R_gender"
    else:
        # Stage C: не переизбираем — только отчёт по shortlist order
        keep = [r for r in ranking if r["role"] == "candidate"]
        keep_rule = "Stage C report: all shortlist candidates (no re-selection)"

    with (out_dir / "ranking.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ranking[0].keys()))
        writer.writeheader()
        writer.writerows(ranking)

    (out_dir / "keep.json").write_text(
        json.dumps(
            {
                "n": len(keep),
                "cap_loss_max": args.cap_loss_max,
                "rule": keep_rule,
                "candidates": [
                    {
                        "config_id": r["config_id"],
                        "cap_loss": r["cap_loss"],
                        "mean_R_gender": r["mean_R_gender"],
                        "smoke_acc_drop": r["smoke_acc_drop"],
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
        "schema": schema,
        "hypothesis": "inlp",
        "tag": args.tag,
        "datetime": datetime.now().isoformat(),
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "subspaces": args.subspaces.name,
        "subspaces_sha256": sub_meta.get("arrays_sha256"),
        "preference": {
            "enabled": with_preference,
            "sample": args.sample.name if with_preference else None,
            "n_base_items": len(items),
            "n_rows": n_pref,
        },
        "capability": {
            "domain_profile": args.domain_profile.name if not args.skip_capability else None,
            "smoke_profile": (
                args.smoke_profile.name
                if not args.skip_capability and not skip_smoke
                else None
            ),
            "n_domain": len(domain_items),
            "n_smoke": len(smoke_items),
            "cap_loss_max": args.cap_loss_max,
            "skipped": args.skip_capability,
            "skip_smoke": skip_smoke,
        },
        "n_configs": len(configs),
        "config_ids": [c for c, _ in configs],
        "runtime_s": round(time.time() - t_all, 1),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "primary_metric": "cap_loss" if reselect_keep else "report",
        "preference_metric": "mean_R_gender",
        "keep": [k["config_id"] for k in keep],
        "reselect_keep": reselect_keep,
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\n=== {stage_label} [{args.tag}] готово за {meta['runtime_s']:.0f}s ===")
    print(f"  {out_dir}")
    print("  keep:" if reselect_keep else "  report:")
    for k in keep:
        print(
            f"    {k['config_id']}  cap_loss={k['cap_loss']}  "
            f"R_gender={k['mean_R_gender']}  smoke_drop={k['smoke_acc_drop']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
