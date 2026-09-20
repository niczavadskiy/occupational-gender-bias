"""
Stage A for XY-control additive steering.

Scores gendered prompts only (never XY). Primary metric is the layout-averaged
gender gap Δ_i = mean_layouts (z_man − z_woman).

    python -m steering.xy_control.run_stagea --device cuda --tag xy_2b_a_v1
    python -m steering.xy_control.run_stagea --limit-items 3 --tag smoke \\
        --candidates main_add_core__vraw__L16__add__a1
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
    ProjectionTrace,
    Scorer,
    load_model,
    num_hidden_layers,
    specs_for_candidate,
    steered,
)
from steering.run_h1_stagea import hook_check, row_gender_probs
from steering.run_inlp_stagea import behavioral_metrics, family_aggregate, load_json, row_margins
from steering.run_hs_recovery_auc import pick_items
from steering.xy_control.build_vectors import gender_prompt_of
from steering.xy_control.mapping import MAPPING

HERE = Path(__file__).resolve().parent
STEERING_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_ID = "baseline"

PATHS = {
    "2b": {
        "candidates": HERE / "candidates" / "xy_control_2b_candidates_v1.json",
        "vectors": HERE / "vectors" / "xy_control_2b_vectors_v1.npz",
        "sample": HERE / "data" / "xy_pairs_full_v1.json",
        "model": "Qwen/Qwen3.5-2B-Base",
    },
    "4b": {
        "candidates": HERE / "candidates" / "xy_control_4b_candidates_v1.json",
        "vectors": HERE / "vectors" / "xy_control_4b_vectors_v1.npz",
        "sample": HERE / "data" / "xy_pairs_full_v1.json",
        "model": "Qwen/Qwen3.5-4B-Base",
    },
}


def filter_items(items: list[dict], family_ids: set[int] | None) -> list[dict]:
    if family_ids is None:
        return items
    return [it for it in items if int(it["scenario_family_id"]) in family_ids]


def eval_family_ids(vec_meta: dict, items: list[dict], split: str) -> set[int] | None:
    present = {int(it["scenario_family_id"]) for it in items}
    train = set(int(x) for x in vec_meta.get("train_family_ids", []))
    val = set(int(x) for x in vec_meta.get("val_family_ids", []))
    test = set(int(x) for x in vec_meta.get("test_family_ids", []))
    if split == "all":
        return None
    if split == "train":
        return train & present
    if split == "val":
        return val & present
    if split == "test":
        return test & present
    if split == "heldout":
        held = (val | test) & present
        if held:
            return held
        if train:
            return present - train
        return present
    raise SystemExit(f"unknown --eval-split {split}")


def gap_block(fam: dict[int, dict]) -> dict:
    D = np.array([v["D_gender"] for v in fam.values()], dtype=np.float64)
    S = np.array([v["S_slot"] for v in fam.values()], dtype=np.float64)
    return {
        "gender_gap": float(D.mean()) if len(D) else float("nan"),
        "mean_abs_delta": float(np.mean(np.abs(D))) if len(D) else float("nan"),
        "p_delta_positive": float(np.mean(D > 0)) if len(D) else float("nan"),
        "slot_gap": float(S.mean()) if len(S) else float("nan"),
        "mean_abs_slot": float(np.mean(np.abs(S))) if len(S) else float("nan"),
        "n_families": int(len(D)),
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
            prompt = gender_prompt_of(src)
            with steered(model, specs, trace):
                out = scorer.score(prompt, list(src["valid_labels"]))

            labels = src["labels"]
            p_man, p_woman = row_gender_probs(out, labels)
            denom = p_man + p_woman
            chosen = labels.get(out["choice"])
            first = "man" if src["context_order"] == "man_first" else "woman"
            margins = row_margins(out, labels)
            row = {
                "id": src["id"],
                "scenario_family_id": item["scenario_family_id"],
                "soc_major_title": item["soc_major_title"],
                "position_variant": src["position_variant"],
                "context_order": src["context_order"],
                "labels": labels,
                "mapping": src.get("mapping", dict(MAPPING)),
                **out,
                **margins,
                "p_man": p_man,
                "p_woman": p_woman,
                "p_man_norm": p_man / denom if denom > 0 else float("nan"),
                "delta_logprob_vocab": float(
                    out.get(f"logprob_vocab_{_slot(labels, 'man')}", float("nan"))
                    - out.get(f"logprob_vocab_{_slot(labels, 'woman')}", float("nan"))
                ),
                "prefers_man": chosen == "man",
                "prefers_woman": chosen == "woman",
                "prefers_first_mentioned": chosen == first,
                "baseline_choice_recorded": src.get("baseline_choice"),
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
                rate = len(rows) / max(time.time() - t0, 1e-9)
                eta = (total - len(rows)) / max(rate, 1e-9) / 60
                print(f"    [{label}] {len(rows)}/{total}  {rate:.2f} row/s  ETA {eta:.1f} min", flush=True)
    return rows


def _slot(labels: dict[str, str], semantic: str) -> str:
    for slot, sem in labels.items():
        if sem == semantic:
            return slot
    raise KeyError(semantic)


def maybe_mmlu(scorer, model, specs, *, profile_name: str | None, parquet: Path | None, log_every: int) -> dict | None:
    if not profile_name or profile_name == "none":
        return None
    from steering.mmlu_eval import (
        expand_domain_profile,
        expand_overall_smoke,
        load_parquet_by_ids,
        run_mmlu,
        summarize_domain_capability,
        summarize_overall_smoke,
    )

    if profile_name == "smoke":
        profile_path = STEERING_DIR / "profiles" / "mmlu_pro_overall_smoke_v1.json"
    elif profile_name == "domain":
        profile_path = STEERING_DIR / "profiles" / "mmlu_pro_domain_val_v1.json"
    else:
        profile_path = Path(profile_name)
    if not profile_path.is_file():
        print(f"  SKIP MMLU: нет профиля {profile_path}")
        return None
    parquet = parquet or STEERING_DIR / ".cache" / "mmlu_pro_test.parquet"
    if not parquet.is_file():
        print(f"  SKIP MMLU: нет parquet {parquet}")
        return None
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile_name == "smoke" or profile.get("kind") == "overall_smoke":
        ids = {int(e["question_id"]) for e in profile["items"]}
        bank = load_parquet_by_ids(parquet, ids)
        items = expand_overall_smoke(profile, bank)
        rows = run_mmlu(scorer, model, items, specs, label="mmlu", log_every=log_every)
        return {"profile": profile_path.name, **summarize_overall_smoke(rows)}
    ids = {int(qid) for d in profile["domains"] for qid in d["question_ids"]}
    bank = load_parquet_by_ids(parquet, ids)
    items = expand_domain_profile(profile, bank)
    rows = run_mmlu(scorer, model, items, specs, label="mmlu", log_every=log_every)
    return {"profile": profile_path.name, **summarize_domain_capability(rows, profile)}


def main(argv: list[str] | None = None) -> int:
    raw = list(argv if argv is not None else sys.argv[1:])
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--scale", choices=["2b", "4b"], default="2b")
    known, rest = pre.parse_known_args(raw)
    defaults = PATHS[known.scale]

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", choices=["2b", "4b"], default=known.scale)
    ap.add_argument("--model", default=defaults["model"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--sample", type=Path, default=defaults["sample"])
    ap.add_argument("--candidates-file", type=Path, default=defaults["candidates"])
    ap.add_argument("--vectors", type=Path, default=defaults["vectors"])
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--roles", default=None)
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--limit-items", type=int, default=None)
    ap.add_argument(
        "--eval-split",
        choices=["heldout", "val", "test", "train", "all"],
        default="val",
        help="val: Stage A GenderGap after (default). test: after α is chosen. heldout=val+test.",
    )
    ap.add_argument("--mmlu", default="none", help="none | smoke | domain | path to profile json")
    ap.add_argument("--mmlu-parquet", type=Path, default=None)
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "results" / "steering" / "xy_control")
    ap.add_argument("--tag", default=f"xy_{known.scale}_a_v1")
    ap.add_argument("--log-every", type=int, default=40)
    args = ap.parse_args(rest)

    sample = load_json(args.sample)
    cand_doc = load_json(args.candidates_file)
    vec_meta = load_json(args.vectors.with_suffix(".json"))
    with np.load(args.vectors) as z:
        vectors = {k: np.asarray(z[k], dtype=np.float32) for k in z.files}
    calib = {e["key"]: e for e in vec_meta["vectors"]}

    fam_filter = eval_family_ids(vec_meta, sample["items"], args.eval_split)
    items = filter_items(sample["items"], fam_filter)
    items = pick_items(items, args.limit_items)
    if not items:
        raise SystemExit(f"пустое eval после --eval-split {args.eval_split}")

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
        f"XY Stage A [{args.tag}]: {len(configs)} конфиг. × {n_rows} строк "
        f"({len(items)} families, split={args.eval_split}) = {len(configs) * n_rows} forward"
    )
    if args.dtype != "float32":
        print(f"  ВНИМАНИЕ: dtype={args.dtype}; для add нужен float32")

    print(f"\n[1] Загрузка {args.model} ({args.dtype}, {args.device})...")
    t_load = time.time()
    model, tokenizer = load_model(args.model, dtype=args.dtype, device=args.device)
    scorer = Scorer(model, tokenizer)
    print(f"  готово за {time.time() - t_load:.1f}s, {num_hidden_layers(model.config)} блоков")

    t_all = time.time()
    ranking = []
    baseline_rows: list[dict] | None = None
    baseline_mmlu: dict | None = None

    for idx, (cfg_id, cand) in enumerate(configs, 1):
        specs = specs_for_candidate(cand, vectors, calib) if cand else []
        print(f"\n[{idx}/{len(configs)}] {cfg_id}")
        t0 = time.time()
        rows = run_config(scorer, model, items, specs, label=cfg_id, log_every=args.log_every)
        runtime = time.time() - t0
        fam = family_aggregate(rows)
        gaps = gap_block(fam)
        if baseline_rows is None:
            paired = None
        else:
            paired = behavioral_metrics(baseline_rows, rows, n_boot=1000, seed=0)
        mmlu = maybe_mmlu(
            scorer, model, specs, profile_name=args.mmlu, parquet=args.mmlu_parquet, log_every=args.log_every
        )
        if cfg_id == BASELINE_ID:
            baseline_rows = rows
            baseline_mmlu = mmlu

        metrics = {
            "config_id": cfg_id,
            "role": cand["role"] if cand else "baseline",
            "candidate": cand,
            "runtime_s": round(runtime, 1),
            "hook_check": hook_check(rows),
            "mapping": dict(MAPPING),
            "eval_split": args.eval_split,
            "n_items": len(items),
            "n_rows": len(rows),
            **gaps,
            "paired": paired,
            "mmlu": mmlu,
            "delta_mmlu": (
                None
                if mmlu is None or baseline_mmlu is None
                else float(mmlu.get("accuracy_micro", mmlu.get("accuracy_macro", float("nan"))))
                - float(baseline_mmlu.get("accuracy_micro", baseline_mmlu.get("accuracy_macro", float("nan"))))
            ),
        }

        cdir = out_dir / cfg_id
        cdir.mkdir(parents=True, exist_ok=True)
        with (cdir / "per_item.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (cdir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        gap_red = (
            ""
            if baseline_rows is None or cfg_id == BASELINE_ID
            else abs(ranking[0]["gender_gap"]) - abs(gaps["gender_gap"])
            if ranking
            else ""
        )
        print(
            f"  GenderGap {gaps['gender_gap']:+.4f}  mean|Δ| {gaps['mean_abs_delta']:.4f}  "
            f"P(Δ>0) {gaps['p_delta_positive']:.3f}  slot {gaps['slot_gap']:+.4f}  {runtime:.1f}s"
        )
        ranking.append(
            {
                "config_id": cfg_id,
                "role": metrics["role"],
                "family": cand["family"] if cand else "",
                "vector_id": cand["vector_id"] if cand else "",
                "layer": cand["layers"][0] if cand else "",
                "intervention": cand["intervention"] if cand else "",
                "alpha": cand.get("alpha") if cand else 0.0,
                "gender_gap": gaps["gender_gap"],
                "mean_abs_delta": gaps["mean_abs_delta"],
                "p_delta_positive": gaps["p_delta_positive"],
                "slot_gap": gaps["slot_gap"],
                "gap_abs_reduction": gap_red,
                "delta_mmlu": metrics["delta_mmlu"] if metrics["delta_mmlu"] is not None else "",
            }
        )

    base = next((r for r in ranking if r["config_id"] == BASELINE_ID), None)
    for r in ranking:
        if base is not None:
            r["gap_vs_baseline"] = r["gender_gap"] - base["gender_gap"]
            r["gap_abs_reduction"] = abs(base["gender_gap"]) - abs(r["gender_gap"])
        else:
            r["gap_vs_baseline"] = ""
    ranking.sort(key=lambda r: (r["role"] != "candidate", -float(r["gap_abs_reduction"] or 0)))

    with (out_dir / "ranking.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ranking[0].keys()))
        writer.writeheader()
        writer.writerows(ranking)

    meta = {
        "schema": "steering.xy_control_stage_a_run/v1",
        "hypothesis": "xy_control",
        "name": "gender-conditioned activation direction",
        "tag": args.tag,
        "datetime": datetime.now().isoformat(),
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "mapping": dict(MAPPING),
        "eval_split": args.eval_split,
        "sample": {
            "file": args.sample.name,
            "n_base_items": len(items),
            "n_rows": n_rows,
            "limited": args.limit_items is not None,
        },
        "candidates_file": args.candidates_file.name,
        "vectors_file": args.vectors.name,
        "vectors_sha256": vec_meta.get("arrays_sha256"),
        "n_configs": len(configs),
        "config_ids": [c for c, _ in configs],
        "runtime_s": round(time.time() - t_all, 1),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n=== XY Stage A [{args.tag}] готово за {meta['runtime_s']:.0f}s ===")
    print(f"  {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
