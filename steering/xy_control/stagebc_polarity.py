"""Stage B/C for pooled polarity XY-control (one unit per polarity set).

Stage B: MMLU-Pro domain_val union over member SOCs; freeze best / best-prior.
Stage C: preference on pooled test families + MMLU domain_test union.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from steering.intervene import Scorer, load_model, specs_for_candidate
from steering.mmlu_eval import collect_profile_ids, load_parquet_by_ids, run_mmlu
from steering.run_inlp_stagea import behavioral_metrics, family_aggregate
from steering.xy_control.mapping import MAPPING
from steering.xy_control.paired_sample import (
    FULL_SPLIT,
    dataset_provenance,
    load_items_file,
    split_path,
)
from steering.xy_control.polarity_pool import make_pool_candidate, mmlu_items_for_titles
from steering.xy_control.run_stagea import filter_items, gap_block, run_config
from steering.xy_control.stagebc import (
    MODEL_BY_SCALE,
    PROFILE_BY_STAGE,
    STEERING_DIR,
    as_bool,
    as_float,
    capability_summary,
    load_json,
    load_vectors,
    read_csv,
    write_csv,
    write_json,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_pool_vectors(stage_a_dir: Path, scale: str, override: Path | None) -> Path:
    if override is not None:
        path = override
    else:
        matches = sorted(stage_a_dir.glob(f"xy_control_{scale}_polarity_pool_vectors_*.npz"))
        if not matches:
            matches = sorted(
                (HERE / "vectors").glob(f"xy_control_{scale}_polarity_pool_vectors_*.npz")
            )
        if not matches:
            raise SystemExit(f"no polarity-pool vectors in {stage_a_dir}")
        path = matches[-1]
    if not path.is_file() or not path.with_suffix(".json").is_file():
        raise SystemExit(f"vectors or metadata missing: {path}")
    return path


def resolve_xy_dataset(stage_a_dir: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    staged = stage_a_dir / split_path(FULL_SPLIT).name
    return staged if staged.is_file() else split_path(FULL_SPLIT)


def load_pool_domain(stage_a_dir: Path) -> dict[str, Any]:
    set_path = stage_a_dir / "set.json"
    if not set_path.is_file():
        raise SystemExit(f"missing {set_path}")
    doc = load_json(set_path)
    return doc["domain"]


def make_shortlist(stage_a_dir: Path, domain: dict[str, Any]) -> dict:
    summary_path = stage_a_dir / "summary.csv"
    if not summary_path.is_file():
        raise SystemExit(f"Stage A summary missing: {summary_path}")
    rows = read_csv(summary_path)
    if not rows:
        raise SystemExit(f"empty Stage A summary: {summary_path}")
    row = rows[0]
    slug = row["slug"]
    ranking_path = stage_a_dir / slug / "ranking.csv"
    ranking = {r["config_id"]: r for r in read_csv(ranking_path)}
    baseline = ranking.get("baseline", {})
    ids = []
    for key in ("best_id", "best_prior_id"):
        cid = row.get(key, "")
        if cid and cid not in ids:
            ids.append(cid)
    candidates = []
    for cid in ids:
        if cid not in ranking:
            raise SystemExit(f"{slug}: {cid} absent from Stage A ranking")
        src = ranking[cid]
        if float(src["gap_abs_reduction"]) <= 0:
            continue
        candidates.append(
            {
                "config_id": cid,
                "layer": int(float(src["layer"])),
                "alpha": float(src["alpha"]),
                "alpha_matches_prior": as_bool(src.get("alpha_matches_prior")),
                "stage_a_gender_gap": float(src["gender_gap"]),
                "stage_a_gap_abs_reduction": float(src["gap_abs_reduction"]),
                "stage_a_slot_gap": float(src["slot_gap"]),
            }
        )
    return {
        "schema": "steering.xy_control_polarity_pool_stageb_shortlist/v1",
        "scale": row.get("scale"),
        "set_id": row.get("set_id") or domain.get("set_id"),
        "source_stage_a": str(stage_a_dir),
        "selection": "unique(best_id, best_prior_id) from pooled Stage A val",
        "domain": {
            "slug": slug,
            "soc_major_title": domain["soc_major_title"],
            "polarity": domain["polarity"],
            "member_titles": domain["member_titles"],
            "stage_a_baseline_gender_gap": float(row["baseline_gender_gap"]),
            "stage_a_baseline_slot_gap": as_float(baseline.get("slot_gap")),
            "candidates": candidates,
        },
    }


def candidate_from_entry(domain: dict, entry: dict) -> dict:
    cand = make_pool_candidate(domain, layer=int(entry["layer"]), alpha=float(entry["alpha"]))
    if cand["id"] != entry["config_id"]:
        raise ValueError(f"candidate mismatch: {cand['id']} != {entry['config_id']}")
    return cand


def select_keep(entry: dict, domain_rows: list[dict], cap_loss_max: float) -> dict:
    passed = [r for r in domain_rows if r["cap_loss"] <= cap_loss_max]
    if passed:
        winner = min(
            passed,
            key=lambda r: (
                r["cap_loss"],
                not r["alpha_matches_prior"],
                -r["stage_a_gap_abs_reduction"],
            ),
        )
        return {
            "slug": entry["slug"],
            "soc_major_title": entry["soc_major_title"],
            "polarity": entry["polarity"],
            "decision": "steer",
            "reason": "capability_pass",
            "config_id": winner["config_id"],
            "layer": winner["layer"],
            "alpha": winner["alpha"],
            "alpha_matches_prior": winner["alpha_matches_prior"],
            "stage_a_gap_abs_reduction": winner["stage_a_gap_abs_reduction"],
            "stage_a_slot_gap": winner["stage_a_slot_gap"],
            "stage_b_cap_loss": winner["cap_loss"],
            "member_titles": entry.get("member_titles"),
        }
    return {
        "slug": entry["slug"],
        "soc_major_title": entry["soc_major_title"],
        "polarity": entry["polarity"],
        "decision": "identity",
        "reason": f"all candidates cap_loss > {cap_loss_max}",
        "config_id": "baseline",
        "member_titles": entry.get("member_titles"),
    }


def stage_b(args: argparse.Namespace) -> Path:
    domain = load_pool_domain(args.stage_a_dir)
    shortlist = make_shortlist(args.stage_a_dir, domain)
    entry = shortlist["domain"]
    vectors_path = resolve_pool_vectors(args.stage_a_dir, args.scale, args.vectors)
    vectors, calib, _ = load_vectors(vectors_path)
    profile = load_json(args.domain_profile)
    bank = load_parquet_by_ids(args.mmlu_parquet, collect_profile_ids(profile))
    items = mmlu_items_for_titles(profile, bank, domain["member_titles"])
    if args.limit_mmlu is not None:
        items = items[: args.limit_mmlu]

    out_dir = args.out_root / "stage_b" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "stageb_shortlist.json", shortlist)

    print(f"load {args.model} for polarity-pool Stage B...")
    model, tokenizer = load_model(args.model, dtype=args.dtype, device=args.device)
    scorer = Scorer(model, tokenizer)
    slug = entry["slug"]
    t0 = time.time()
    print(
        f"[B] {domain.get('label', domain['soc_major_title'])}: {len(items)} MMLU val "
        f"({len(domain['member_titles'])} SOCs union)"
    )
    baseline_rows = run_mmlu(
        scorer, model, items, [], label=f"{slug}:baseline", log_every=args.log_every
    )
    base_acc = capability_summary(baseline_rows)["accuracy"]
    baseline_dir = out_dir / slug / "baseline"
    write_json(
        baseline_dir / "metrics.json",
        {"config_id": "baseline", "n_mmlu": len(items), "mmlu_accuracy": base_acc},
    )
    with (baseline_dir / "mmlu_per_item.jsonl").open("w", encoding="utf-8") as f:
        for row in baseline_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    domain_rows = []
    for cand_entry in entry["candidates"]:
        candidate = candidate_from_entry(domain, cand_entry)
        specs = specs_for_candidate(candidate, vectors, calib)
        rows = run_mmlu(
            scorer, model, items, specs, label=candidate["id"], log_every=args.log_every
        )
        acc = capability_summary(rows)["accuracy"]
        cap_loss = max(0.0, float(base_acc) - float(acc))
        result = {
            **cand_entry,
            "scale": args.scale,
            "set_id": domain.get("set_id"),
            "slug": slug,
            "soc_major_title": domain["soc_major_title"],
            "mmlu_profile": "val_union",
            "n_mmlu": len(items),
            "mmlu_accuracy_baseline": base_acc,
            "mmlu_accuracy": acc,
            "cap_loss": cap_loss,
            "cap_loss_max": args.cap_loss_max,
            "capability_pass": cap_loss <= args.cap_loss_max,
        }
        domain_rows.append(result)
        cdir = out_dir / slug / candidate["id"]
        write_json(cdir / "metrics.json", result)
        with (cdir / "mmlu_per_item.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {candidate['id']}: acc={acc:.3f}, loss={cap_loss:.3f}")

    keep = select_keep(entry, domain_rows, args.cap_loss_max)
    stagec_keep = {
        "schema": "steering.xy_control_polarity_pool_stagec_keep/v1",
        "scale": args.scale,
        "set_id": domain.get("set_id"),
        "source_stage_a": str(args.stage_a_dir),
        "source_stage_b": str(out_dir),
        "vectors": str(vectors_path),
        "cap_loss_max": args.cap_loss_max,
        "rule": "min cap_loss among capability-pass; prior wins ties; identity if none pass",
        "domains": [keep],
    }
    write_json(out_dir / "stagec_keep.json", stagec_keep)
    write_csv(out_dir / "ranking.csv", domain_rows)
    write_csv(out_dir / "summary.csv", [keep])
    write_csv(out_dir / slug / "ranking.csv", sorted(domain_rows, key=lambda r: r["cap_loss"]))
    shutil.copy2(vectors_path, out_dir / vectors_path.name)
    shutil.copy2(vectors_path.with_suffix(".json"), out_dir / vectors_path.with_suffix(".json").name)
    write_json(
        out_dir / "run_meta.json",
        {
            "schema": "steering.xy_control_polarity_pool_stage_b/v1",
            "datetime": datetime.now().isoformat(),
            "scale": args.scale,
            "set_id": domain.get("set_id"),
            "model": args.model,
            "profile": str(args.domain_profile),
            "protocol": "pooled_polarity_one_v",
            "runtime_s": round(time.time() - t0, 1),
        },
    )
    print(f"Stage B complete → {out_dir}  decision={keep['decision']}")
    return out_dir


def stage_c(args: argparse.Namespace) -> Path:
    keep = load_json(args.stagec_keep)
    if keep["scale"] != args.scale:
        raise SystemExit(f"keep scale {keep['scale']} != --scale {args.scale}")
    domain = load_pool_domain(args.stage_a_dir)
    entry = keep["domains"][0]
    vectors_path = resolve_pool_vectors(args.stage_a_dir, args.scale, args.vectors)
    vectors, calib, vec_meta = load_vectors(vectors_path)
    split_meta = {d["slug"]: d for d in vec_meta["domains"]}
    dataset_path = resolve_xy_dataset(args.stage_a_dir, args.xy_data)
    if not dataset_path.is_file():
        raise SystemExit(f"XY dataset missing: {dataset_path}")
    pooled = load_items_file(dataset_path)
    provenance = dataset_provenance(dataset_path)
    expected_sha = vec_meta.get("xy_dataset_sha256")
    if expected_sha and provenance["xy_dataset_sha256"] != expected_sha:
        raise SystemExit(
            f"dataset SHA mismatch: vectors={expected_sha} current={provenance['xy_dataset_sha256']}"
        )
    smeta = split_meta[entry["slug"]]
    test_ids = set(int(x) for x in smeta["test_family_ids"])
    # pooled test: any family in the frozen test id set
    pref_items = filter_items(pooled, test_ids)
    if args.limit_items is not None:
        pref_items = pref_items[: args.limit_items]
    missing = test_ids - {int(it["scenario_family_id"]) for it in pref_items}
    if missing and args.limit_items is None:
        raise SystemExit(f"Stage C missing {len(missing)} pooled test families")

    profile = load_json(args.domain_profile)
    bank = load_parquet_by_ids(args.mmlu_parquet, collect_profile_ids(profile))
    mmlu_items = mmlu_items_for_titles(profile, bank, domain["member_titles"])
    if args.limit_mmlu is not None:
        mmlu_items = mmlu_items[: args.limit_mmlu]

    out_dir = args.out_root / "stage_c" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"load {args.model} for polarity-pool Stage C...")
    model, tokenizer = load_model(args.model, dtype=args.dtype, device=args.device)
    scorer = Scorer(model, tokenizer)
    t0 = time.time()
    slug = entry["slug"]
    print(
        f"[C] {domain.get('label', domain['soc_major_title'])}: {len(pref_items)} test families, "
        f"{len(mmlu_items)} MMLU test"
    )
    baseline_pref = run_config(
        scorer, model, pref_items, [], label="baseline", log_every=args.log_every
    )
    baseline_gap = gap_block(family_aggregate(baseline_pref))
    baseline_mmlu_rows = run_mmlu(
        scorer, model, mmlu_items, [], label=f"{slug}:baseline", log_every=args.log_every
    )
    baseline_acc = capability_summary(baseline_mmlu_rows)["accuracy"]

    if entry["decision"] == "steer":
        candidate = candidate_from_entry(domain, entry)
        specs = specs_for_candidate(candidate, vectors, calib)
        steered_pref = run_config(
            scorer, model, pref_items, specs, label=candidate["id"], log_every=args.log_every
        )
        steered_mmlu_rows = run_mmlu(
            scorer, model, mmlu_items, specs, label=candidate["id"], log_every=args.log_every
        )
        after_gap = gap_block(family_aggregate(steered_pref))
        after_acc = capability_summary(steered_mmlu_rows)["accuracy"]
        paired = behavioral_metrics(baseline_pref, steered_pref, n_boot=5000, seed=20260926)
    else:
        steered_pref = baseline_pref
        steered_mmlu_rows = baseline_mmlu_rows
        after_gap = baseline_gap
        after_acc = baseline_acc
        paired = None

    result = {
        "scale": args.scale,
        "set_id": domain.get("set_id"),
        "slug": slug,
        "soc_major_title": domain["soc_major_title"],
        "polarity": entry["polarity"],
        "decision": entry["decision"],
        "config_id": entry["config_id"],
        "n_test_families": len(pref_items),
        "baseline_gender_gap": baseline_gap["gender_gap"],
        "test_gender_gap": after_gap["gender_gap"],
        "test_gap_abs_reduction": abs(baseline_gap["gender_gap"]) - abs(after_gap["gender_gap"]),
        "test_mean_abs_delta": after_gap["mean_abs_delta"],
        "test_p_delta_positive": after_gap["p_delta_positive"],
        "test_slot_gap": after_gap["slot_gap"],
        "n_mmlu_test": len(mmlu_items),
        "mmlu_test_accuracy_baseline": baseline_acc,
        "mmlu_test_accuracy": after_acc,
        "mmlu_test_cap_loss": max(0.0, float(baseline_acc) - float(after_acc)),
        "paired": paired,
        "protocol": "pooled_polarity_one_v",
    }
    ddir = out_dir / slug
    write_json(ddir / "metrics.json", result)
    with (ddir / "baseline_preference_per_item.jsonl").open("w", encoding="utf-8") as f:
        for row in baseline_pref:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (ddir / "preference_per_item.jsonl").open("w", encoding="utf-8") as f:
        for row in steered_pref:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (ddir / "baseline_mmlu_per_item.jsonl").open("w", encoding="utf-8") as f:
        for row in baseline_mmlu_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (ddir / "mmlu_per_item.jsonl").open("w", encoding="utf-8") as f:
        for row in steered_mmlu_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(out_dir / "summary.csv", [result])
    write_json(out_dir / "stagec_keep.json", keep)
    shutil.copy2(vectors_path, out_dir / vectors_path.name)
    shutil.copy2(vectors_path.with_suffix(".json"), out_dir / vectors_path.with_suffix(".json").name)
    shutil.copy2(dataset_path, out_dir / dataset_path.name)
    write_json(
        out_dir / "run_meta.json",
        {
            "schema": "steering.xy_control_polarity_pool_stage_c/v1",
            "datetime": datetime.now().isoformat(),
            "scale": args.scale,
            "set_id": domain.get("set_id"),
            "model": args.model,
            "profile": str(args.domain_profile),
            "xy_dataset": str(dataset_path),
            **provenance,
            "preference_split": "pooled_test",
            "protocol": "pooled_polarity_one_v",
            "runtime_s": round(time.time() - t0, 1),
            "mapping": dict(MAPPING),
        },
    )
    print(
        f"  gap {baseline_gap['gender_gap']:+.4f} → {after_gap['gender_gap']:+.4f}; "
        f"MMLU {baseline_acc:.3f} → {after_acc:.3f}"
    )
    print(f"Stage C complete → {out_dir}")
    return out_dir


def parser(stage: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=f"Pooled polarity XY-control Stage {stage.upper()}")
    ap.add_argument("--scale", choices=["2b", "4b"], required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--stage-a-dir", type=Path, required=True)
    ap.add_argument("--vectors", type=Path, default=None)
    ap.add_argument("--domain-profile", type=Path, default=PROFILE_BY_STAGE[stage])
    ap.add_argument(
        "--mmlu-parquet",
        type=Path,
        default=STEERING_DIR / ".cache" / "mmlu_pro_test.parquet",
    )
    ap.add_argument("--limit-mmlu", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument(
        "--out-root",
        type=Path,
        default=REPO_ROOT / "results" / "steering" / "xy_control",
    )
    ap.add_argument("--tag", default=None)
    if stage == "b":
        ap.add_argument("--cap-loss-max", type=float, default=0.03)
    else:
        ap.add_argument("--stagec-keep", type=Path, required=True)
        ap.add_argument("--xy-data", type=Path, default=None)
        ap.add_argument("--limit-items", type=int, default=None)
    return ap


def main(stage: str, argv: list[str] | None = None) -> int:
    args = parser(stage).parse_args(argv)
    args.model = args.model or MODEL_BY_SCALE[args.scale]
    args.tag = args.tag or f"xy_{args.scale}_polarity_pool_{stage}_v1"
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    if not args.mmlu_parquet.is_file():
        raise SystemExit(f"MMLU-Pro parquet missing: {args.mmlu_parquet}")
    if stage == "b":
        stage_b(args)
    else:
        stage_c(args)
    return 0
