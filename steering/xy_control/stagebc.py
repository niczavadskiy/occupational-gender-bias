"""Shared Stage B/C runner for per-SOC XY-control.

Stage B:
  - reads Stage A val rankings;
  - evaluates best + best-prior candidate for each SOC on MMLU-Pro domain_val;
  - freezes one candidate (or identity) in stagec_keep.json.

Stage C:
  - reads stagec_keep.json without re-selection;
  - reports GenderGap on the SOC's test families;
  - reports capability on disjoint MMLU-Pro domain_test questions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from steering.intervene import Scorer, load_model, specs_for_candidate
from steering.mmlu_eval import (
    collect_profile_ids,
    expand_domain_profile,
    load_parquet_by_ids,
    run_mmlu,
)
from steering.run_inlp_stagea import behavioral_metrics, family_aggregate
from steering.xy_control.domains import CATALOG_JSON, load_catalog
from steering.xy_control.mapping import MAPPING
from steering.xy_control.paired_sample import (
    FULL_SPLIT,
    dataset_provenance,
    filter_soc,
    load_items_file,
    split_path,
)
from steering.xy_control.per_soc import eval_ids_for_split, make_candidate
from steering.xy_control.run_stagea import filter_items, gap_block, run_config

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
STEERING_DIR = HERE.parent
MODEL_BY_SCALE = {
    "2b": "Qwen/Qwen3.5-2B-Base",
    "4b": "Qwen/Qwen3.5-4B-Base",
}
PROFILE_BY_STAGE = {
    "b": STEERING_DIR / "profiles" / "mmlu_pro_domain_val_v1.json",
    "c": STEERING_DIR / "profiles" / "mmlu_pro_domain_test_v1.json",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def as_bool(raw: Any) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes"}


def domain_map(catalog: dict, scale: str) -> dict[str, dict]:
    return {d["slug"]: d for d in catalog["scales"][scale]["steer"]}


def resolve_vectors(stage_a_dir: Path, scale: str, override: Path | None) -> Path:
    if override is not None:
        path = override
    else:
        matches = sorted(stage_a_dir.glob(f"xy_control_{scale}_per_soc_vectors_*.npz"))
        if not matches:
            matches = sorted((HERE / "vectors").glob(f"xy_control_{scale}_per_soc_vectors_*.npz"))
        if not matches:
            raise SystemExit(
                f"no {scale} per-SOC vectors in {stage_a_dir} or {HERE / 'vectors'}"
            )
        path = matches[-1]
    if not path.is_file() or not path.with_suffix(".json").is_file():
        raise SystemExit(f"vectors or metadata missing: {path}")
    return path


def ranking_by_id(stage_a_dir: Path, slug: str) -> dict[str, dict[str, str]]:
    path = stage_a_dir / slug / "ranking.csv"
    if not path.is_file():
        raise SystemExit(f"Stage A ranking missing: {path}")
    return {r["config_id"]: r for r in read_csv(path)}


def make_stageb_shortlist(stage_a_dir: Path, catalog: dict, scale: str) -> dict:
    """Freeze best and best-prior from Stage A; no new preference scoring in B."""
    summary_path = stage_a_dir / "summary.csv"
    if not summary_path.is_file():
        raise SystemExit(f"Stage A summary missing: {summary_path}")
    domains = domain_map(catalog, scale)
    entries = []
    for row in read_csv(summary_path):
        slug = row["slug"]
        if slug not in domains:
            continue
        ranking = ranking_by_id(stage_a_dir, slug)
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
                    "layer": int(src["layer"]),
                    "alpha": float(src["alpha"]),
                    "alpha_matches_prior": as_bool(src.get("alpha_matches_prior")),
                    "stage_a_gender_gap": float(src["gender_gap"]),
                    "stage_a_gap_abs_reduction": float(src["gap_abs_reduction"]),
                    "stage_a_slot_gap": float(src["slot_gap"]),
                }
            )
        entries.append(
            {
                "slug": slug,
                "soc_major_title": row["soc_major_title"],
                "polarity": row["polarity"],
                "stage_a_baseline_gender_gap": float(row["baseline_gender_gap"]),
                "stage_a_baseline_slot_gap": as_float(baseline.get("slot_gap")),
                "candidates": candidates,
            }
        )
    if not entries:
        raise SystemExit(f"empty Stage B shortlist from {summary_path}")
    return {
        "schema": "steering.xy_control_per_soc_stageb_shortlist/v1",
        "scale": scale,
        "source_stage_a": str(stage_a_dir),
        "selection": "unique(best_id, best_prior_id) per SOC from Stage A val",
        "domains": entries,
    }


def load_vectors(path: Path) -> tuple[dict[str, np.ndarray], dict[str, dict], dict]:
    meta = load_json(path.with_suffix(".json"))
    with np.load(path) as z:
        vectors = {key: np.asarray(z[key], dtype=np.float32) for key in z.files}
    calib = {entry["key"]: entry for entry in meta["vectors"]}
    return vectors, calib, meta


def resolve_xy_dataset(stage_a_dir: Path, override: Path | None) -> Path:
    if override is not None:
        path = override
    else:
        staged = stage_a_dir / split_path(FULL_SPLIT).name
        path = staged if staged.is_file() else split_path(FULL_SPLIT)
    if not path.is_file():
        raise SystemExit(f"XY preference dataset missing: {path}")
    return path


def validate_stagec_preference_pool(
    pooled: list[dict],
    vec_meta: dict,
    keep_domains: list[dict],
    dataset_path: Path,
) -> dict:
    """Require the exact Stage A dataset and every frozen Stage C test ID."""
    provenance = dataset_provenance(dataset_path)
    expected_sha = vec_meta.get("xy_dataset_sha256")
    if not expected_sha:
        raise SystemExit(
            "vector metadata has no xy_dataset_sha256; rerun Stage A so Stage C "
            "can prove it uses the same preference dataset"
        )

    problems: list[str] = []
    if provenance["xy_dataset_sha256"] != expected_sha:
        problems.append(
            "dataset SHA-256 differs: "
            f"vectors={expected_sha}, current={provenance['xy_dataset_sha256']}"
        )
    expected_ids_sha = vec_meta.get("xy_dataset_family_ids_sha256")
    if expected_ids_sha and provenance["xy_dataset_family_ids_sha256"] != expected_ids_sha:
        problems.append(
            "family-ID SHA-256 differs: "
            f"vectors={expected_ids_sha}, current={provenance['xy_dataset_family_ids_sha256']}"
        )

    split_meta = {d["slug"]: d for d in vec_meta.get("domains", [])}
    all_locations = {
        int(item["scenario_family_id"]): str(item.get("soc_major_title") or "")
        for item in pooled
    }
    for entry in keep_domains:
        slug = entry["slug"]
        title = entry["soc_major_title"]
        if slug not in split_meta:
            problems.append(f"{slug}: split metadata missing")
            continue
        expected = {int(x) for x in split_meta[slug]["test_family_ids"]}
        present = {
            int(item["scenario_family_id"])
            for item in pooled
            if item.get("soc_major_title") == title
        }
        missing = sorted(expected - present)
        if missing:
            wrong_soc = [fid for fid in missing if fid in all_locations]
            sample = ", ".join(str(fid) for fid in missing[:8])
            detail = f"{slug}: missing {len(missing)}/{len(expected)} test IDs ({sample})"
            if wrong_soc:
                detail += f"; {len(wrong_soc)} IDs exist under another SOC"
            problems.append(detail)

    if problems:
        raise SystemExit(
            "Stage C preference holdout validation failed before model loading.\n"
            f"Dataset: {dataset_path}\n  - "
            + "\n  - ".join(problems)
            + "\nUse the xy_pairs_full_v1.json copied into the matching Stage A directory, "
            "or rerun Stage A with the current frozen dataset."
        )
    return provenance


def candidate_from_entry(domain: dict, entry: dict) -> dict:
    candidate = make_candidate(
        domain,
        layer=int(entry["layer"]),
        alpha=float(entry["alpha"]),
    )
    if candidate["id"] != entry["config_id"]:
        raise ValueError(f"candidate mismatch: {candidate['id']} != {entry['config_id']}")
    return candidate


def profile_items_for_soc(profile: dict, bank: dict[int, dict], title: str) -> list[dict]:
    scoped = {**profile, "domains": [d for d in profile["domains"] if d["soc_major_title"] == title]}
    if not scoped["domains"]:
        raise ValueError(f"MMLU profile has no domain {title}")
    return expand_domain_profile(scoped, bank)


def capability_summary(rows: list[dict]) -> dict:
    return {
        "n_questions": len(rows),
        "accuracy": float(np.mean([r["correct"] for r in rows])) if rows else None,
    }


def select_stagec_entry(entry: dict, domain_rows: list[dict], cap_loss_max: float) -> dict:
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
        }
    return {
        "slug": entry["slug"],
        "soc_major_title": entry["soc_major_title"],
        "polarity": entry["polarity"],
        "decision": "identity",
        "reason": f"all candidates cap_loss > {cap_loss_max}",
        "config_id": "baseline",
    }


def stage_b(args: argparse.Namespace) -> Path:
    catalog = load_catalog(args.catalog)
    domains = domain_map(catalog, args.scale)
    shortlist = make_stageb_shortlist(args.stage_a_dir, catalog, args.scale)
    vectors_path = resolve_vectors(args.stage_a_dir, args.scale, args.vectors)
    vectors, calib, _ = load_vectors(vectors_path)
    profile = load_json(args.domain_profile)
    bank = load_parquet_by_ids(args.mmlu_parquet, collect_profile_ids(profile))

    out_dir = args.out_root / "stage_b" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "stageb_shortlist.json", shortlist)

    print(f"load {args.model} for XY-control Stage B...")
    model, tokenizer = load_model(args.model, dtype=args.dtype, device=args.device)
    scorer = Scorer(model, tokenizer)
    ranking_all: list[dict] = []
    keep_domains: list[dict] = []
    t0 = time.time()

    for idx, entry in enumerate(shortlist["domains"], 1):
        slug = entry["slug"]
        title = entry["soc_major_title"]
        domain = domains[slug]
        items = profile_items_for_soc(profile, bank, title)
        if args.limit_mmlu is not None:
            items = items[: args.limit_mmlu]
        print(f"[B {idx}/{len(shortlist['domains'])}] {title}: {len(items)} MMLU val")
        baseline_rows = run_mmlu(
            scorer, model, items, [], label=f"{slug}:baseline", log_every=args.log_every
        )
        base_acc = capability_summary(baseline_rows)["accuracy"]
        baseline_dir = out_dir / slug / "baseline"
        write_json(
            baseline_dir / "metrics.json",
            {
                "config_id": "baseline",
                "soc_major_title": title,
                "n_mmlu": len(items),
                "mmlu_accuracy": base_acc,
            },
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
                "slug": slug,
                "soc_major_title": title,
                "mmlu_profile": "val",
                "n_mmlu": len(items),
                "mmlu_accuracy_baseline": base_acc,
                "mmlu_accuracy": acc,
                "cap_loss": cap_loss,
                "cap_loss_max": args.cap_loss_max,
                "capability_pass": cap_loss <= args.cap_loss_max,
            }
            domain_rows.append(result)
            ranking_all.append(result)
            cdir = out_dir / slug / candidate["id"]
            write_json(cdir / "metrics.json", result)
            with (cdir / "mmlu_per_item.jsonl").open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  {candidate['id']}: acc={acc:.3f}, loss={cap_loss:.3f}")

        keep = select_stagec_entry(entry, domain_rows, args.cap_loss_max)
        keep_domains.append(keep)
        write_csv(out_dir / slug / "ranking.csv", sorted(domain_rows, key=lambda r: r["cap_loss"]))

    stagec_keep = {
        "schema": "steering.xy_control_per_soc_stagec_keep/v1",
        "scale": args.scale,
        "source_stage_a": str(args.stage_a_dir),
        "source_stage_b": str(out_dir),
        "vectors": str(vectors_path),
        "cap_loss_max": args.cap_loss_max,
        "rule": "min cap_loss among capability-pass; prior wins ties; identity if none pass",
        "domains": keep_domains,
    }
    write_json(out_dir / "stagec_keep.json", stagec_keep)
    write_csv(out_dir / "ranking.csv", ranking_all)
    write_csv(out_dir / "summary.csv", keep_domains)
    shutil.copy2(vectors_path, out_dir / vectors_path.name)
    shutil.copy2(vectors_path.with_suffix(".json"), out_dir / vectors_path.with_suffix(".json").name)
    write_json(
        out_dir / "run_meta.json",
        {
            "schema": "steering.xy_control_per_soc_stage_b/v1",
            "datetime": datetime.now().isoformat(),
            "scale": args.scale,
            "model": args.model,
            "profile": str(args.domain_profile),
            "split": "MMLU-Pro domain_val; Stage A preference remains frozen",
            "n_domains": len(keep_domains),
            "n_steer": sum(d["decision"] == "steer" for d in keep_domains),
            "n_identity": sum(d["decision"] == "identity" for d in keep_domains),
            "runtime_s": round(time.time() - t0, 1),
        },
    )
    print(f"Stage B complete → {out_dir}")
    return out_dir


def stage_c(args: argparse.Namespace) -> Path:
    keep = load_json(args.stagec_keep)
    if keep["scale"] != args.scale:
        raise SystemExit(f"keep scale {keep['scale']} != --scale {args.scale}")
    catalog = load_catalog(args.catalog)
    domains = domain_map(catalog, args.scale)
    vectors_path = resolve_vectors(args.stage_a_dir, args.scale, args.vectors)
    vectors, calib, vec_meta = load_vectors(vectors_path)
    split_meta = {d["slug"]: d for d in vec_meta["domains"]}
    dataset_path = resolve_xy_dataset(args.stage_a_dir, args.xy_data)
    pooled = load_items_file(dataset_path)
    dataset_meta = validate_stagec_preference_pool(
        pooled, vec_meta, keep["domains"], dataset_path
    )
    profile = load_json(args.domain_profile)
    bank = load_parquet_by_ids(args.mmlu_parquet, collect_profile_ids(profile))

    out_dir = args.out_root / "stage_c" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"load {args.model} for XY-control Stage C...")
    model, tokenizer = load_model(args.model, dtype=args.dtype, device=args.device)
    scorer = Scorer(model, tokenizer)
    summary: list[dict] = []
    t0 = time.time()

    for idx, entry in enumerate(keep["domains"], 1):
        slug = entry["slug"]
        title = entry["soc_major_title"]
        domain = domains[slug]
        smeta = split_meta[slug]
        test_ids = eval_ids_for_split(
            set(smeta["train_family_ids"]),
            set(smeta["val_family_ids"]),
            set(smeta["test_family_ids"]),
            "test",
        )
        pref_items = filter_items(filter_soc(pooled, title), test_ids)
        if args.limit_items is not None:
            pref_items = pref_items[: args.limit_items]
        mmlu_items = profile_items_for_soc(profile, bank, title)
        if args.limit_mmlu is not None:
            mmlu_items = mmlu_items[: args.limit_mmlu]
        print(
            f"[C {idx}/{len(keep['domains'])}] {title}: "
            f"{len(pref_items)} test families, {len(mmlu_items)} MMLU test"
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
            paired = behavioral_metrics(baseline_pref, steered_pref, n_boot=5000, seed=20260820)
        else:
            candidate = None
            steered_pref = baseline_pref
            steered_mmlu_rows = baseline_mmlu_rows
            after_gap = baseline_gap
            after_acc = baseline_acc
            paired = None

        result = {
            "scale": args.scale,
            "slug": slug,
            "soc_major_title": title,
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
        }
        summary.append(result)
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
        print(
            f"  gap {baseline_gap['gender_gap']:+.4f} → {after_gap['gender_gap']:+.4f}; "
            f"MMLU {baseline_acc:.3f} → {after_acc:.3f}"
        )

    write_csv(out_dir / "summary.csv", summary)
    write_json(out_dir / "stagec_keep.json", keep)
    shutil.copy2(vectors_path, out_dir / vectors_path.name)
    shutil.copy2(vectors_path.with_suffix(".json"), out_dir / vectors_path.with_suffix(".json").name)
    shutil.copy2(dataset_path, out_dir / dataset_path.name)
    write_json(
        out_dir / "run_meta.json",
        {
            "schema": "steering.xy_control_per_soc_stage_c/v1",
            "datetime": datetime.now().isoformat(),
            "scale": args.scale,
            "model": args.model,
            "profile": str(args.domain_profile),
            "xy_dataset": str(dataset_path),
            **dataset_meta,
            "preference_split": "test",
            "selection": "none; frozen Stage B keep",
            "n_domains": len(summary),
            "runtime_s": round(time.time() - t0, 1),
            "mapping": dict(MAPPING),
        },
    )
    print(f"Stage C complete → {out_dir}")
    return out_dir


def parser(stage: str) -> argparse.ArgumentParser:
    label = stage.upper()
    ap = argparse.ArgumentParser(description=f"Per-SOC XY-control Stage {label}")
    ap.add_argument("--scale", choices=["2b", "4b"], required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--stage-a-dir", type=Path, required=True)
    ap.add_argument("--vectors", type=Path, default=None)
    ap.add_argument("--catalog", type=Path, default=CATALOG_JSON)
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
        ap.add_argument(
            "--xy-data",
            type=Path,
            default=None,
            help="Exact Stage A xy_pairs_full_v1.json; defaults to the copy in --stage-a-dir.",
        )
        ap.add_argument("--limit-items", type=int, default=None)
    return ap


def main(stage: str, argv: list[str] | None = None) -> int:
    args = parser(stage).parse_args(argv)
    args.model = args.model or MODEL_BY_SCALE[args.scale]
    args.tag = args.tag or f"xy_{args.scale}_per_soc_{stage}_v1"
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    if not args.mmlu_parquet.is_file():
        raise SystemExit(
            f"MMLU-Pro parquet missing: {args.mmlu_parquet}\n"
            "run the Vast B/C script or download test-00000-of-00001.parquet"
        )
    if stage == "b":
        stage_b(args)
    else:
        stage_c(args)
    return 0
