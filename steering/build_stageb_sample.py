"""
Заморозка выборки Stage B (preference): остаток val после Stage A.

Stage A взял 95 из 190 val-семей (`h1_stagea_sample_v1.json`). Stage B — те же
источники, но семьи, которых нет в Stage A. Направления по-прежнему обучены
только на train; Stage C остаётся на test.

Вход:
  - steering/samples/h1_stagea_sample_v1.json
  - results/<run>/per_item.jsonl
  - results/<run>/probes/_shared/group_split_v1_scenario.json
  - steering/configs/h1_steering_candidates_v1.yaml (source.run)

Выход:
  - steering/samples/h1_stageb_sample_v1.json

Запуск из корня репозитория:
    python -m steering.build_stageb_sample
    python -m steering.build_stageb_sample --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from probes.v1_rep import build_v1_group_registry, is_v1_rep_row, v1_scenario_key
from steering.build_stagea_sample import load_rows, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
STEERING_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = STEERING_DIR / "configs" / "h1_steering_candidates_v1.yaml"
DEFAULT_STAGE_A = STEERING_DIR / "samples" / "h1_stagea_sample_v1.json"
DEFAULT_OUT_DIR = STEERING_DIR / "samples"


def build(cfg: dict, run_dir: Path, stage_a: dict, *, complement_path: Path) -> dict:
    source_split = stage_a["source_split"]
    rows_per_item = int(stage_a["rows_per_item"])
    stage_a_ids = {int(i["scenario_family_id"]) for i in stage_a["items"]}

    per_item = run_dir / "per_item.jsonl"
    split_path = run_dir / "probes" / "_shared" / "group_split_v1_scenario.json"
    rows = load_rows(per_item)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    group_to_split = {int(k): v for k, v in split["group_to_split"].items()}

    registry = build_v1_group_registry(rows)
    fam_rows: dict[int, list[dict]] = defaultdict(list)
    fam_soc: dict[int, str] = {}
    for row in rows:
        if not is_v1_rep_row(row):
            continue
        gid = registry[v1_scenario_key(row)]
        if group_to_split.get(gid) != source_split:
            continue
        fam_rows[gid].append(row)
        fam_soc.setdefault(gid, str(row.get("soc_major_title") or "UNKNOWN"))

    bad = {gid: len(rs) for gid, rs in fam_rows.items() if len(rs) != rows_per_item}
    if bad:
        raise ValueError(f"families with != {rows_per_item} rows: {dict(list(bad.items())[:5])}")

    missing_a = stage_a_ids - set(fam_rows)
    if missing_a:
        raise ValueError(f"Stage A families missing in val split: {sorted(missing_a)[:5]}")

    picked = sorted(gid for gid in fam_rows if gid not in stage_a_ids)
    if not picked:
        raise ValueError("остаток val пуст — Stage A забрал все семьи")

    by_soc: dict[str, list[int]] = defaultdict(list)
    for gid in picked:
        by_soc[fam_soc[gid]].append(gid)

    per_soc = [
        {
            "soc_major_title": soc,
            "n_families_in_split": sum(1 for g, s in fam_soc.items() if s == soc),
            "n_sampled": len(gids),
            "scenario_family_ids": sorted(gids),
        }
        for soc, gids in sorted(by_soc.items())
    ]

    items = []
    for gid in picked:
        rs = sorted(fam_rows[gid], key=lambda r: str(r["id"]))
        first = rs[0]
        items.append(
            {
                "scenario_family_id": gid,
                "soc_major_title": fam_soc[gid],
                "soc": str(first["soc"]),
                "profession": str(first["profession"]),
                "onet_action": str(first["onet_action"]),
                "rows": [
                    {
                        "id": str(r["id"]),
                        "position_variant": str(r["position_variant"]),
                        "context_order": str(r["context_order"]),
                        "labels": r["labels"],
                        "valid_labels": r["valid_labels"],
                        "prompt": r["prompt"],
                        "baseline_choice": str(r["choice"]),
                    }
                    for r in rs
                ],
            }
        )

    return {
        "schema": "steering.h1_stageb_sample/v1",
        "map_version": stage_a.get("map_version", cfg["version"]),
        "map_config": DEFAULT_CONFIG.name,
        "run": cfg["source"]["run"],
        "source_split": source_split,
        "complement_of": complement_path.name,
        "stage_a_n_base_items": stage_a["n_base_items"],
        "stage_a_family_ids_sha256": sha256_file(complement_path),
        "seed": stage_a.get("seed"),
        "stratify_by": "complement_of_stage_a",
        "rows_per_item": rows_per_item,
        "slice": stage_a.get("slice"),
        "n_families_in_split": len(fam_rows),
        "n_base_items": len(items),
        "n_rows": sum(len(i["rows"]) for i in items),
        "per_item_sha256": sha256_file(per_item),
        "split_sha256": sha256_file(split_path),
        "per_soc": per_soc,
        "items": items,
    }


def summary_markdown(doc: dict, *, title: str, capability_note: str) -> str:
    split = doc["source_split"]
    lines = [
        f"# {title} — `{doc['map_version']}`",
        "",
        f"- Run: `{doc['run']}`, split=`{split}` ({doc['n_families_in_split']} семей)",
        f"- Complement of `{doc['complement_of']}` ({doc['stage_a_n_base_items']} семей)",
        f"- Отобрано: **{doc['n_base_items']}** base items, **{doc['n_rows']}** строк",
        "",
        f"| soc_major_title | в {split} | в выборке |",
        "| :--- | ---: | ---: |",
    ]
    for r in sorted(doc["per_soc"], key=lambda x: -x["n_sampled"]):
        lines.append(
            f"| {r['soc_major_title']} | {r['n_families_in_split']} | {r['n_sampled']} |"
        )
    lines += ["", capability_note, ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--stage-a", type=Path, default=DEFAULT_STAGE_A, help="prior sample to complement")
    ap.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--out-stem",
        default=None,
        help="имя файлов без расширения (default: h1_stageb_sample_<ver>)",
    )
    ap.add_argument(
        "--title",
        default="Stage B sample",
        help="заголовок markdown",
    )
    ap.add_argument(
        "--capability-note",
        default="Preference Stage B; capability — `mmlu_pro_domain_val` + `overall_smoke`.",
    )
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    stage_a = json.loads(args.stage_a.read_text(encoding="utf-8"))
    run_dir = args.results_root / cfg["source"]["run"]
    if not (run_dir / "per_item.jsonl").exists():
        raise SystemExit(f"нет per_item.jsonl в {run_dir}")

    doc = build(cfg, run_dir, stage_a, complement_path=args.stage_a)
    md = summary_markdown(doc, title=args.title, capability_note=args.capability_note)
    ver = doc["map_version"]
    stem = args.out_stem or f"h1_stageb_sample_{ver}"
    json_path = args.out_dir / f"{stem}.json"
    md_path = args.out_dir / f"{stem}.md"
    text = json.dumps(doc, ensure_ascii=False, indent=2) + "\n"

    if args.verify:
        bad = []
        if not json_path.exists() or json_path.read_text(encoding="utf-8") != text:
            bad.append(json_path.name)
        if not md_path.exists() or md_path.read_text(encoding="utf-8") != md:
            bad.append(md_path.name)
        if bad:
            print("MISMATCH: " + ", ".join(bad))
            return 1
        print(f"OK — {stem} совпадает с пересборкой")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(text, encoding="utf-8")
    md_path.write_text(md, encoding="utf-8")
    print(
        f"{args.title}: {doc['n_base_items']} base items / {doc['n_rows']} строк "
        f"(остаток {doc['source_split']} {doc['n_families_in_split']} − "
        f"{doc['stage_a_n_base_items']} из {doc['complement_of']})"
    )
    print(f"  -> {json_path.name}")
    print(f"  -> {md_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
