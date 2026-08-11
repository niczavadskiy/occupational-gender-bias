"""
Заморозка выборки Stage A: base items для скрининга H1 steering-кандидатов.

Берутся ТОЛЬКО val-семьи probe-split (направления обучены на train), выборка
стратифицирована по soc_major_title пропорционально размеру домена внутри val.
Каждая семья даёт 4 строки (p0/p1 × man_first/woman_first) — те же, на которых
считается θ_i = M_i/(M_i+W_i).

Вход:
  - steering/configs/h1_steering_candidates_v1.yaml (screening.sample)
  - results/<run>/per_item.jsonl
  - results/<run>/probes/_shared/group_split_v1_scenario.json

Выход:
  - steering/samples/h1_stagea_sample_v1.json

Запуск из корня репозитория:
    python -m steering.build_stagea_sample
    python -m steering.build_stagea_sample --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from probes.v1_rep import build_v1_group_registry, is_v1_rep_row, v1_scenario_key

REPO_ROOT = Path(__file__).resolve().parents[1]
STEERING_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = STEERING_DIR / "configs" / "h1_steering_candidates_v1.yaml"
DEFAULT_OUT_DIR = STEERING_DIR / "samples"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_rng(*parts) -> np.random.Generator:
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))


def load_rows(per_item: Path) -> list[dict]:
    rows = []
    with per_item.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def largest_remainder(weights: dict[str, float], total: int) -> dict[str, int]:
    norm = sum(weights.values())
    if norm <= 0:
        raise ValueError("empty weights")
    exact = {k: total * w / norm for k, w in weights.items()}
    base = {k: int(v) for k, v in exact.items()}
    left = total - sum(base.values())
    order = sorted(exact, key=lambda k: (-(exact[k] - base[k]), k))
    for k in order[:left]:
        base[k] += 1
    return base


def build(cfg: dict, run_dir: Path) -> dict:
    spec = cfg["screening"]["sample"]
    n_target = int(spec["n_base_items"])
    seed = spec["seed"]
    source_split = spec["source_split"]

    per_item = run_dir / "per_item.jsonl"
    split_path = run_dir / "probes" / "_shared" / "group_split_v1_scenario.json"
    rows = load_rows(per_item)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    group_to_split = {int(k): v for k, v in split["group_to_split"].items()}

    registry = build_v1_group_registry(rows)
    if len(registry) != split["n_scenario_families"]:
        raise ValueError(
            f"registry {len(registry)} != split {split['n_scenario_families']} families"
        )

    # Семьи целевого сплита + их soc_major_title и ровно 4 строки на семью.
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

    rows_per_item = int(spec["rows_per_item"])
    bad = {gid: len(rs) for gid, rs in fam_rows.items() if len(rs) != rows_per_item}
    if bad:
        raise ValueError(f"families with != {rows_per_item} rows: {dict(list(bad.items())[:5])}")

    by_soc: dict[str, list[int]] = defaultdict(list)
    for gid, soc in fam_soc.items():
        by_soc[soc].append(gid)
    for soc in by_soc:
        by_soc[soc].sort()

    n_available = len(fam_rows)
    if n_target > n_available:
        raise ValueError(f"n_base_items={n_target} > доступно {n_available} в split={source_split}")

    # Пропорционально размеру домена внутри val: сохраняем структуру сплита,
    # но гарантируем минимум 1 семью каждому присутствующему домену.
    alloc = largest_remainder({soc: float(len(g)) for soc, g in by_soc.items()}, n_target)
    for soc in sorted(alloc):
        if alloc[soc] == 0:
            alloc[soc] = 1
    while sum(alloc.values()) > n_target:
        biggest = max(sorted(alloc), key=lambda s: (alloc[s], len(by_soc[s])))
        if alloc[biggest] <= 1:
            break
        alloc[biggest] -= 1
    for soc in sorted(alloc):
        alloc[soc] = min(alloc[soc], len(by_soc[soc]))

    picked: list[int] = []
    per_soc: list[dict] = []
    for soc in sorted(by_soc):
        k = alloc.get(soc, 0)
        pool = by_soc[soc]
        rng = stable_rng(seed, source_split, soc)
        chosen = sorted(int(g) for g in rng.choice(np.asarray(pool), size=k, replace=False))
        picked.extend(chosen)
        per_soc.append(
            {
                "soc_major_title": soc,
                "n_families_in_split": len(pool),
                "n_sampled": len(chosen),
                "scenario_family_ids": chosen,
            }
        )
    picked.sort()

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
        "schema": "steering.h1_stagea_sample/v1",
        "map_version": cfg["version"],
        "map_config": DEFAULT_CONFIG.name,
        "run": cfg["source"]["run"],
        "source_split": source_split,
        "seed": seed,
        "stratify_by": spec["stratify_by"],
        "rows_per_item": rows_per_item,
        "slice": {
            "task": "main",
            "question_format": "choice",
            "abstain_variant": "without_abstain",
            "position_variants": ["p0", "p1"],
            "context_orders": ["man_first", "woman_first"],
        },
        "n_families_in_split": n_available,
        "n_base_items": len(items),
        "n_rows": sum(len(i["rows"]) for i in items),
        "per_item_sha256": sha256_file(per_item),
        "split_sha256": sha256_file(split_path),
        "per_soc": per_soc,
        "items": items,
    }


def summary_markdown(doc: dict) -> str:
    lines = [
        f"# Stage A sample — `{doc['map_version']}`",
        "",
        f"- Run: `{doc['run']}`, split=`{doc['source_split']}` ({doc['n_families_in_split']} семей)",
        f"- Отобрано: **{doc['n_base_items']}** base items, **{doc['n_rows']}** строк",
        f"- Стратификация: `{doc['stratify_by']}`, seed=`{doc['seed']}`",
        "",
        "| soc_major_title | в val | отобрано |",
        "| :--- | ---: | ---: |",
    ]
    for r in sorted(doc["per_soc"], key=lambda x: -x["n_sampled"]):
        lines.append(f"| {r['soc_major_title']} | {r['n_families_in_split']} | {r['n_sampled']} |")
    lines += ["", "Выборка заморожена: Stage B использует остаток val, Stage C — test.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_dir = args.results_root / cfg["source"]["run"]
    if not (run_dir / "per_item.jsonl").exists():
        raise SystemExit(f"нет per_item.jsonl в {run_dir}")

    doc = build(cfg, run_dir)
    md = summary_markdown(doc)
    ver = cfg["version"]
    json_path = args.out_dir / f"h1_stagea_sample_{ver}.json"
    md_path = args.out_dir / f"h1_stagea_sample_{ver}.md"
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
        print("OK — выборка совпадает с пересборкой")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(text, encoding="utf-8")
    md_path.write_text(md, encoding="utf-8")
    print(
        f"Stage A sample: {doc['n_base_items']} base items / {doc['n_rows']} строк "
        f"из {doc['n_families_in_split']} val-семей, {len(doc['per_soc'])} доменов"
    )
    print(f"  -> {json_path.name}")
    print(f"  -> {md_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
