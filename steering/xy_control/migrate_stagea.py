"""Migrate pre-fingerprint per-SOC Stage A artifacts without rerunning the model.

The affected Vast runs built the full XY dataset without the optional
group_split artifact. Their unknown families therefore received consecutive
IDs after the legacy sample IDs (3881, 3882, ...). Reconstruct that exact
CPU-only dataset, prove every frozen train/val/test split matches it, then add
dataset provenance to the vector and run metadata.

Example:
    python -m steering.xy_control.migrate_stagea \
      --stage-a-dir results/steering/xy_control/per_soc/xy_2b_per_soc_v1 \
      --scale 2b
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from steering.xy_control.build_dataset import FULL_SPEC, build_document
from steering.xy_control.h1_families import (
    LEGACY_SAMPLES,
    build_without_abstain_items,
    family_key,
    known_family_ids,
    per_soc_counts,
)
from steering.xy_control.paired_sample import dataset_provenance


def legacy_no_group_split_items() -> list[dict[str, Any]]:
    """Reproduce the ID branch used by the affected clean Vast checkout."""
    items = build_without_abstain_items()
    keys = [family_key(item) for item in items]
    known = known_family_ids(LEGACY_SAMPLES)
    assigned: dict[tuple[str, str, str], int] = {}
    used: set[int] = set()
    missing: list[tuple[str, str, str]] = []
    for key in keys:
        if key in known:
            assigned[key] = int(known[key])
            used.add(int(known[key]))
        else:
            missing.append(key)

    next_id = max(used) + 1 if used else 1
    for key in sorted(missing):
        while next_id in used:
            next_id += 1
        assigned[key] = next_id
        used.add(next_id)
        next_id += 1

    migrated = []
    for source in items:
        item = deepcopy(source)
        item["scenario_family_id"] = assigned[family_key(item)]
        migrated.append(item)
    migrated.sort(key=lambda item: int(item["scenario_family_id"]))
    return migrated


def legacy_xy_document() -> dict[str, Any]:
    items = legacy_no_group_split_items()
    n_rows = sum(len(item["rows"]) for item in items)
    source = {
        "schema": "steering.h1_without_abstain_full/v1",
        "n_base_items": len(items),
        "n_rows": n_rows,
        "source_split": "all",
        "items": items,
        "per_soc": per_soc_counts(items),
    }
    spec = {
        "role": FULL_SPEC["role"],
        "note": FULL_SPEC["note"],
        "path": None,
    }
    return build_document("full", spec, source)


def validate_splits(dataset: dict, vector_meta: dict) -> dict[str, int]:
    by_soc: dict[str, set[int]] = {}
    for item in dataset["items"]:
        title = str(item.get("soc_major_title") or "")
        by_soc.setdefault(title, set()).add(int(item["scenario_family_id"]))

    checked = 0
    expected_total = 0
    for domain in vector_meta.get("domains", []):
        slug = str(domain["slug"])
        title = str(
            next(
                (
                    entry.get("soc_major_title")
                    for entry in vector_meta.get("vectors", [])
                    if entry.get("slug") == slug and entry.get("soc_major_title")
                ),
                "",
            )
        )
        if not title:
            raise ValueError(f"{slug}: cannot recover soc_major_title from vector metadata")
        train = {int(x) for x in domain["train_family_ids"]}
        val = {int(x) for x in domain["val_family_ids"]}
        test = {int(x) for x in domain["test_family_ids"]}
        if train & val or train & test or val & test:
            raise ValueError(f"{slug}: overlapping train/val/test IDs")
        expected = train | val | test
        actual = by_soc.get(title, set())
        if expected != actual:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise ValueError(
                f"{slug}: reconstructed legacy IDs do not match vector splits; "
                f"missing={missing[:8]} unexpected={unexpected[:8]} "
                f"(expected {len(expected)}, actual {len(actual)})"
            )
        n_families = domain.get("n_families")
        if n_families is not None and int(n_families) != len(expected):
            raise ValueError(
                f"{slug}: metadata n_families={n_families}, split union={len(expected)}"
            )
        checked += 1
        expected_total += len(expected)
    if not checked:
        raise ValueError("vector metadata contains no per-SOC split domains")
    return {"n_domains": checked, "n_domain_families": expected_total}


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _json_text(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2) + "\n"


def resolve_vector_meta(stage_a_dir: Path, scale: str, override: Path | None) -> Path:
    if override is not None:
        candidates = [override]
    else:
        candidates = sorted(stage_a_dir.glob(f"xy_control_{scale}_per_soc_vectors_*.json"))
    if len(candidates) != 1:
        raise SystemExit(
            f"expected exactly one {scale} vector metadata JSON in {stage_a_dir}, "
            f"found {len(candidates)}"
        )
    path = candidates[0]
    if not path.is_file():
        raise SystemExit(f"vector metadata missing: {path}")
    return path


def migrate(
    stage_a_dir: Path,
    scale: str,
    *,
    vectors_json: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    vector_path = resolve_vector_meta(stage_a_dir, scale, vectors_json)
    vector_meta = json.loads(vector_path.read_text(encoding="utf-8"))
    if vector_meta.get("scale") != scale:
        raise SystemExit(f"vector scale {vector_meta.get('scale')} != --scale {scale}")

    dataset = legacy_xy_document()
    validation = validate_splits(dataset, vector_meta)
    dataset_path = stage_a_dir / "xy_pairs_full_v1.json"
    dataset_text = _json_text(dataset)

    if dry_run:
        with tempfile.TemporaryDirectory() as td:
            probe = Path(td) / dataset_path.name
            probe.write_text(dataset_text, encoding="utf-8", newline="\n")
            provenance = dataset_provenance(probe)
    else:
        _atomic_write(dataset_path, dataset_text)
        provenance = dataset_provenance(dataset_path)

    migration = {
        "xy_dataset_migration": "legacy_no_group_split_ids/v1",
        **provenance,
        **validation,
    }
    vector_meta.update(migration)

    run_meta_path = stage_a_dir / "run_meta.json"
    run_meta = (
        json.loads(run_meta_path.read_text(encoding="utf-8"))
        if run_meta_path.is_file()
        else {}
    )
    run_meta.update(migration)

    if not dry_run:
        _atomic_write(vector_path, _json_text(vector_meta))
        _atomic_write(run_meta_path, _json_text(run_meta))
    return {
        "stage_a_dir": str(stage_a_dir),
        "vectors_json": str(vector_path),
        "dataset": str(dataset_path),
        "dry_run": dry_run,
        **migration,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-a-dir", type=Path, required=True)
    ap.add_argument("--scale", choices=["2b", "4b"], required=True)
    ap.add_argument("--vectors-json", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    result = migrate(
        args.stage_a_dir,
        args.scale,
        vectors_json=args.vectors_json,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
