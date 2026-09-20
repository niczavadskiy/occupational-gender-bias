"""All H1 without_abstain families (951 × 4 layouts) from v1 inference items."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
STEERING_DIR = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent

INFERENCE_FILES = (
    REPO_ROOT / "data" / "v1" / "inference_items_v1_man_first.jsonl",
    REPO_ROOT / "data" / "v1" / "inference_items_v1_woman_first.jsonl",
)
LEGACY_SAMPLES = (
    STEERING_DIR / "samples" / "h1_stagea_sample_v1.json",
    STEERING_DIR / "samples" / "h1_stageb_sample_v1.json",
    STEERING_DIR / "samples" / "inlp_test_sample_v1.json",
    STEERING_DIR / "samples" / "inlp_stagec_sample_v1.json",
)
FROZEN_FULL_DATASET = HERE / "data" / "xy_pairs_full_v1.json"
FAMILY_ID_REFERENCES = (*LEGACY_SAMPLES, FROZEN_FULL_DATASET)

SLICE = {
    "task": "main",
    "question_format": "choice",
    "abstain_variant": "without_abstain",
    "position_variants": ("p0", "p1"),
    "context_orders": ("man_first", "woman_first"),
    "evidence_shift": "no_evidence",
}
ROWS_PER_ITEM = 4


def family_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row["soc"]), str(row["profession"]), str(row["onet_action"]))


def is_without_abstain_layout(row: dict[str, Any]) -> bool:
    if str(row.get("task", "main")) != SLICE["task"]:
        return False
    if str(row.get("question_format", "")) != SLICE["question_format"]:
        return False
    if str(row.get("abstain_variant", "")) != SLICE["abstain_variant"]:
        return False
    if str(row.get("position_variant", "")) not in SLICE["position_variants"]:
        return False
    if str(row.get("context_order", "")) not in SLICE["context_orders"]:
        return False
    evidence = row.get("evidence_shift")
    if evidence is not None and str(evidence) != SLICE["evidence_shift"]:
        return False
    return True


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_inference_rows(paths: tuple[Path, ...] | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths or INFERENCE_FILES:
        if not path.is_file():
            raise FileNotFoundError(path)
        out.extend(load_jsonl(path))
    return out


def known_family_ids(sample_paths: tuple[Path, ...] | None = None) -> dict[tuple[str, str, str], int]:
    mapping: dict[tuple[str, str, str], int] = {}
    for path in sample_paths or FAMILY_ID_REFERENCES:
        if not path.is_file():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        for item in doc.get("items", []):
            key = family_key(item)
            fid = int(item["scenario_family_id"])
            prev = mapping.get(key)
            if prev is not None and prev != fid:
                raise ValueError(f"family id clash for {key}: {prev} vs {fid}")
            mapping[key] = fid
    return mapping


def assign_family_ids(keys: list[tuple[str, str, str]]) -> dict[tuple[str, str, str], int]:
    """Preserve frozen IDs; assign any genuinely new families deterministically.

    The committed full XY dataset is the authoritative ID map. Do not derive
    IDs from optional experiment artifacts: those are absent on clean Vast
    checkouts and previously made Stage A IDs differ from Stage C IDs.
    """
    known = known_family_ids()
    assigned: dict[tuple[str, str, str], int] = {}
    used = set()
    missing: list[tuple[str, str, str]] = []
    for key in keys:
        if key in known:
            assigned[key] = known[key]
            used.add(known[key])
        else:
            missing.append(key)
    leftover: list[int] = []
    nxt = (max(used) + 1) if used else 1
    while len(leftover) < len(missing):
        if nxt not in used:
            leftover.append(nxt)
        nxt += 1
    for key, fid in zip(sorted(missing), leftover):
        assigned[key] = fid
        used.add(fid)
    return assigned


def row_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "id": str(row["id"]),
        "position_variant": str(row["position_variant"]),
        "context_order": str(row["context_order"]),
        "labels": dict(row["labels"]),
        "valid_labels": list(row["valid_labels"]),
        "prompt": row["prompt"],
    }
    if row.get("baseline_choice") is not None:
        payload["baseline_choice"] = str(row["baseline_choice"])
    return payload


def build_without_abstain_items(
    rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    src = [r for r in (rows if rows is not None else load_inference_rows()) if is_without_abstain_layout(r)]
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    meta: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in src:
        key = family_key(row)
        by_key[key].append(row)
        meta.setdefault(
            key,
            {
                "soc": str(row["soc"]),
                "profession": str(row["profession"]),
                "onet_action": str(row["onet_action"]),
                "soc_major_title": str(row.get("soc_major_title") or "UNKNOWN"),
            },
        )
    bad = {k: len(v) for k, v in by_key.items() if len(v) != ROWS_PER_ITEM}
    if bad:
        sample = list(bad.items())[:5]
        raise ValueError(f"families with != {ROWS_PER_ITEM} without_abstain rows: {sample}")
    ids = assign_family_ids(sorted(by_key))
    items = []
    for key in sorted(by_key):
        info = meta[key]
        rs = sorted(
            by_key[key],
            key=lambda r: (str(r["context_order"]), str(r["position_variant"]), str(r["id"])),
        )
        items.append(
            {
                "scenario_family_id": ids[key],
                "soc_major_title": info["soc_major_title"],
                "soc": info["soc"],
                "profession": info["profession"],
                "onet_action": info["onet_action"],
                "rows": [row_payload(r) for r in rs],
            }
        )
    items.sort(key=lambda it: int(it["scenario_family_id"]))
    return items


def per_soc_counts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = defaultdict(int)
    for it in items:
        counts[str(it["soc_major_title"])] += 1
    return [
        {"soc_major_title": title, "n_families": counts[title], "n_rows": counts[title] * ROWS_PER_ITEM}
        for title in sorted(counts)
    ]
