"""
Build paired gendered / XY-control datasets.

Primary: all H1 without_abstain families (951 × 4 = 3804) from v1 inference
items. Legacy 95-family Stage A/B/C files are still written for reference.

Mapping is fixed: X = man, Y = woman. Structure, A/B order, and the question
wording stay identical; only gender designators are rewritten.

    python -m steering.xy_control.build_dataset
    python -m steering.xy_control.build_dataset --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from steering.xy_control.h1_families import (
    ROWS_PER_ITEM,
    SLICE,
    build_without_abstain_items,
    per_soc_counts,
)
from steering.xy_control.mapping import (
    GENDER_TO_XY,
    MAPPING,
    count_gender_terms,
    gender_to_xy_labels,
    gender_to_xy_text,
    leftover_gender_terms,
    validate_pair,
)

HERE = Path(__file__).resolve().parent
STEERING_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = HERE / "data"

SOURCE_SAMPLES = {
    "stagea": {
        "path": STEERING_DIR / "samples" / "h1_stagea_sample_v1.json",
        "role": "fit",
        "note": "Probe-val Stage A (95 families). Used to fit v_raw for the first experiment.",
    },
    "stageb": {
        "path": STEERING_DIR / "samples" / "h1_stageb_sample_v1.json",
        "role": "val",
        "note": "Probe-val Stage B complement. Held out of v_raw; used to pick α / layer.",
    },
    "test": {
        "path": STEERING_DIR / "samples" / "inlp_test_sample_v1.json",
        "role": "test",
        "note": "Probe-test Stage C sample. Not used to fit v or pick α / layer.",
    },
    "stagec": {
        "path": STEERING_DIR / "samples" / "inlp_stagec_sample_v1.json",
        "role": "test_complement",
        "note": "Probe-test complement of inlp_test_sample_v1.",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def convert_row(src: dict[str, Any]) -> dict[str, Any]:
    gender_prompt = src["prompt"]
    labels = dict(src["labels"])
    xy_prompt = gender_to_xy_text(gender_prompt)
    xy_labels = gender_to_xy_labels(labels)
    validate_pair(gender_prompt, xy_prompt, labels)
    leftover = leftover_gender_terms(xy_prompt)
    if leftover:
        raise ValueError(f"{src.get('id')}: leftover gender terms {leftover}")
    row = deepcopy(src)
    row["gender_prompt"] = gender_prompt
    row["xy_prompt"] = xy_prompt
    row["xy_labels"] = xy_labels
    row["mapping"] = dict(MAPPING)
    row["prompt"] = gender_prompt
    return row


def convert_item(item: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(item)
    out["rows"] = [convert_row(row) for row in item["rows"]]
    return out


def pair_records(doc: dict[str, Any]) -> list[dict[str, Any]]:
    pairs = []
    for item in doc["items"]:
        for row in item["rows"]:
            g_counts = count_gender_terms(row["gender_prompt"])
            pairs.append(
                {
                    "pair_id": row["id"],
                    "scenario_family_id": item["scenario_family_id"],
                    "soc_major_title": item["soc_major_title"],
                    "profession": item.get("profession"),
                    "position_variant": row["position_variant"],
                    "context_order": row["context_order"],
                    "labels": row["labels"],
                    "xy_labels": row["xy_labels"],
                    "valid_labels": row["valid_labels"],
                    "gender_prompt": row["gender_prompt"],
                    "xy_prompt": row["xy_prompt"],
                    "mapping": row["mapping"],
                    "n_man": g_counts["man"],
                    "n_woman": g_counts["woman"],
                }
            )
    return pairs


def build_full_source() -> dict[str, Any]:
    items = build_without_abstain_items()
    n_rows = sum(len(it["rows"]) for it in items)
    if len(items) != 951 or n_rows != 3804:
        raise ValueError(f"expected 951×4=3804, got {len(items)} families / {n_rows} rows")
    return {
        "schema": "steering.h1_without_abstain_full/v1",
        "n_base_items": len(items),
        "n_rows": n_rows,
        "source_split": "all",
        "items": items,
        "per_soc": per_soc_counts(items),
    }


FULL_SPEC = {
    "role": "all_without_abstain",
    "note": (
        "All H1 without_abstain families: 951 scenarios × p0/p1 × man_first/woman_first. "
        "v_raw is fit on 70% of families inside each SOC; Stage A GenderGap "
        "is scored on that SOC's val (15%). test (15%) stays locked until "
        "α/layer are chosen. XY prompts are only used to fit v."
    ),
    "path": DATA_DIR / "xy_pairs_full_v1.json",
}


def build_document(split_name: str, spec: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    items = [convert_item(item) for item in source["items"]]
    n_rows = sum(len(it["rows"]) for it in items)
    layouts = sorted(
        {
            (row["context_order"], row["position_variant"])
            for item in items
            for row in item["rows"]
        }
    )
    return {
        "schema": "steering.xy_control_pairs/v1",
        "map_version": "v1",
        "split_name": split_name,
        "role": spec["role"],
        "note": spec["note"],
        "mapping": dict(MAPPING),
        "gender_to_xy": dict(GENDER_TO_XY),
        "replacement": {
            "rule": r"word-boundary \b(woman|man)\b, woman before man",
            "preserve": [
                "question wording",
                "A/B order",
                "context_order",
                "position_variant",
                "system prompt (none)",
                "trailing Answer:",
            ],
        },
        "slice": dict(SLICE),
        "rows_per_item": ROWS_PER_ITEM,
        "source_sample": {
            "file": (
                spec["path"].name
                if spec.get("path") is not None and Path(spec["path"]).is_file()
                else "data/v1/inference_items_v1_{man,woman}_first.jsonl"
            ),
            "schema": source.get("schema"),
            "source_split": source.get("source_split"),
            "n_base_items": source.get("n_base_items"),
            "n_rows": source.get("n_rows"),
            "sha256": (
                sha256_file(spec["path"])
                if spec.get("path") is not None and Path(spec["path"]).is_file()
                else None
            ),
            "per_soc": source.get("per_soc"),
        },
        "n_base_items": len(items),
        "n_rows": n_rows,
        "n_pairs": n_rows,
        "layouts": [{"context_order": c, "position_variant": p} for c, p in layouts],
        "items": items,
    }


def dump(path: Path, doc: dict[str, Any]) -> str:
    text = json.dumps(doc, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def write_jsonl(path: Path, pairs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_mapping_file(path: Path) -> None:
    payload = {
        "schema": "steering.xy_control_mapping/v1",
        "mapping": dict(MAPPING),
        "gender_to_xy": dict(GENDER_TO_XY),
        "note": (
            "X always denotes the man candidate, Y the woman candidate. "
            "Position swap (p0/p1) and mention order (man_first/woman_first) "
            "are preserved, so A. Y / B. X is the p1 counterpart of A. man / B. woman."
        ),
    }
    dump(path, payload)


def out_paths(split_name: str, out_dir: Path) -> tuple[Path, Path]:
    return (
        out_dir / f"xy_pairs_{split_name}_v1.json",
        out_dir / f"xy_pairs_{split_name}_v1.jsonl",
    )


def build_index(built: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "steering.xy_control_index/v1",
        "mapping": dict(MAPPING),
        "splits": built,
    }


def write_split(
    *,
    name: str,
    spec: dict[str, Any],
    source: dict[str, Any],
    out_dir: Path,
    verify: bool,
    mismatched: list[str],
) -> dict[str, Any]:
    doc = build_document(name, spec, source)
    json_path, jsonl_path = out_paths(name, out_dir)
    text = json.dumps(doc, ensure_ascii=False, indent=2) + "\n"
    pairs = pair_records(doc)
    if verify:
        if not json_path.is_file() or json_path.read_text(encoding="utf-8") != text:
            mismatched.append(json_path.name)
        expected_jsonl = "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs)
        if not jsonl_path.is_file() or jsonl_path.read_text(encoding="utf-8") != expected_jsonl:
            mismatched.append(jsonl_path.name)
    else:
        dump(json_path, doc)
        write_jsonl(jsonl_path, pairs)
    print(
        f"  {name:<16} {doc['n_base_items']:>3} families  {doc['n_pairs']:>4} pairs  "
        f"role={spec['role']}  -> {json_path.name}"
    )
    src_name = spec["path"].name if spec.get("path") is not None else "v1 inference jsonl"
    return {
        "split_name": name,
        "role": spec["role"],
        "file": json_path.name,
        "jsonl": jsonl_path.name,
        "n_base_items": doc["n_base_items"],
        "n_pairs": doc["n_pairs"],
        "source_sample": src_name,
        "sha256": sha256_bytes(text.encode("utf-8")),
        "primary": name == "full",
    }


def full_summary_markdown(source: dict[str, Any]) -> str:
    lines = [
        "# XY-control full without_abstain — `v1`",
        "",
        f"- Families: **{source['n_base_items']}**",
        f"- Rows: **{source['n_rows']}** (4 layouts: p0/p1 × man_first/woman_first)",
        "- Slice: `without_abstain`, no evidence factor",
        "- Fit `v_raw` on 70% families **inside each SOC**; GenderGap on val 15%; test 15% locked",
        "",
        "| soc_major_title | families | rows |",
        "| :--- | ---: | ---: |",
    ]
    for row in source.get("per_soc") or []:
        lines.append(f"| {row['soc_major_title']} | {row['n_families']} | {row['n_rows']} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument(
        "--splits",
        default="full",
        help="Comma-separated: full (default) and/or legacy stagea,stageb,test,stagec",
    )
    ap.add_argument("--legacy", action="store_true", help="Also write the old 95-family splits")
    args = ap.parse_args(argv)

    wanted = [s.strip() for s in args.splits.split(",") if s.strip()]
    if args.legacy and "full" in wanted:
        wanted = ["full", *SOURCE_SAMPLES]
    unknown = [s for s in wanted if s not in SOURCE_SAMPLES and s != "full"]
    if unknown:
        raise SystemExit(f"unknown splits: {unknown}; known {['full', *SOURCE_SAMPLES]}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = out_dir / "mapping.json"
    write_mapping_file(mapping_path)

    index_rows: list[dict[str, Any]] = []
    mismatched: list[str] = []

    for name in wanted:
        if name == "full":
            source = build_full_source()
            spec = {"role": FULL_SPEC["role"], "note": FULL_SPEC["note"], "path": None}
            index_rows.append(
                write_split(
                    name=name,
                    spec=spec,
                    source=source,
                    out_dir=out_dir,
                    verify=args.verify,
                    mismatched=mismatched,
                )
            )
            md_path = out_dir / "xy_pairs_full_v1.md"
            md = full_summary_markdown(source)
            if args.verify:
                if not md_path.is_file() or md_path.read_text(encoding="utf-8") != md:
                    mismatched.append(md_path.name)
            else:
                md_path.write_text(md, encoding="utf-8")
            continue
        spec = SOURCE_SAMPLES[name]
        src_path = spec["path"]
        if not src_path.is_file():
            raise SystemExit(f"нет исходного sample {src_path}")
        source = json.loads(src_path.read_text(encoding="utf-8"))
        index_rows.append(
            write_split(
                name=name,
                spec=spec,
                source=source,
                out_dir=out_dir,
                verify=args.verify,
                mismatched=mismatched,
            )
        )

    index = build_index(index_rows)
    index["primary"] = "full"
    index_path = out_dir / "xy_pairs_index_v1.json"
    index_text = json.dumps(index, ensure_ascii=False, indent=2) + "\n"
    if args.verify:
        if not index_path.is_file() or index_path.read_text(encoding="utf-8") != index_text:
            mismatched.append(index_path.name)
        expected_map = json.dumps(
            {
                "schema": "steering.xy_control_mapping/v1",
                "mapping": dict(MAPPING),
                "gender_to_xy": dict(GENDER_TO_XY),
                "note": (
                    "X always denotes the man candidate, Y the woman candidate. "
                    "Position swap (p0/p1) and mention order (man_first/woman_first) "
                    "are preserved, so A. Y / B. X is the p1 counterpart of A. man / B. woman."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n"
        if not mapping_path.is_file() or mapping_path.read_text(encoding="utf-8") != expected_map:
            mismatched.append(mapping_path.name)
        if mismatched:
            print("MISMATCH: " + ", ".join(mismatched))
            return 1
        print("OK — XY datasets совпадают с пересборкой")
        return 0

    dump(index_path, index)
    print(f"  mapping          -> {mapping_path.name}")
    print(f"  index            -> {index_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
