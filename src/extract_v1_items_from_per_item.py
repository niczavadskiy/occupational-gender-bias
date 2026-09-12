"""
Extract inference input JSONL from a completed per_item.jsonl (strip model outputs).

Source of truth for occupational v1 × evidence × layout:
  results/highlight-h3-full/per_item.jsonl  (45648 rows, Qwen3.5-2B-Base outputs)

Writes:
  data/v1/inference_items_h3_full.jsonl
  data/v1/inference_items_v1_{man,woman}_first.jsonl          # no_evidence × context
  data/v1/inference_items_hl_{man,woman}_{man,woman}_first.jsonl

Factorial check (must pass):
  951 scenarios × 3 evidence × 2 context × 2 positions (without_abstain) = 11412
  951 × 3 × 2 × 6 positions (with_abstain) = 34236
  total = 45648
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SRC = REPO / "results" / "highlight-h3-full" / "per_item.jsonl"
DEFAULT_OUT = REPO / "data" / "v1"

DROP_PREFIXES = ("logit_", "logprob_", "prob_constrained_")
DROP_KEYS = frozenset({"choice", "from_cache"})

KEEP_HINT = (
    "id",
    "prompt",
    "labels",
    "valid_labels",
    "has_abstain",
    "abstain_variant",
    "evidence_shift",
    "context_order",
    "position_variant",
    "scenario_text",
    "question",
    "soc",
    "profession",
    "soc_major_title",
    "onet_action",
    "task",
    "answer_type",
    "question_format",
)


def strip_row(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if k in DROP_KEYS or k.startswith(DROP_PREFIXES):
            continue
        out[k] = v
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate(rows: list[dict]) -> dict:
    n = len(rows)
    abst = Counter(r.get("abstain_variant") for r in rows)
    ev = Counter(r.get("evidence_shift") for r in rows)
    ctx = Counter(r.get("context_order") for r in rows)
    pos = Counter(r.get("position_variant") for r in rows)
    families = {
        (r.get("soc"), r.get("profession"), r.get("onet_action")) for r in rows
    }
    missing = [k for k in KEEP_HINT if any(k not in r for r in rows[: min(20, n)])]
    report = {
        "n": n,
        "n_families": len(families),
        "abstain": dict(abst),
        "evidence": dict(ev),
        "context": dict(ctx),
        "position": dict(sorted(pos.items(), key=lambda x: str(x[0]))),
        "missing_keys_in_sample": missing,
    }
    expect_without = 951 * 3 * 2 * 2
    expect_with = 951 * 3 * 2 * 6
    errors = []
    if n != 45648:
        errors.append(f"n={n} != 45648")
    if abst.get("without_abstain") != expect_without:
        errors.append(f"without_abstain={abst.get('without_abstain')} != {expect_without}")
    if abst.get("with_abstain") != expect_with:
        errors.append(f"with_abstain={abst.get('with_abstain')} != {expect_with}")
    if len(families) != 951:
        errors.append(f"families={len(families)} != 951")
    for e in ("no_evidence", "man", "woman"):
        if ev.get(e) != 15216:
            errors.append(f"evidence {e}={ev.get(e)} != 15216")
    for c in ("man_first", "woman_first"):
        if ctx.get(c) != 22824:
            errors.append(f"context {c}={ctx.get(c)} != 22824")
    if missing:
        errors.append(f"missing keys: {missing}")
    report["ok"] = not errors
    report["errors"] = errors
    return report


def shard_name(row: dict) -> str:
    ev = row["evidence_shift"]
    ctx = row["context_order"]
    if ev == "no_evidence":
        return f"inference_items_v1_{ctx}.jsonl"
    # man / woman evidence → hl_man / hl_woman
    return f"inference_items_hl_{ev}_{ctx}.jsonl"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    if not args.src.is_file():
        print(f"ERROR: source not found: {args.src}")
        return 1

    rows: list[dict] = []
    with args.src.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(strip_row(json.loads(line)))

    report = validate(rows)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["ok"]:
        print("ERROR: factorial validation failed")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    full_path = args.out_dir / "inference_items_h3_full.jsonl"
    write_jsonl(full_path, rows)
    print(f"wrote {full_path} ({len(rows)} rows)")

    shards: dict[str, list[dict]] = {}
    for row in rows:
        shards.setdefault(shard_name(row), []).append(row)
    for name, chunk in sorted(shards.items()):
        path = args.out_dir / name
        write_jsonl(path, chunk)
        print(f"wrote {path} ({len(chunk)} rows)")

    meta = {
        "source_per_item": str(args.src.as_posix()),
        "n_items": len(rows),
        "n_families": report["n_families"],
        "factorial": {
            "scenarios": 951,
            "evidence": 3,
            "context_order": 2,
            "without_abstain_positions": 2,
            "with_abstain_positions": 6,
            "without_abstain": 11412,
            "with_abstain": 34236,
            "total": 45648,
        },
        "files": {
            "full": "inference_items_h3_full.jsonl",
            "shards": sorted(shards.keys()),
        },
        "validation": report,
    }
    meta_path = args.out_dir / "README_items.md"
    meta_path.write_text(
        "\n".join(
            [
                "# data/v1 — occupational inference items (input only)",
                "",
                "Extracted from `results/highlight-h3-full/per_item.jsonl` (2B outputs stripped).",
                "",
                "## Factorial",
                "",
                "- scenarios (soc × profession × onet_action): **951**",
                "- evidence_shift: no_evidence / man / woman (**3**)",
                "- context_order: man_first / woman_first (**2**)",
                "- without_abstain positions: **2** → 951×3×2×2 = **11412**",
                "- with_abstain positions: **6** → 951×3×2×6 = **34236**",
                "- **total 45648**",
                "",
                "## Files",
                "",
                "| File | Rows |",
                "|---|---:|",
                "| `inference_items_h3_full.jsonl` | 45648 |",
                *[f"| `{n}` | {len(shards[n])} |" for n in sorted(shards)],
                "",
                "Rebuild:",
                "",
                "```bash",
                "python -m src.extract_v1_items_from_per_item",
                "```",
                "",
                "```json",
                json.dumps(meta, indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
