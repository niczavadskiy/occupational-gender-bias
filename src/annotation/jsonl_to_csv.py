"""
Convert annotation JSONL into flat CSV.

Input format: data/annotation/runs/<run_id>/annotations.jsonl
Output: same path with .csv suffix by default.

Usage:
    python src/annotation/jsonl_to_csv.py \
        --input data/annotation/runs/<run_id>/annotations.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build_row(rec: dict) -> dict:
    labels = rec.get("labels") or {}
    ann = rec.get("annotation") or {}
    ann_labels = ann.get("option_labels") or {}
    batch = rec.get("batch") or {}
    usage = batch.get("usage") or {}
    cost_details = usage.get("cost_details") or {}

    return {
        "example_id": rec.get("example_id"),
        "base_id": rec.get("base_id"),
        "predicate": rec.get("predicate"),
        "evidence_shift": rec.get("evidence_shift"),
        "question_format": rec.get("question_format"),
        "abstain_variant": rec.get("abstain_variant"),
        "scenario_text": rec.get("scenario_text"),
        "question": rec.get("question"),
        "label_A": labels.get("A"),
        "label_B": labels.get("B"),
        "label_C": labels.get("C"),
        "answerability": ann.get("answerability"),
        "ann_A": ann_labels.get("A"),
        "ann_B": ann_labels.get("B"),
        "ann_C": ann_labels.get("C"),
        "annotator_model": rec.get("annotator_model"),
        "annotator_provider": rec.get("annotator_provider"),
        "batch_n": batch.get("n"),
        "batch_request_id": batch.get("openrouter_request_id"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cost_usd": usage.get("cost"),
        "prompt_cost_usd": cost_details.get("upstream_inference_prompt_cost"),
        "completion_cost_usd": cost_details.get("upstream_inference_completions_cost"),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert annotations.jsonl to flat CSV.")
    p.add_argument("--input", type=Path, required=True, help="Path to annotations.jsonl")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output CSV (default: <input>.csv)",
    )
    args = p.parse_args(argv)

    in_path = args.input.resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    out_path = args.output.resolve() if args.output else in_path.with_suffix(".csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with open(in_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num}: {e}") from e
            rows.append(build_row(rec))

    if not rows:
        raise ValueError(f"No records found in {in_path}")

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Converted {len(rows)} rows: {in_path} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
