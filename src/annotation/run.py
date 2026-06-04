"""
Batch LLM annotation via OpenRouter (default: anthropic/claude-opus-4.8).

Items are sent in batches (default 10 per API call). Structured JSON output.

Outputs under data/annotation/runs/<run_id>/:
  - annotations.jsonl
  - errors.jsonl
  - meta.json

Usage (from repo root):
    python src/annotation/run.py --limit 5 --dry-run
    python src/annotation/run.py --csv data/factorial_v2_with_gender.csv
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from annotation.client import DEFAULT_MODEL, chat_with_retries  # noqa: E402
from annotation.format_item import (  # noqa: E402
    format_batch_user_message,
    option_letters,
    row_from_csv_series,
    row_from_jsonl_item,
)
from annotation.rubric import build_system_prompt, load_rubric  # noqa: E402
from annotation.validate import extract_json_object, validate_batch_response  # noqa: E402

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = _REPO_ROOT / "data" / "factorial_v2_with_gender.csv"
DEFAULT_OUT_BASE = _REPO_ROOT / "data" / "annotation" / "runs"
DEFAULT_BATCH_SIZE = 10


def load_rows_csv(path: Path) -> list[dict]:
    df = pd.read_csv(path)
    return [row_from_csv_series(row) for _, row in df.iterrows()]


def load_rows_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(row_from_jsonl_item(json.loads(line)))
    return rows


def load_done_ids(annotations_path: Path) -> set[int]:
    if not annotations_path.is_file():
        return set()
    done: set[int] = set()
    with open(annotations_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(int(json.loads(line)["example_id"]))
    return done


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def model_slug(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def chunk_rows(rows: list[dict], size: int) -> list[list[dict]]:
    return [rows[i : i + size] for i in range(0, len(rows), size)]


def max_tokens_for_batch(batch_size: int) -> int:
    return max(512, batch_size * 128)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Annotate factorial items via OpenRouter (batched).")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Factorial CSV input")
    src.add_argument("--jsonl", type=Path, help="Prepared JSONL from prepare_factorial.py")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rubric", type=Path, default=None)
    p.add_argument("--max-retries", type=int, default=3, help="Retries per batch on parse/validation error")
    p.add_argument("--sleep", type=float, default=0.0, help="Seconds between batch API calls")
    args = p.parse_args(argv)

    if args.batch_size < 1:
        print("--batch-size must be >= 1", file=sys.stderr)
        return 1

    if args.csv and not args.csv.is_file() and args.jsonl is None:
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        return 1

    rubric_text = load_rubric(args.rubric)
    system_prompt = build_system_prompt(rubric_text)
    rubric_sha256 = hashlib.sha256(rubric_text.encode("utf-8")).hexdigest()

    if args.jsonl:
        rows = load_rows_jsonl(args.jsonl)
        dataset_path = str(args.jsonl.resolve())
    else:
        rows = load_rows_csv(args.csv.resolve())
        dataset_path = str(args.csv.resolve())

    if args.offset:
        rows = rows[args.offset :]
    if args.limit is not None:
        rows = rows[: args.limit]

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = args.run_id or f"{ts}_{model_slug(args.model)}"
    out_dir = args.out_dir or (DEFAULT_OUT_BASE / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    annotations_path = out_dir / "annotations.jsonl"
    errors_path = out_dir / "errors.jsonl"
    meta_path = out_dir / "meta.json"

    if args.dry_run:
        if not rows:
            print("No rows to show.", file=sys.stderr)
            return 1
        sample = rows[: min(2, len(rows))]
        print("=== system ===")
        print(system_prompt)
        print("\n=== user (batch sample) ===")
        print(format_batch_user_message(sample))
        print("\n=== expected keys per id ===")
        for r in sample:
            print(f"  id {r['example_id']}: {option_letters(r['has_abstain'])}")
        print(f"\nWould write to: {out_dir}")
        print(f"Batch size: {args.batch_size}")
        return 0

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("Set OPENROUTER_API_KEY in .env or environment.", file=sys.stderr)
        return 1

    done_ids = set() if args.no_resume else load_done_ids(annotations_path)
    todo = [r for r in rows if int(r["example_id"]) not in done_ids]
    batches = chunk_rows(todo, args.batch_size)

    meta = {
        "run_id": run_id,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "batch_size": args.batch_size,
        "dataset_path": dataset_path,
        "rubric_sha256": rubric_sha256,
        "n_dataset": len(rows),
        "n_skip_resume": len(rows) - len(todo),
        "n_todo": len(todo),
        "n_batches": len(batches),
        "offset": args.offset,
        "limit": args.limit,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    n_ok = 0
    n_err = 0

    for batch in tqdm(batches, desc="batches"):
        batch_ids = [int(r["example_id"]) for r in batch]
        user_msg = format_batch_user_message(batch)

        annotations_by_id = None
        last_error = None
        raw_content = None
        usage = None
        resolved_model = args.model
        request_id = None

        for attempt in range(args.max_retries):
            try:
                if attempt > 0 and raw_content is not None:
                    user_retry = (
                        user_msg
                        + "\n\nPrevious response invalid:\n"
                        + str(last_error)
                        + "\nReturn valid JSON: {\"items\": [{\"id\", \"answerability\", \"option_labels\"}, ...]}."
                    )
                else:
                    user_retry = user_msg

                resp = chat_with_retries(
                    api_key=api_key,
                    model=args.model,
                    system=system_prompt,
                    user=user_retry,
                    max_tokens=max_tokens_for_batch(len(batch)),
                )
                raw_content = resp["content"]
                usage = resp.get("usage")
                resolved_model = resp.get("model", args.model)
                request_id = resp.get("id")

                parsed = extract_json_object(raw_content)
                annotations_by_id = validate_batch_response(parsed, batch)
                break
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt == args.max_retries - 1:
                    append_jsonl(
                        errors_path,
                        {
                            "example_ids": batch_ids,
                            "error": str(e),
                            "raw_content": raw_content,
                        },
                    )
                    n_err += len(batch)

        if annotations_by_id is None:
            continue

        batch_info = {
            "ids": batch_ids,
            "n": len(batch_ids),
            "usage": usage,
            "openrouter_request_id": request_id,
        }

        for row in batch:
            eid = int(row["example_id"])
            rec = {
                "example_id": eid,
                "base_id": row["base_id"],
                "predicate": row["predicate"],
                "evidence_shift": row["evidence_shift"],
                "question_format": row["question_format"],
                "abstain_variant": row["abstain_variant"],
                "scenario_text": row["scenario_text"],
                "question": row["question"],
                "labels": row["labels"],
                "annotation": annotations_by_id[eid],
                "annotator_model": resolved_model,
                "annotator_provider": "openrouter",
                "batch": batch_info,
            }
            append_jsonl(annotations_path, rec)
            n_ok += 1

        if args.sleep > 0:
            time.sleep(args.sleep)

    meta["finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    meta["n_ok"] = n_ok
    meta["n_err"] = n_err
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nDone. ok={n_ok} err={n_err} batches={len(batches)} → {out_dir}")
    print("Note: batch.usage is duplicated per item; do not sum usage across all jsonl lines.")
    return 0 if n_err == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
