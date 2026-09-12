"""
Merge two inference runs that share the same row order modulo context_order.

Join key: v3 — (context_order, position_variant, example_id, task);
v1 — (context_order, id) with pair check on profession × onet_action × format × abstain × position × task.
Output row order: for each index i in the source runs, man_first[i] then woman_first[i].
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

MERGE_KEY_V3 = ("context_order", "position_variant", "example_id", "task")
PAIR_KEY_V3 = ("position_variant", "example_id", "task")
MERGE_KEY_V1 = ("context_order", "id")
PAIR_KEY_V1 = (
    "position_variant",
    "profession",
    "onet_action",
    "task",
    "question_format",
    "abstain_variant",
)
MERGE_KEY_HIGHLIGHT = ("highlight", "context_order", "id")
PAIR_KEY_HIGHLIGHT = PAIR_KEY_V1
HIGHLIGHT_RUN_ORDER = (
    "highlight-MAN__ctx-man-first",
    "highlight-MAN__ctx-woman-first",
    "highlight-WOMAN__ctx-man-first",
    "highlight-WOMAN__ctx-woman-first",
)
H3_EVIDENCE_DEFAULT_SOURCES: tuple[tuple[str, str], ...] = (
    ("run_2026-06-09_12-50-33_Qwen3.5-2B-Base_v1_full_pos_shuffle", "no_evidence"),
    ("highlight-MAN__ctx-man-first", "man"),
    ("highlight-MAN__ctx-woman-first", "man"),
    ("highlight-WOMAN__ctx-man-first", "woman"),
    ("highlight-WOMAN__ctx-woman-first", "woman"),
)
MERGE_KEY_H3_EVIDENCE = ("evidence_shift", "context_order", "id")
CONTEXT_ORDER = ("man_first", "woman_first")


def _highlight_from_row(row: dict[str, Any]) -> str:
    text = row.get("scenario_text") or row.get("prompt") or ""
    if "highlighted the man" in text:
        return "man"
    if "highlighted the woman" in text:
        return "woman"
    raise ValueError(f"cannot infer highlight from row id={row.get('id')!r}")


def _resolve_keys(row: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if "example_id" in row:
        return MERGE_KEY_V3, PAIR_KEY_V3
    return MERGE_KEY_V1, PAIR_KEY_V1


def _load_run(run_dir: Path) -> tuple[list[dict[str, Any]], np.ndarray | None, dict[str, Any]]:
    items_path = run_dir / "per_item.jsonl"
    npz_path = run_dir / "hidden_states.npz"
    meta_path = run_dir / "meta.json"

    if not items_path.is_file():
        raise FileNotFoundError(f"per_item.jsonl not found: {items_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")

    items: list[dict[str, Any]] = []
    with items_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    hs: np.ndarray | None = None
    if npz_path.is_file():
        data = np.load(npz_path, mmap_mode="r")
        hs = data["hs"]
        item_ids = [str(x) for x in data["item_ids"]]

        if len(items) != hs.shape[0]:
            raise ValueError(
                f"{run_dir.name}: per_item={len(items)} vs hidden_states={hs.shape[0]}"
            )
        expected_ids = [r["id"] for r in items]
        if item_ids != expected_ids:
            raise ValueError(f"{run_dir.name}: item_ids do not match per_item.jsonl order")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return items, hs, meta


def _key(row: dict[str, Any], merge_key: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(row[k] for k in merge_key)


def merge_runs(
    run_a: Path,
    run_b: Path,
    out_dir: Path,
    *,
    context_a: str = "man_first",
    context_b: str = "woman_first",
    run_tag: str = "v3_full_pos_shuffle",
) -> dict[str, Any]:
    items_a, hs_a, meta_a = _load_run(run_a)
    items_b, hs_b, meta_b = _load_run(run_b)

    if meta_a.get("model_id") != meta_b.get("model_id"):
        raise ValueError("model_id mismatch between runs")
    if (hs_a is None) != (hs_b is None):
        raise ValueError("hidden_states.npz present in only one run")
    if hs_a is not None and hs_b is not None and hs_a.shape[1:] != hs_b.shape[1:]:
        raise ValueError(f"hidden state shape mismatch: {hs_a.shape} vs {hs_b.shape}")

    merge_key, pair_key = _resolve_keys(items_a[0])
    if _resolve_keys(items_b[0]) != (merge_key, pair_key):
        raise ValueError("merge key schema mismatch between runs")

    keys_a = {_key(r, merge_key) for r in items_a}
    keys_b = {_key(r, merge_key) for r in items_b}
    overlap = keys_a & keys_b
    if overlap:
        raise ValueError(f"overlapping merge keys: {len(overlap)} duplicates")

    if len(items_a) != len(items_b):
        raise ValueError(f"row count mismatch: {len(items_a)} vs {len(items_b)}")

    for i, (ra, rb) in enumerate(zip(items_a, items_b)):
        if ra["context_order"] != context_a:
            raise ValueError(
                f"{run_a.name} row {i}: expected context_order={context_a!r}, "
                f"got {ra['context_order']!r}"
            )
        if rb["context_order"] != context_b:
            raise ValueError(
                f"{run_b.name} row {i}: expected context_order={context_b!r}, "
                f"got {rb['context_order']!r}"
            )
        pa = tuple(ra[k] for k in pair_key)
        pb = tuple(rb[k] for k in pair_key)
        if pa != pb:
            raise ValueError(f"pair key mismatch at index {i}: {pa} vs {pb}")

    n = len(items_a) * 2
    merged_items: list[dict[str, Any]] = []
    merged_hs: np.ndarray | None = None
    if hs_a is not None:
        assert hs_b is not None
        merged_hs = np.empty((n, *hs_a.shape[1:]), dtype=hs_a.dtype)
        for i, (ra, rb, ha, hb) in enumerate(zip(items_a, items_b, hs_a, hs_b)):
            merged_items.append(ra)
            merged_items.append(rb)
            merged_hs[2 * i] = np.asarray(ha)
            merged_hs[2 * i + 1] = np.asarray(hb)
    else:
        for ra, rb in zip(items_a, items_b):
            merged_items.append(ra)
            merged_items.append(rb)

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_written: Path | None = None
    if merged_hs is not None:
        merged_ids = np.array([r["id"] for r in merged_items])
        build_path = out_dir / "_hidden_states_building.npz"
        final_path = out_dir / "hidden_states.npz"
        if build_path.is_file():
            build_path.unlink()
        # Uncompressed write avoids zlib failures on multi-GB arrays on Windows.
        np.savez(build_path, hs=merged_hs, item_ids=merged_ids)
        del merged_hs
        with np.load(build_path) as check:
            if check["hs"].shape[0] != len(merged_items):
                raise ValueError("post-write validation failed: hs row count")
        npz_written = build_path
        try:
            if final_path.is_file():
                os.remove(final_path)
            os.replace(build_path, final_path)
            npz_written = final_path
        except OSError:
            # e.g. corrupt hidden_states.npz locked on Windows — validated build file kept.
            npz_written = build_path
    with (out_dir / "per_item.jsonl").open("w", encoding="utf-8") as f:
        for row in merged_items:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_cache = sum(1 for r in merged_items if r.get("from_cache"))
    meta = {
        "run_id": out_dir.name.removeprefix("run_") if out_dir.name.startswith("run_") else out_dir.name,
        "model_id": meta_a["model_id"],
        "datetime": datetime.datetime.now().isoformat(),
        "n_items": len(merged_items),
        "n_layers_plus_emb": int(meta_a["n_layers_plus_emb"]),
        "d_model": int(meta_a["d_model"]),
        "items_file": "merged",
        "run_tag": run_tag,
        "tok_ids": meta_a.get("tok_ids") or meta_b.get("tok_ids"),
        "merged_from": [run_a.name, run_b.name],
        "merge_key": list(merge_key),
        "pair_key": list(pair_key),
        "merge_order": f"interleaved_by_index: {context_a}, {context_b}",
        "source_meta": {
            run_a.name: meta_a,
            run_b.name: meta_b,
        },
        "n_cache_hits": n_cache,
        "n_forward_passes": len(merged_items) - n_cache,
        "npz_path": npz_written.name if npz_written is not None else None,
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta


def merge_highlight_runs(
    run_dirs: list[Path],
    out_dir: Path,
    *,
    run_tag: str = "highlight_full",
) -> dict[str, Any]:
    if len(run_dirs) < 2:
        raise ValueError("need at least two runs to merge")

    loaded = [_load_run(p) for p in run_dirs]
    items_list = [x[0] for x in loaded]
    hs_list = [x[1] for x in loaded]
    meta_list = [x[2] for x in loaded]

    model_id = meta_list[0].get("model_id")
    if any(m.get("model_id") != model_id for m in meta_list):
        raise ValueError("model_id mismatch between runs")
    if any(hs is None for hs in hs_list):
        raise ValueError("hidden_states.npz missing in one or more runs")

    n_per = len(items_list[0])
    if any(len(items) != n_per for items in items_list):
        raise ValueError("row count mismatch between highlight runs")

    hs0 = hs_list[0]
    assert hs0 is not None
    if any(hs.shape[1:] != hs0.shape[1:] for hs in hs_list if hs is not None):
        raise ValueError("hidden state shape mismatch between runs")

    all_keys: set[tuple[Any, ...]] = set()
    for items in items_list:
        for row in items:
            key = (_highlight_from_row(row), row["context_order"], row["id"])
            if key in all_keys:
                raise ValueError(f"duplicate merge key across runs: {key}")
            all_keys.add(key)

    for i in range(n_per):
        pair = tuple(items_list[0][i][k] for k in PAIR_KEY_HIGHLIGHT)
        for items in items_list[1:]:
            other = tuple(items[i][k] for k in PAIR_KEY_HIGHLIGHT)
            if other != pair:
                raise ValueError(f"pair key mismatch at index {i}: {pair} vs {other}")

    n_total = n_per * len(run_dirs)
    merged_items: list[dict[str, Any]] = []
    merged_hs = np.empty((n_total, *hs0.shape[1:]), dtype=hs0.dtype)
    out_i = 0
    for i in range(n_per):
        for items, hs in zip(items_list, hs_list):
            assert hs is not None
            merged_items.append(items[i])
            merged_hs[out_i] = np.asarray(hs[i])
            out_i += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    merged_ids = np.array([r["id"] for r in merged_items])
    build_path = out_dir / "_hidden_states_building.npz"
    final_path = out_dir / "hidden_states.npz"
    if build_path.is_file():
        build_path.unlink()
    np.savez(build_path, hs=merged_hs, item_ids=merged_ids)
    del merged_hs
    with np.load(build_path) as check:
        if check["hs"].shape[0] != len(merged_items):
            raise ValueError("post-write validation failed: hs row count")
    npz_written: Path = build_path
    try:
        if final_path.is_file():
            os.remove(final_path)
        os.replace(build_path, final_path)
        npz_written = final_path
    except OSError:
        npz_written = build_path

    with (out_dir / "per_item.jsonl").open("w", encoding="utf-8") as f:
        for row in merged_items:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_cache = sum(1 for r in merged_items if r.get("from_cache"))
    meta = {
        "run_id": out_dir.name,
        "model_id": model_id,
        "datetime": datetime.datetime.now().isoformat(),
        "n_items": len(merged_items),
        "n_layers_plus_emb": int(meta_list[0]["n_layers_plus_emb"]),
        "d_model": int(meta_list[0]["d_model"]),
        "items_file": "merged",
        "run_tag": run_tag,
        "tok_ids": meta_list[0].get("tok_ids"),
        "merged_from": [p.name for p in run_dirs],
        "merge_key": list(MERGE_KEY_HIGHLIGHT),
        "pair_key": list(PAIR_KEY_HIGHLIGHT),
        "merge_order": "interleaved_by_index across highlight×context runs",
        "source_meta": {p.name: m for p, m in zip(run_dirs, meta_list)},
        "n_cache_hits": n_cache,
        "n_forward_passes": len(merged_items) - n_cache,
        "npz_path": npz_written.name,
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def _tag_evidence_shift(row: dict[str, Any], evidence_shift: str) -> dict[str, Any]:
    out = dict(row)
    out["evidence_shift"] = evidence_shift
    return out


def _validate_highlight_text(row: dict[str, Any], evidence_shift: str) -> None:
    text = row.get("scenario_text") or row.get("prompt") or ""
    if evidence_shift == "no_evidence":
        if "highlighted the" in text:
            raise ValueError(f"no_evidence row contains highlight prefix: id={row.get('id')!r}")
    elif evidence_shift == "man":
        if "highlighted the man" not in text:
            raise ValueError(f"man evidence row missing highlight man: id={row.get('id')!r}")
    elif evidence_shift == "woman":
        if "highlighted the woman" not in text:
            raise ValueError(f"woman evidence row missing highlight woman: id={row.get('id')!r}")
    else:
        raise ValueError(f"unknown evidence_shift: {evidence_shift!r}")


def merge_h3_evidence_runs(
    sources: list[tuple[Path, str]],
    out_dir: Path,
    *,
    run_tag: str = "h3_evidence_full",
    merge_npz: bool = False,
) -> dict[str, Any]:
    """
    Merge v1 baseline (no_evidence) + highlight MAN/WOMAN runs with evidence_shift tags.

    sources: list of (run_dir, evidence_shift) where evidence_shift is
    no_evidence | man | woman.
    """
    if not sources:
        raise ValueError("need at least one source run")

    loaded: list[tuple[list[dict[str, Any]], np.ndarray | None, dict[str, Any], str]] = []
    for run_dir, ev in sources:
        items, hs, meta = _load_run(run_dir)
        loaded.append((items, hs, meta, ev))

    model_id = loaded[0][2].get("model_id")
    if any(meta.get("model_id") != model_id for _, _, meta, _ in loaded):
        raise ValueError("model_id mismatch between source runs")

    has_hs = [hs is not None for _, hs, _, _ in loaded]
    if merge_npz and not all(has_hs):
        raise ValueError("--merge-npz requested but hidden_states.npz missing in a source run")
    if merge_npz and any(
        hs.shape[1:] != loaded[0][1].shape[1:]  # type: ignore[union-attr]
        for _, hs, _, _ in loaded
        if hs is not None
    ):
        raise ValueError("hidden state shape mismatch between source runs")

    merged_items: list[dict[str, Any]] = []
    merged_hs_chunks: list[np.ndarray] = []
    seen_keys: set[tuple[Any, ...]] = set()
    source_meta: dict[str, Any] = {}

    for (run_dir, ev), (items, hs, meta, _) in zip(sources, loaded):
        for row in items:
            _validate_highlight_text(row, ev)
            tagged = _tag_evidence_shift(row, ev)
            key = _key(tagged, MERGE_KEY_H3_EVIDENCE)
            if key in seen_keys:
                raise ValueError(f"duplicate merge key: {key}")
            seen_keys.add(key)
            merged_items.append(tagged)
        if merge_npz and hs is not None:
            merged_hs_chunks.append(np.asarray(hs))

    for (run_dir, ev), (items, _, meta, _) in zip(sources, loaded):
        rid = str(meta.get("run_id", run_dir.name))
        source_meta[rid] = {
            "run_dir": run_dir.name,
            "evidence_shift": ev,
            "n_items": len(items),
            "model_id": meta.get("model_id"),
            "datetime": meta.get("datetime"),
            "items_file": meta.get("items_file"),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_written: Path | None = None
    if merge_npz and merged_hs_chunks:
        merged_hs = np.concatenate(merged_hs_chunks, axis=0)
        if merged_hs.shape[0] != len(merged_items):
            raise ValueError("hidden_states row count mismatch after merge")
        merged_ids = np.array([r["id"] for r in merged_items])
        build_path = out_dir / "_hidden_states_building.npz"
        final_path = out_dir / "hidden_states.npz"
        if build_path.is_file():
            build_path.unlink()
        np.savez(build_path, hs=merged_hs, item_ids=merged_ids)
        del merged_hs
        npz_written = build_path
        try:
            if final_path.is_file():
                os.remove(final_path)
            os.replace(build_path, final_path)
            npz_written = final_path
        except OSError:
            npz_written = build_path

    with (out_dir / "per_item.jsonl").open("w", encoding="utf-8") as f:
        for row in merged_items:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_cache = sum(1 for r in merged_items if r.get("from_cache"))
    counts_by_ev: dict[str, int] = {}
    for row in merged_items:
        ev = str(row["evidence_shift"])
        counts_by_ev[ev] = counts_by_ev.get(ev, 0) + 1

    meta_out = {
        "run_id": out_dir.name,
        "model_id": model_id,
        "datetime": datetime.datetime.now().isoformat(),
        "n_items": len(merged_items),
        "n_layers_plus_emb": int(loaded[0][2]["n_layers_plus_emb"]),
        "d_model": int(loaded[0][2]["d_model"]),
        "items_file": "merged",
        "run_tag": run_tag,
        "tok_ids": loaded[0][2].get("tok_ids"),
        "merged_from": [run_dir.name for run_dir, _ in sources],
        "evidence_shift_by_source": {run_dir.name: ev for run_dir, ev in sources},
        "evidence_shift_counts": counts_by_ev,
        "merge_key": list(MERGE_KEY_H3_EVIDENCE),
        "pair_key": list(PAIR_KEY_V1),
        "merge_order": "concat: no_evidence block, then MAN×2, then WOMAN×2",
        "source_meta": source_meta,
        "n_cache_hits": n_cache,
        "n_forward_passes": len(merged_items) - n_cache,
        "npz_path": npz_written.name if npz_written is not None else None,
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2, ensure_ascii=False)
    return meta_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", type=Path)
    parser.add_argument("--run-b", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--context-a", default="man_first")
    parser.add_argument("--context-b", default="woman_first")
    parser.add_argument(
        "--run-tag",
        default="v3_full_pos_shuffle",
        help="run_tag in merged meta.json (default: v3_full_pos_shuffle)",
    )
    parser.add_argument(
        "--highlight",
        action="store_true",
        help="merge 4 highlight runs passed via --runs (ignores --run-a/--run-b)",
    )
    parser.add_argument(
        "--h3-evidence",
        action="store_true",
        help="merge v1 baseline + 4 highlight runs with evidence_shift tags",
    )
    parser.add_argument(
        "--merge-npz",
        action="store_true",
        help="concatenate hidden_states.npz when present in all --h3-evidence sources",
    )
    parser.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        help="run directories for --highlight / --h3-evidence merge",
    )
    args = parser.parse_args(argv)

    try:
        if args.h3_evidence:
            repo_results = Path(__file__).resolve().parents[1] / "results"
            if args.runs:
                if len(args.runs) % 2 != 0:
                    raise ValueError(
                        "--runs for --h3-evidence must be pairs: run_dir evidence_shift ..."
                    )
                sources = [
                    (Path(args.runs[i]), str(args.runs[i + 1]))
                    for i in range(0, len(args.runs), 2)
                ]
            else:
                sources = [
                    (repo_results / name, ev) for name, ev in H3_EVIDENCE_DEFAULT_SOURCES
                ]
            meta = merge_h3_evidence_runs(
                sources, args.out, run_tag=args.run_tag, merge_npz=args.merge_npz
            )
        elif args.highlight:
            repo_results = Path(__file__).resolve().parents[1] / "results"
            if args.runs:
                run_dirs = args.runs
            else:
                run_dirs = [repo_results / name for name in HIGHLIGHT_RUN_ORDER]
            meta = merge_highlight_runs(run_dirs, args.out, run_tag=args.run_tag)
        else:
            if args.run_a is None or args.run_b is None:
                raise ValueError("--run-a and --run-b are required unless --highlight is set")
            meta = merge_runs(
                args.run_a,
                args.run_b,
                args.out,
                context_a=args.context_a,
                context_b=args.context_b,
                run_tag=args.run_tag,
            )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Merged {meta['n_items']} items → {args.out}")
    npz_path = args.out / "hidden_states.npz"
    if npz_path.is_file():
        print(f"  hidden_states.npz: {npz_path.stat().st_size / 1e6:.1f} MB")
    else:
        print("  hidden_states.npz: (not merged — absent in source runs)")
    print(f"  cache hits: {meta['n_cache_hits']}, forward passes: {meta['n_forward_passes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
