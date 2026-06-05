"""CLI: python -m src.metrics --run results/<run>/"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.metrics.load import resolve_run_dir
from src.metrics.pipeline import DEFAULT_HYPOTHESES, run_pipeline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Behavioral metrics H1–H9 from per_item.jsonl")
    p.add_argument(
        "--run",
        "--run-dir",
        "--input",
        dest="run",
        default=None,
        metavar="PATH",
        help=(
            "Run directory under results/ (e.g. 5400, results/5400), "
            "absolute path to run dir, or path to per_item.jsonl "
            f"(default: canonical 1350 run)"
        ),
    )
    p.add_argument(
        "--hypotheses",
        default=",".join(DEFAULT_HYPOTHESES),
        help="Comma-separated hypothesis ids (e.g. H1,H3)",
    )
    p.add_argument("--fdr", type=float, default=0.05, help="Benjamini–Hochberg FDR level")
    p.add_argument(
        "--level",
        choices=("row", "family"),
        default="family",
        help="Primary clustering for H1/H3 sensitivity (family recommended)",
    )
    p.add_argument("--out", default=None, help="Output directory (default: run/metrics/<timestamp>)")
    p.add_argument(
        "--annotations",
        default=None,
        help="Path to annotations.jsonl, annotation run dir, or run_id under data/annotation/runs/",
    )
    p.add_argument(
        "--h4-formats",
        default=None,
        help=(
            "Optional H4 per-format strata: choice, yesno (=yesno_man+yesno_woman), "
            "yesno_man, yesno_woman, all, or comma-separated list. "
            "Default: one pooled test over all formats (10800 answerability rows)."
        ),
    )
    p.add_argument(
        "--position-variant",
        default="auto",
        help="Option-position slice: auto/all (default, keep all positions), or explicit e.g. p0, canonical",
    )
    p.add_argument(
        "--context-order",
        default="auto",
        help="Context order slice: auto/all (default, keep all), or man_first, woman_first",
    )
    args = p.parse_args(argv)

    hypo = [h.strip().upper() for h in args.hypotheses.split(",") if h.strip()]
    out = Path(args.out) if args.out else None
    ann = Path(args.annotations) if args.annotations else None

    run_dir = resolve_run_dir(args.run)
    print(f"Input: {run_dir / 'per_item.jsonl'}")

    pos_kw = None if args.position_variant == "auto" else args.position_variant
    ctx_kw = None if args.context_order == "auto" else args.context_order

    out_dir = run_pipeline(
        run_dir,
        hypotheses=hypo,
        fdr=args.fdr,
        level=args.level,
        out_dir=out,
        annotation_path=ann,
        h4_formats=args.h4_formats,
        position_variant=pos_kw,
        context_order=ctx_kw,
    )
    print(f"Wrote metrics to {out_dir}")
    print(f"  summary: {out_dir / 'summary.md'}")
    print(f"  tests:   {out_dir / 'tests_all.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
