"""CLI: python -m src.metrics --run results/<run>/"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.metrics.pipeline import DEFAULT_HYPOTHESES, run_pipeline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Behavioral metrics H1–H9 from per_item.jsonl")
    p.add_argument(
        "--run",
        default=None,
        help="Inference run dir (default: canonical factorial v2 run)",
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
    args = p.parse_args(argv)

    hypo = [h.strip().upper() for h in args.hypotheses.split(",") if h.strip()]
    out = Path(args.out) if args.out else None
    ann = Path(args.annotations) if args.annotations else None

    out_dir = run_pipeline(
        args.run,
        hypotheses=hypo,
        fdr=args.fdr,
        level=args.level,
        out_dir=out,
        annotation_path=ann,
    )
    print(f"Wrote metrics to {out_dir}")
    print(f"  summary: {out_dir / 'summary.md'}")
    print(f"  tests:   {out_dir / 'tests_all.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
