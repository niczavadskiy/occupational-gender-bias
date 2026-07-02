"""CLI: python -m src.metrics.h1_v1 --run results/<v1_run>/"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.metrics.h1_v1.pipeline import run_h1_v1_pipeline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="H1 metrics for v1 occupation-based per_item.jsonl")
    p.add_argument(
        "--run",
        "--run-dir",
        "--input",
        dest="run",
        default=None,
        metavar="PATH",
        help="Run directory or per_item.jsonl path",
    )
    p.add_argument("--fdr", type=float, default=0.05, help="Target FDR level Q for Benjamini–Hochberg")
    p.add_argument("--out", default=None, help="Output directory (default: run/metrics_h1_v1/<timestamp>)")
    p.add_argument(
        "--min-stratum-n",
        type=int,
        default=14,
        help="Minimum rows per stratum (default: 14 ≈ one v1 scenario block)",
    )
    p.add_argument(
        "--min-soc-g",
        type=int,
        default=15,
        help="Minimum base items (scenario_family_id) per soc_major_title stratum (default: 15)",
    )
    p.add_argument("--no-soc-strata", action="store_true", help="Skip per soc_major_title tests")
    p.add_argument("--no-onet-strata", action="store_true", help="Skip per onet_action tests")
    p.add_argument(
        "--position-variant",
        default="auto",
        help="Position slice: auto/all (default) or e.g. p0",
    )
    p.add_argument(
        "--context-order",
        default="auto",
        help="Context order slice: auto/all (default) or man_first, woman_first",
    )
    args = p.parse_args(argv)

    pos_kw = None if args.position_variant == "auto" else args.position_variant
    ctx_kw = None if args.context_order == "auto" else args.context_order
    out = Path(args.out) if args.out else None

    try:
        out_dir = run_h1_v1_pipeline(
            args.run,
            fdr=args.fdr,
            out_dir=out,
            min_stratum_n=args.min_stratum_n,
            min_soc_g=args.min_soc_g,
            include_soc_strata=not args.no_soc_strata,
            include_onet_strata=not args.no_onet_strata,
            position_variant=pos_kw,
            context_order=ctx_kw,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote h1_v1 metrics to {out_dir}")
    print(f"  summary: {out_dir / 'summary.md'}")
    print(f"  tests:   {out_dir / 'tests_all.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
