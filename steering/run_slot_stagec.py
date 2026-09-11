"""
Slot Stage C (report): preference на test + capability на domain_test.

Победитель Stage B не переизбирается — только подтверждается:
  - preference: `slot_stagec_sample_v1.json` (95 test-семей, seed 20260911)
  - capability: `mmlu_pro_domain_test_v1`

Shortlist: keep Stage B (`slot_stagec_keep_v1.json`).

Примеры:
    python -m steering.build_stagea_sample --source-split test --n-base-items 95 \\
        --seed 20260911 --out-stem slot_stagec_sample_v1

    python -m steering.run_slot_stagec --device cuda --dtype float32 --tag slot_c_v1

    python -m steering.run_slot_stagec --device cuda --dtype float32 \\
        --tag slot_c_smoke --limit-items 2 --limit-mmlu 20 \\
        --candidates main_center_core__wslot__L23__center__a2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from steering.run_slot_stageb import run_from_args

REPO_ROOT = Path(__file__).resolve().parents[1]
STEERING_DIR = Path(__file__).resolve().parent
DEFAULT_SHORTLIST = STEERING_DIR / "candidates" / "slot_stagec_keep_v1.json"
DEFAULT_SAMPLE = STEERING_DIR / "samples" / "slot_stagec_sample_v1.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    ap.add_argument(
        "--candidates-file",
        type=Path,
        default=STEERING_DIR / "candidates" / "slot_candidates_v1.json",
    )
    ap.add_argument("--vectors", type=Path, default=STEERING_DIR / "vectors" / "slot_vectors_v1.npz")
    ap.add_argument("--shortlist", type=Path, default=DEFAULT_SHORTLIST)
    ap.add_argument("--from-ranking", type=Path, default=None)
    ap.add_argument("--keep-top", type=int, default=15)
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--roles", default="candidate")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--skip-preference", action="store_true")
    ap.add_argument("--skip-capability", action="store_true")
    ap.add_argument(
        "--with-smoke",
        action="store_true",
        help="добавить overall_smoke (по умолчанию только domain_test)",
    )
    ap.add_argument("--limit-items", type=int, default=None)
    ap.add_argument("--limit-mmlu", type=int, default=None)
    ap.add_argument(
        "--domain-profile",
        type=Path,
        default=STEERING_DIR / "profiles" / "mmlu_pro_domain_test_v1.json",
    )
    ap.add_argument(
        "--smoke-profile",
        type=Path,
        default=STEERING_DIR / "profiles" / "mmlu_pro_overall_smoke_v1.json",
    )
    ap.add_argument(
        "--mmlu-parquet",
        type=Path,
        default=STEERING_DIR / ".cache" / "mmlu_pro_test.parquet",
    )
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "results" / "steering")
    ap.add_argument("--tag", default="slot_c_v1")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--cap-loss-max", type=float, default=0.03)
    args = ap.parse_args(argv)
    args.skip_smoke = not args.with_smoke

    return run_from_args(
        args,
        stage_label="Stage C slot",
        out_subdir="stage_c",
        schema="steering.stage_c_slot_run/v1",
        reselect_keep=False,
    )


if __name__ == "__main__":
    sys.exit(main())
