"""
INLP Stage C (report): preference на остатке test + capability на domain_test.

Победитель Stage B не переизбирается — только подтверждается на held-out:
  - preference: `inlp_stagec_sample_v1.json` (95 test-семей ∉ confirmatory)
  - capability: `mmlu_pro_domain_test_v1` (вопросы дизъюнктны с domain_val)

Shortlist по умолчанию — keep Stage B (k8/k16 α=1), без random.

Примеры:
    python -m steering.build_stageb_sample \\
        --stage-a steering/samples/inlp_test_sample_v1.json \\
        --out-stem inlp_stagec_sample_v1 \\
        --title "INLP Stage C sample" \\
        --capability-note "Preference Stage C; capability — mmlu_pro_domain_test_v1."

    python -m steering.run_inlp_stagec --device cuda --dtype float32 --tag inlp_c_v1

    # smoke
    python -m steering.run_inlp_stagec --device cuda --dtype float32 \\
        --tag inlp_c_smoke --limit-items 2 --limit-mmlu 20 \\
        --candidates inlp__L15__k8__a1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from steering.run_inlp_stagea import DEFAULT_SUBSPACES
from steering.run_inlp_stageb import run_from_args

REPO_ROOT = Path(__file__).resolve().parents[1]
STEERING_DIR = Path(__file__).resolve().parent
DEFAULT_SHORTLIST = STEERING_DIR / "candidates" / "inlp_stagec_keep_v1.json"
DEFAULT_SAMPLE = STEERING_DIR / "samples" / "inlp_stagec_sample_v1.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--subspaces", type=Path, default=DEFAULT_SUBSPACES)
    ap.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    ap.add_argument("--shortlist", type=Path, default=DEFAULT_SHORTLIST)
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--roles", default="candidate", help="default: только candidate (без random)")
    ap.add_argument("--layers", default="15")
    ap.add_argument("--ranks", default="8,16")
    ap.add_argument("--alphas", default="1.0")
    ap.add_argument("--no-random-control", action="store_true", default=True)
    ap.add_argument(
        "--with-random-control",
        action="store_true",
        help="включить random в build_configs (нужен shortlist с control)",
    )
    ap.add_argument("--random-seeds", default="0")
    ap.add_argument(
        "--skip-preference",
        action="store_true",
        help="только capability (по умолчанию preference включён)",
    )
    ap.add_argument("--skip-capability", action="store_true")
    ap.add_argument(
        "--with-smoke",
        action="store_true",
        help="добавить overall_smoke (по умолчанию только domain_test)",
    )
    ap.add_argument("--skip-baseline", action="store_true")
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
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260820)
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "results" / "steering")
    ap.add_argument("--tag", default="inlp_c_v1")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--cap-loss-max", type=float, default=0.03)
    args = ap.parse_args(argv)

    # Normalize flags to stageb run_from_args contract
    args.with_preference = not args.skip_preference
    args.skip_smoke = not args.with_smoke
    args.no_random_control = not args.with_random_control
    if args.with_random_control and args.roles == "candidate":
        args.roles = None

    return run_from_args(
        args,
        stage_label="INLP Stage C",
        out_subdir="inlp_stage_c",
        schema="steering.stage_c_inlp_run/v1",
        reselect_keep=False,
    )


if __name__ == "__main__":
    sys.exit(main())
