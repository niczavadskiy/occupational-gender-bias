"""
Stage A для contrastive steering — тот же раннер, что H1, другие дефолты.

Нужны уже собранные векторы:
    python -m steering.contrastive.build_vectors --scale 2b --device cuda

Примеры:
    python -m steering.contrastive.run_stagea --device cuda --tag contrastive_2b_a_v1
    python -m steering.contrastive.run_stagea --scale 4b --device cuda --tag contrastive_4b_a_v1
    python -m steering.contrastive.run_stagea --limit-items 3 --tag smoke \\
        --candidates main_center_core__vmd__L16__center__a1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from steering.run_h1_stagea import main as stagea_main

HERE = Path(__file__).resolve().parent
STEERING_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]

PATHS = {
    "2b": {
        "candidates": HERE / "candidates" / "contrastive_gender_2b_candidates_v1.json",
        "vectors": HERE / "vectors" / "contrastive_gender_2b_vectors_v1.npz",
        "sample": STEERING_DIR / "samples" / "h1_stagea_sample_v1.json",
        "model": "Qwen/Qwen3.5-2B-Base",
    },
    "4b": {
        "candidates": HERE / "candidates" / "contrastive_gender_4b_candidates_v1.json",
        "vectors": HERE / "vectors" / "contrastive_gender_4b_vectors_v1.npz",
        "sample": STEERING_DIR / "samples" / "h1_stagea_sample_v1.json",
        "model": "Qwen/Qwen3.5-4B-Base",
    },
}


def _has_flag(argv: list[str], flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in argv)


def main(argv: list[str] | None = None) -> int:
    raw = list(argv if argv is not None else sys.argv[1:])
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--scale", choices=["2b", "4b"], default="2b")
    known, rest = pre.parse_known_args(raw)
    p = PATHS[known.scale]
    injected: list[str] = []
    defaults = {
        "--model": p["model"],
        "--sample": str(p["sample"]),
        "--candidates-file": str(p["candidates"]),
        "--vectors": str(p["vectors"]),
        "--out-root": str(REPO_ROOT / "results" / "steering" / "contrastive"),
        "--tag": f"contrastive_{known.scale}_a_v1",
    }
    for flag, val in defaults.items():
        if not _has_flag(rest, flag):
            injected.extend([flag, val])
    want_help = any(a in ("-h", "--help") for a in raw)
    if not want_help and not p["vectors"].is_file() and not _has_flag(rest, "--vectors"):
        raise SystemExit(
            f"нет векторов {p['vectors']}\n"
            f"сначала: python -m steering.contrastive.build_vectors --scale {known.scale}"
        )
    if not want_help and not p["candidates"].is_file() and not _has_flag(rest, "--candidates-file"):
        raise SystemExit(
            f"нет каталога {p['candidates']}\n"
            f"сначала: python -m steering.contrastive.build_candidates --scale {known.scale}"
        )
    return stagea_main(injected + rest)


if __name__ == "__main__":
    sys.exit(main())
