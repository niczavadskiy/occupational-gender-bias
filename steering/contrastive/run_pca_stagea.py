"""
Stage A bake-off for contrastive PCA subspace V_k.

k=1 is v_G (mean-diff). k>1 adds unit-centered SOC PCs. Same runner as INLP
(center: h' = h − α V(Vᵀh − c)).

Нужен npz из analyze_rank:

    python -m steering.contrastive.analyze_rank --scale 2b --device cuda
    python -m steering.contrastive.run_pca_stagea --device cuda --tag pca_2b_a_v1
    python -m steering.contrastive.run_pca_stagea --scale 4b --layers 23 --ranks 1,2,3,4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from steering.run_inlp_stagea import main as inlp_stagea_main

HERE = Path(__file__).resolve().parent
STEERING_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]

PATHS = {
    "2b": {
        "subspaces": STEERING_DIR / "subspaces" / "contrastive_pca_2b_v1.npz",
        "sample": STEERING_DIR / "samples" / "h1_stagea_sample_v1.json",
        "model": "Qwen/Qwen3.5-2B-Base",
        "layers": "16",
    },
    "4b": {
        "subspaces": STEERING_DIR / "subspaces" / "contrastive_pca_4b_v1.npz",
        "sample": STEERING_DIR / "samples" / "h1_stagea_sample_v1.json",
        "model": "Qwen/Qwen3.5-4B-Base",
        "layers": "23",
    },
}


def _has_flag(argv: list[str], flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in argv)


def main(argv: list[str] | None = None) -> int:
    raw = list(argv if argv is not None else sys.argv[1:])
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--scale", choices=["2b", "4b"], default="2b")
    pre.add_argument("--subspaces", type=Path, default=None)
    known, rest = pre.parse_known_args(raw)
    p = PATHS[known.scale]
    sub = known.subspaces or p["subspaces"]
    injected: list[str] = []
    defaults = {
        "--model": p["model"],
        "--sample": str(p["sample"]),
        "--subspaces": str(sub),
        "--out-root": str(REPO_ROOT / "results" / "steering" / "contrastive_pca"),
        "--tag": f"pca_{known.scale}_a_v1",
        "--layers": p["layers"],
        "--ranks": "1,2,3,4",
        "--alphas": "1.0",
        "--primary-axis": "gender",
    }
    for flag, val in defaults.items():
        if not _has_flag(rest, flag):
            injected.extend([flag, val])
    want_help = any(a in ("-h", "--help") for a in raw)
    if not want_help and not Path(sub).is_file():
        raise SystemExit(
            f"нет подпространства {sub}\n"
            f"сначала: python -m steering.contrastive.analyze_rank --scale {known.scale}"
        )
    return inlp_stagea_main(injected + rest)


if __name__ == "__main__":
    sys.exit(main())
