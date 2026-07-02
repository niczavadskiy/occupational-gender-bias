"""One-off: assemble variant-A publish tree from the main repo."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parent / "repo"
RUN_SRC = "run_2026-06-09_12-50-33_Qwen3.5-2B-Base_v1_full_pos_shuffle"
RUN_DST = "v1_full_pos_shuffle"
H11_PROBE_SRC = (
    f"results/{RUN_SRC}/probes/v1_rep_prob/"
    "without_abstain_p0_p1_mf_wf_gender_choice"
)
H11_PROBE_DST = f"results/{RUN_DST}/probes/h11_gender_choice"

METRICS_SHARED = [
    "src/__init__.py",
    "src/metrics/__init__.py",
    "src/metrics/fdr.py",
    "src/metrics/stats_tests.py",
    "src/metrics/gender_probs.py",
    "src/metrics/load.py",
    "src/metrics/labels.py",
    "src/metrics/report.py",
    "src/metrics/reliability.py",
]
METRICS_DIRS = ["src/metrics/h1_v1", "src/metrics/h3_v1"]

COPY_TREE = [
    ("analysis/combined_h1_h3_h5_h11.html", "analysis/combined_h1_h3_h5_h11.html"),
    ("analysis/build_combined_report.py", "analysis/build_combined_report.py"),
    ("results/highlight-h3-full/metrics_h3_v1/latest", "results/highlight-h3-full/metrics_h3_v1/latest"),
    (f"results/{RUN_SRC}/metrics_h1_v1/latest", f"results/{RUN_DST}/metrics_h1_v1/latest"),
    (f"results/{RUN_SRC}/probes/v1_rep_prob/latest", f"results/{RUN_DST}/probes/v1_rep_prob/latest"),
    (H11_PROBE_SRC, H11_PROBE_DST),
]

PROBE_SKIP_SUFFIX = {".npz", ".log"}


def sanitize_meta(path: Path) -> None:
    if path.name != "meta.json" or not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    if "run_dir" in data and isinstance(data["run_dir"], str):
        data["run_dir"] = Path(data["run_dir"]).name
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def copy_probe_tree(src: Path, dst: Path) -> None:
    for p in src.rglob("*"):
        if p.is_dir():
            continue
        if p.suffix.lower() in PROBE_SKIP_SUFFIX:
            continue
        if p.suffix.lower() not in {".json", ".csv", ".md", ".html"}:
            continue
        rel = p.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        sanitize_meta(out)


def main() -> None:
    preserve = {"README.md", "LICENSE", "requirements.txt", ".gitignore", "scripts", ".git"}
    if ROOT.exists():
        for child in list(ROOT.iterdir()):
            if child.name in preserve:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    ROOT.mkdir(parents=True, exist_ok=True)

    for rel in METRICS_SHARED:
        src = SRC / rel
        dst = ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    for rel in METRICS_DIRS:
        shutil.copytree(SRC / rel, ROOT / rel)

    for src_rel, dst_rel in COPY_TREE:
        src = SRC / src_rel
        dst = ROOT / dst_rel
        if src.is_dir():
            if "probes" in src_rel:
                copy_probe_tree(src, dst)
            else:
                shutil.copytree(src, dst)
                for meta in dst.rglob("meta.json"):
                    sanitize_meta(meta)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    docs = ROOT / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "analysis/combined_h1_h3_h5_h11.html", docs / "index.html")
    (docs / ".nojekyll").touch()

    print(f"Built publish tree at {ROOT}")


if __name__ == "__main__":
    main()
