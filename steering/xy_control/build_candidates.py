"""
Разворачивание сетки XY-control steering-кандидатов в замороженный каталог.

    python -m steering.xy_control.build_candidates
    python -m steering.xy_control.build_candidates --scale 4b
    python -m steering.xy_control.build_candidates --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from steering.build_h1_candidates import build_document, load_config, summary_markdown

HERE = Path(__file__).resolve().parent
CONFIG_BY_SCALE = {
    "2b": HERE / "configs" / "xy_control_2b_v1.yaml",
    "4b": HERE / "configs" / "xy_control_4b_v1.yaml",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", choices=["2b", "4b"], default="2b")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=HERE / "candidates")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)

    cfg_path = args.config or CONFIG_BY_SCALE[args.scale]
    cfg = load_config(cfg_path)
    scale = str(cfg.get("scale", args.scale))
    ver = cfg["version"]
    doc = build_document(cfg, cfg_path)
    md = summary_markdown(doc)

    prefix = f"xy_control_{scale}_candidates"
    json_path = args.out_dir / f"{prefix}_{ver}.json"
    md_path = args.out_dir / f"{prefix}_{ver}.md"

    if args.verify:
        expected = json.dumps(doc, ensure_ascii=False, indent=2) + "\n"
        bad = []
        if not json_path.exists() or json_path.read_text(encoding="utf-8") != expected:
            bad.append(json_path.name)
        if not md_path.exists() or md_path.read_text(encoding="utf-8") != md:
            bad.append(md_path.name)
        if bad:
            print("MISMATCH: " + ", ".join(bad))
            return 1
        print("OK — XY-control candidates совпадают с пересборкой")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(md, encoding="utf-8")

    print(
        f"xy_control {scale} {ver}: {doc['n_candidates']} candidates "
        f"(candidate {doc['n_by_role'].get('candidate', 0)}, "
        f"control {doc['n_by_role'].get('control', 0)})"
    )
    for fam, n in doc["n_by_family"].items():
        print(f"  {fam:<22} {n:>3}")
    print(f"  -> {json_path.name}")
    print(f"  -> {md_path.name}")
    print(f"  Stage A forwards ≈ {doc['forwards_estimate']['total_with_baseline']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
