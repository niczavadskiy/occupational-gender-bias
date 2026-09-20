"""Freeze H1 SOC-FDR significant domains for per-SOC XY-control.

    python -m steering.xy_control.build_domains
    python -m steering.xy_control.build_domains --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from steering.xy_control.domains import (
    CATALOG_JSON,
    CATALOG_MD,
    SOC_CSV,
    domains_for_scale,
    parse_soc_csv,
)

HERE = Path(__file__).resolve().parent


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    return obj


def build_catalog() -> dict:
    scales = {}
    for scale, path in SOC_CSV.items():
        by_title = parse_soc_csv(path)
        steer, skip = domains_for_scale(by_title)
        scales[scale] = {
            "csv": _rel(path),
            "n_soc_tested": len(by_title),
            "n_steer": len(steer),
            "n_skip": len(skip),
            "n_male": sum(1 for d in steer if d["polarity"] == "male"),
            "n_female": sum(1 for d in steer if d["polarity"] == "female"),
            "steer": steer,
            "skip": skip,
        }
    return {
        "schema": "steering.xy_control_soc_fdr/v1",
        "version": "v1",
        "datetime": datetime.now().replace(microsecond=0).isoformat(),
        "rule": {
            "abstain_variant": "without_abstain",
            "include": "union of FDR-rejected H1 SOC strata (choice family or prob family)",
            "polarity": "sign of the significant effect; choice preferred if both reject",
            "hypothesized_alpha_sign": (
                "male → α<0 (add v); female → α>0 (subtract v). "
                "Both signs are still gridded; this is only the prior."
            ),
            "skip": "non-significant SOCs are not steered (identity)",
        },
        "scales": scales,
    }


def _rel(path: Path) -> str:
    try:
        return path.relative_to(HERE.parents[1]).as_posix()
    except ValueError:
        return path.as_posix()


def catalog_markdown(doc: dict) -> str:
    lines = [
        "# H1 SOC-FDR domains — XY-control per-SOC",
        "",
        "Steer only FDR-significant `soc_major_title` (without_abstain). "
        "Insignificant domains are skipped.",
        "",
        f"- Rule: `{doc['rule']['include']}`",
        f"- Polarity: {doc['rule']['polarity']}",
        f"- α prior: {doc['rule']['hypothesized_alpha_sign']}",
        "",
    ]
    for scale, block in doc["scales"].items():
        lines.append(f"## {scale.upper()}")
        lines.append("")
        lines.append(
            f"Tested {block['n_soc_tested']} SOCs → steer **{block['n_steer']}** "
            f"({block['n_male']} male, {block['n_female']} female), skip {block['n_skip']}."
        )
        lines.append("")
        lines.append("| polarity | gate | effect | q_choice | q_prob | SOC |")
        lines.append("|---|---|---:|---:|---:|---|")
        for d in block["steer"]:
            qch = d.get("choice", {}).get("q_value")
            qpr = d.get("prob", {}).get("q_value")
            qch_s = f"{qch:.3g}" if isinstance(qch, float) else "—"
            qpr_s = f"{qpr:.3g}" if isinstance(qpr, float) else "—"
            lines.append(
                f"| {d['polarity']} | {d['gate']} | {d['effect']:+.3f} | {qch_s} | {qpr_s} | {d['soc_major_title']} |"
            )
        lines.append("")
        if block["skip"]:
            names = ", ".join(d["soc_major_title"] for d in block["skip"])
            lines.append(f"Skip: {names}.")
            lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)

    doc = _jsonable(build_catalog())
    # freeze datetime on verify by comparing without datetime
    md = catalog_markdown(doc)
    payload = json.dumps(doc, ensure_ascii=False, indent=2) + "\n"

    if args.verify:
        if not CATALOG_JSON.is_file() or not CATALOG_MD.is_file():
            print("MISSING catalog files")
            return 1
        on_disk = json.loads(CATALOG_JSON.read_text(encoding="utf-8"))
        a, b = dict(on_disk), dict(doc)
        a.pop("datetime", None)
        b.pop("datetime", None)
        bad = []
        if a != b:
            bad.append(CATALOG_JSON.name)
        if CATALOG_MD.read_text(encoding="utf-8") != md:
            bad.append(CATALOG_MD.name)
        if bad:
            print("MISMATCH: " + ", ".join(bad))
            return 1
        print("OK — SOC-FDR domain catalog matches rebuild")
        return 0

    CATALOG_JSON.parent.mkdir(parents=True, exist_ok=True)
    CATALOG_JSON.write_text(payload, encoding="utf-8")
    CATALOG_MD.write_text(md, encoding="utf-8")
    for scale, block in doc["scales"].items():
        print(
            f"  {scale}: steer {block['n_steer']} "
            f"(male {block['n_male']}, female {block['n_female']})  skip {block['n_skip']}"
        )
        for d in block["steer"]:
            print(f"    {d['polarity']:<7} {d['gate']:<10} {d['effect']:+.3f}  {d['soc_major_title']}")
    print(f"  -> {CATALOG_JSON}")
    print(f"  -> {CATALOG_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
