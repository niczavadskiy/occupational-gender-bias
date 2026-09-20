"""Frozen H1 SOC-FDR gate for per-domain XY-control.

A domain is steered only if H1 rejected FDR on that soc_major_title
(choice family and/or prob family, without_abstain). Polarity is the
sign of the significant effect: male if P(man)>P(woman), else female.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]

SOC_CSV = {
    "2b": REPO_ROOT
    / "results"
    / "v1_full_pos_shuffle"
    / "metrics_h1_v1"
    / "latest"
    / "tests_h1_v1_soc.csv",
    "4b": REPO_ROOT
    / "experiments"
    / "qwen35-4b-base"
    / "results"
    / "qwen35_4b_h1h3_pack"
    / "run_2026-09-12_20-25-31_Qwen3.5-4B-Base_v1_full_pos_shuffle"
    / "metrics_h1_v1"
    / "latest"
    / "tests_h1_v1_soc.csv",
}

CATALOG_JSON = HERE / "domains" / "h1_soc_fdr_v1.json"
CATALOG_MD = HERE / "domains" / "h1_soc_fdr_v1.md"

CHOICE_KIND = "man_vs_woman"
PROB_KIND = "mean_margin_man_minus_woman"
ABSTAIN = "without_abstain"


def slug_soc(title: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    if len(clean) > 48:
        clean = clean[:48].rstrip("_")
    if not clean:
        raise ValueError(f"empty slug for {title!r}")
    return clean


def _as_bool(raw: str) -> bool:
    return str(raw).strip().lower() == "true"


def _as_float(raw: str) -> float:
    text = str(raw).strip()
    if not text:
        return float("nan")
    return float(text)


def parse_soc_csv(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """title -> {choice|prob -> fields}."""
    if not path.is_file():
        raise FileNotFoundError(path)
    by_title: dict[str, dict[str, dict[str, Any]]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("extra_abstain_variant") or "") != ABSTAIN:
                continue
            title = (row.get("extra_soc_major_title") or "").strip()
            kind = (row.get("extra_comparison") or "").strip()
            if not title or kind not in {CHOICE_KIND, PROB_KIND}:
                continue
            key = "choice" if kind == CHOICE_KIND else "prob"
            by_title.setdefault(title, {})[key] = {
                "rejected_fdr": _as_bool(row.get("rejected_fdr", "")),
                "q_value": _as_float(row.get("q_value", "")),
                "p_raw": _as_float(row.get("p_raw", "")),
                "effect": _as_float(row.get("effect", "")),
                "test_id": row.get("test_id", ""),
                "family_name": row.get("family_name", ""),
                "n_clusters": _as_float(row.get("extra_stratum_g") or row.get("n1") or ""),
            }
    return by_title


def polarity_of(effect: float) -> str:
    if effect > 0:
        return "male"
    if effect < 0:
        return "female"
    return "none"


def hypothesized_alpha_sign(polarity: str) -> str:
    """Prior from global 2B: +v is pro-female, so male domains want α<0 (add v)."""
    if polarity == "male":
        return "negative"
    if polarity == "female":
        return "positive"
    return "unknown"


def select_domain(title: str, tests: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    choice = tests.get("choice") or {}
    prob = tests.get("prob") or {}
    ch_rej = bool(choice.get("rejected_fdr"))
    pr_rej = bool(prob.get("rejected_fdr"))
    if not ch_rej and not pr_rej:
        return None
    if ch_rej:
        gate = "both" if pr_rej else "choice_fdr"
        effect = float(choice["effect"])
    else:
        gate = "prob_fdr"
        effect = float(prob["effect"])
    polarity = polarity_of(effect)
    return {
        "soc_major_title": title,
        "slug": slug_soc(title),
        "polarity": polarity,
        "gate": gate,
        "effect": effect,
        "hypothesized_alpha_sign": hypothesized_alpha_sign(polarity),
        "choice": choice,
        "prob": prob,
        "steer": True,
    }


def domains_for_scale(
    by_title: dict[str, dict[str, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for title in sorted(by_title):
        selected = select_domain(title, by_title[title])
        if selected:
            out.append(selected)
        else:
            skipped.append(
                {
                    "soc_major_title": title,
                    "slug": slug_soc(title),
                    "steer": False,
                    "choice": by_title[title].get("choice") or {},
                    "prob": by_title[title].get("prob") or {},
                }
            )
    slugs = [d["slug"] for d in out]
    if len(slugs) != len(set(slugs)):
        raise ValueError(f"slug collision: {slugs}")
    return out, skipped


def load_catalog(path: Path | None = None) -> dict[str, Any]:
    p = path or CATALOG_JSON
    import json

    return json.loads(p.read_text(encoding="utf-8"))


def significant_titles(catalog: dict[str, Any], scale: str) -> list[str]:
    return [d["soc_major_title"] for d in catalog["scales"][scale]["steer"]]


def domain_by_title(catalog: dict[str, Any], scale: str, title: str) -> dict[str, Any] | None:
    for d in catalog["scales"][scale]["steer"]:
        if d["soc_major_title"] == title:
            return d
    return None
