"""Pooled polarity INLP: one subspace / one (L,k) per pro-male or pro-female set.

Matched to XY polarity pool: same SOC lists, same family_split3_by_soc
(seed 0, 70/15/15), candidate layers = probe peak + peak−1 + peak−2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from steering.xy_control.domains import load_catalog
from steering.xy_control.polarity_pool import (
    catalog_domains_for_set,
    make_pool_domain,
    plan_pool_split,
    pool_slug,
)

STEERING_DIR = Path(__file__).resolve().parent
SETS_JSON = STEERING_DIR / "domains" / "inlp_polarity_sets_2b_v1.json"
DEFAULT_CONFIG = STEERING_DIR / "configs" / "inlp_polarity_pool_2b_peak_prepeak.yaml"
XY_PAIRS = STEERING_DIR / "xy_control" / "data" / "xy_pairs_full_v1.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_polarity_sets(path: Path | None = None) -> dict[str, Any]:
    return load_json(path or SETS_JSON)


def get_set(doc: dict[str, Any], set_id: str) -> dict[str, Any]:
    for s in doc["sets"]:
        if s["id"] == set_id:
            return s
    raise KeyError(f"unknown polarity set id: {set_id}")


def resolve_peak_layers(cfg: dict[str, Any]) -> list[int]:
    """Primary grid: peak, then peak−1, peak−2 (order preserved)."""
    layers_block = cfg.get("layers") or {}
    if "belt" in layers_block and layers_block["belt"]:
        return [int(x) for x in layers_block["belt"]]
    peak = int(layers_block.get("peak", layers_block.get("anchor", 16)))
    pre = [int(x) for x in layers_block.get("prepeak", [peak - 1, peak - 2])]
    out = [peak]
    for L in pre:
        if L not in out:
            out.append(L)
    return out


def resolve_ranks(cfg: dict[str, Any]) -> list[int]:
    return [int(r) for r in (cfg.get("steering") or {}).get("ranks", [4, 8, 16])]


def resolve_alphas(cfg: dict[str, Any]) -> list[float]:
    return [float(a) for a in (cfg.get("steering") or {}).get("alphas", [1.0])]


def expand_inlp_pool_candidates(
    cfg: dict[str, Any],
    *,
    smoke: bool = False,
    smoke_rank: int = 4,
) -> list[dict[str, Any]]:
    """Build Stage A candidate ids: inlp__L{ℓ}__k{r}__a{α}."""
    layers = resolve_peak_layers(cfg)
    ranks = resolve_ranks(cfg)
    alphas = resolve_alphas(cfg)
    if smoke:
        layers = [int((cfg.get("layers") or {}).get("anchor", layers[0]))]
        ranks = [smoke_rank]
        alphas = [1.0]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for layer in layers:
        for rank in ranks:
            for alpha in alphas:
                # match run_inlp_stagea / inlp_shortlist CONFIG_ID_RE
                a_s = str(alpha).replace(".", "p")
                if a_s.endswith("p0"):
                    a_s = a_s[:-2]
                cid = f"inlp__L{int(layer)}__k{int(rank)}__a{a_s}"
                if cid in seen:
                    continue
                seen.add(cid)
                out.append(
                    {
                        "id": cid,
                        "basis": "inlp",
                        "layer": int(layer),
                        "rank": int(rank),
                        "alpha": float(alpha),
                        "role": "candidate",
                    }
                )
    return out


def load_xy_pool_items(
    set_entry: dict[str, Any], *, scale: str = "2b"
) -> tuple[list[dict], list[dict]]:
    """Gendered H1-shaped items for member SOCs from xy_pairs_full."""
    catalog = load_catalog()
    soc_domains = catalog_domains_for_set(
        catalog, scale=scale, slugs=list(set_entry["slugs"])
    )
    titles = {d["soc_major_title"] for d in soc_domains}
    pairs = load_json(XY_PAIRS)
    items_in = pairs["items"] if isinstance(pairs, dict) and "items" in pairs else pairs
    out: list[dict] = []
    seen: set[int] = set()
    for it in items_in:
        title = str(it.get("soc_major_title") or "")
        if title not in titles:
            continue
        fid = int(it["scenario_family_id"])
        if fid in seen:
            continue
        seen.add(fid)
        rows = []
        for r in it["rows"]:
            rows.append(
                {
                    "id": r["id"],
                    "position_variant": r["position_variant"],
                    "context_order": r["context_order"],
                    "labels": dict(r["labels"]),
                    "valid_labels": list(r.get("valid_labels") or ["A", "B"]),
                    "prompt": r.get("gender_prompt") or r["prompt"],
                    # No frozen H1 baseline_choice on xy_pairs; capture_condition
                    # and Stage A fall back to live score choice.
                }
            )
        out.append(
            {
                "scenario_family_id": fid,
                "soc_major_title": title,
                "soc": it.get("soc"),
                "profession": it.get("profession"),
                "onet_action": it.get("onet_action"),
                "rows": rows,
            }
        )
    out.sort(key=lambda x: int(x["scenario_family_id"]))
    return out, soc_domains


def build_pool_sample_doc(
    set_entry: dict[str, Any],
    cfg: dict[str, Any],
    *,
    scale: str = "2b",
    split_name: str | None = None,
) -> dict[str, Any]:
    """Full pool sample with stratified split meta (train/val/test ids)."""
    items, soc_domains = load_xy_pool_items(set_entry, scale=scale)
    split_cfg = cfg.get("split") or {}
    split = plan_pool_split(
        items,
        seed=int(split_cfg.get("seed", 0)),
        train_frac=float(split_cfg.get("train_frac", 0.7)),
        val_frac=float(split_cfg.get("val_frac", 0.15)),
    )
    domain = make_pool_domain(set_entry, soc_domains)
    train = set(split["train_family_ids"])
    val = set(split["val_family_ids"])
    test = set(split["test_family_ids"])

    def subset(fams: set[int]) -> list[dict]:
        return [it for it in items if int(it["scenario_family_id"]) in fams]

    return {
        "schema": "steering.inlp_polarity_pool_sample/v1",
        "protocol": "pooled_polarity_one_subspace",
        "scale": scale,
        "set_id": set_entry["id"],
        "polarity": set_entry["polarity"],
        "label": set_entry["label"],
        "slug": pool_slug(set_entry["polarity"]),
        "member_slugs": domain["member_slugs"],
        "member_titles": domain["member_titles"],
        "n_soc": domain["n_soc"],
        "config": str(DEFAULT_CONFIG.as_posix()),
        "xy_dataset": str(XY_PAIRS.as_posix()),
        "split": split,
        "split_name": split_name or "full_with_split",
        "n_families": len(items),
        "n_rows": sum(len(it["rows"]) for it in items),
        "items": items,
        "items_train": subset(train),
        "items_val": subset(val),
        "items_test": subset(test),
        "probe_peak_layers": resolve_peak_layers(cfg),
        "candidate_grid": {
            "layers": resolve_peak_layers(cfg),
            "ranks": resolve_ranks(cfg),
            "alphas": resolve_alphas(cfg),
            "n": len(expand_inlp_pool_candidates(cfg)),
        },
    }


def write_split_sample(doc: dict[str, Any], split: str, path: Path) -> Path:
    """Write a Stage-A-compatible sample (items = one split only)."""
    key = f"items_{split}"
    if key not in doc:
        raise KeyError(key)
    items = doc[key]
    slim = {
        "schema": "steering.h1_stagea_sample/v1",
        "map_version": f"inlp_polarity_pool_{doc['set_id']}_{split}",
        "source_split": split,
        "protocol": doc["protocol"],
        "set_id": doc["set_id"],
        "polarity": doc["polarity"],
        "n_families_in_split": len(items),
        "n_rows": sum(len(it["rows"]) for it in items),
        "parent_schema": doc["schema"],
        "split_family_ids": doc["split"][f"{split}_family_ids"],
        "member_titles": doc["member_titles"],
        "items": items,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(slim, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
