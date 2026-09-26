"""Pooled polarity XY-control: one v / one (L,α) per pro-male or pro-female set.

Unlike per-SOC, train/val/test are unions of FDR SOCs of one polarity
(stratified 70/15/15 inside each SOC, then pooled). Stage A fits a single
v_raw on all train families and picks one winner on pooled val.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from steering.build_h1_candidates import fmt_strength
from steering.xy_control.algebra import family_split3_by_soc, fit_v_raw
from steering.xy_control.domains import CATALOG_JSON, load_catalog
from steering.xy_control.per_soc import (
    alpha_matches_prior,
    resolve_family_layers,
)
from steering.contrastive.build_vectors import maybe_cos_with_probe, needed_layers

HERE = Path(__file__).resolve().parent
SETS_JSON = HERE / "domains" / "polarity_sets_2b_v1.json"
PRIOR_BY_POLARITY = {"male": "negative", "female": "positive"}


def load_polarity_sets(path: Path | None = None) -> dict[str, Any]:
    p = path or SETS_JSON
    return json.loads(p.read_text(encoding="utf-8"))


def get_set(doc: dict[str, Any], set_id: str) -> dict[str, Any]:
    for s in doc["sets"]:
        if s["id"] == set_id:
            return s
    raise KeyError(f"unknown polarity set id: {set_id}")


def pool_slug(polarity: str) -> str:
    return f"polarity_{polarity}"


def vector_id_for_pool(polarity: str) -> str:
    return f"v_raw__{pool_slug(polarity)}"


def catalog_domains_for_set(
    catalog: dict[str, Any],
    *,
    scale: str,
    slugs: list[str],
) -> list[dict[str, Any]]:
    by_slug = {d["slug"]: d for d in catalog["scales"][scale]["steer"]}
    out = []
    for slug in slugs:
        if slug not in by_slug:
            raise KeyError(f"{slug} not in {scale} steer catalog")
        out.append(by_slug[slug])
    return out


def make_pool_domain(set_entry: dict[str, Any], soc_domains: list[dict[str, Any]]) -> dict[str, Any]:
    polarity = set_entry["polarity"]
    prior = PRIOR_BY_POLARITY.get(polarity, "negative")
    return {
        "slug": pool_slug(polarity),
        "soc_major_title": set_entry["label"],
        "polarity": polarity,
        "hypothesized_alpha_sign": prior,
        "set_id": set_entry["id"],
        "label": set_entry["label"],
        "member_slugs": [d["slug"] for d in soc_domains],
        "member_titles": [d["soc_major_title"] for d in soc_domains],
        "n_soc": len(soc_domains),
    }


def make_pool_candidate(
    domain: dict[str, Any],
    *,
    layer: int,
    alpha: float,
) -> dict[str, Any]:
    slug = domain["slug"]
    return {
        "id": f"pool_{slug}__vraw__L{layer}__add__a{fmt_strength(float(alpha))}",
        "family": f"pool_{slug}",
        "role": "candidate",
        "vector_id": vector_id_for_pool(domain["polarity"]),
        "layers": [int(layer)],
        "intervention": "add",
        "alpha": float(alpha),
        "soc_major_title": domain["soc_major_title"],
        "slug": slug,
        "polarity": domain["polarity"],
        "hypothesized_alpha_sign": domain["hypothesized_alpha_sign"],
        "alpha_matches_prior": alpha_matches_prior(
            float(alpha), domain["hypothesized_alpha_sign"]
        ),
        "set_id": domain.get("set_id"),
    }


def expand_pool_candidates(
    domain: dict[str, Any],
    cfg: dict,
    *,
    smoke: bool = False,
    smoke_alphas: tuple[float, ...] = (-1.0, 1.0),
) -> list[dict[str, Any]]:
    if smoke:
        return [
            make_pool_candidate(domain, layer=int(cfg["layers"]["anchor"]), alpha=a)
            for a in smoke_alphas
        ]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fam in cfg["families"]:
        if not fam.get("enabled", True) or fam.get("role") == "control":
            continue
        if "v_raw" not in fam.get("vectors", ["v_raw"]):
            continue
        layers = resolve_family_layers(cfg, fam["layers"])
        alphas = fam.get("alphas") or cfg["interventions"][0]["alphas"]
        for layer in layers:
            for alpha in alphas:
                cand = make_pool_candidate(domain, layer=layer, alpha=alpha)
                if cand["id"] in seen:
                    continue
                seen.add(cand["id"])
                out.append(cand)
    return out


def plan_pool_split(
    items: list[dict],
    *,
    seed: int,
    train_frac: float,
    val_frac: float,
) -> dict[str, Any]:
    """Stratified 70/15/15 inside each SOC, then union → pooled train/val/test."""
    train, val, test = family_split3_by_soc(
        items, seed=seed, train_frac=train_frac, val_frac=val_frac
    )
    by_soc: dict[str, dict[str, list[int]]] = {}
    for it in items:
        title = str(it.get("soc_major_title") or "UNKNOWN")
        fid = int(it["scenario_family_id"])
        slot = (
            "train"
            if fid in train
            else "val"
            if fid in val
            else "test"
            if fid in test
            else None
        )
        if slot is None:
            continue
        by_soc.setdefault(title, {"train": [], "val": [], "test": []})[slot].append(fid)
    for title in by_soc:
        for k in ("train", "val", "test"):
            by_soc[title][k] = sorted(set(by_soc[title][k]))
    return {
        "train_family_ids": sorted(train),
        "val_family_ids": sorted(val),
        "test_family_ids": sorted(test),
        "n_families": len(train | val | test),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "per_soc": by_soc,
        "split_rule": "family_split3_by_soc then pool",
    }


def fit_pool_vectors(
    cfg: dict,
    H_g: dict[int, Any],
    H_xy: dict[int, Any],
    capture_rows: list[dict],
    domain: dict[str, Any],
    *,
    train_fams: set[int],
    val_fams: set[int],
    test_fams: set[int],
    probe_bank: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict]]:
    import numpy as np

    probe_bank = probe_bank or {}
    vid = vector_id_for_pool(domain["polarity"])
    layers = needed_layers(cfg)
    d_model = int(cfg["source"]["d_model"])
    train_mask = np.array(
        [int(r["scenario_family_id"]) in train_fams for r in capture_rows]
    )
    if not train_mask.any():
        raise ValueError(f"{domain['slug']}: empty train mask for pooled v_raw")
    arrays: dict[str, Any] = {}
    entries: list[dict] = []
    for layer in layers:
        if layer not in H_g:
            continue
        g = H_g[layer].astype(np.float64)[train_mask]
        x = H_xy[layer].astype(np.float64)[train_mask]
        if g.shape != x.shape:
            raise ValueError(f"pool L{layer}: gender {g.shape} vs XY {x.shape}")
        if g.shape[1] != d_model:
            raise ValueError(f"d_model mismatch: config {d_model}, HS L{layer} {g.shape[1]}")
        w, norm_raw = fit_v_raw(g - x)
        key = f"{vid}__L{layer}"
        arrays[key] = w.astype(np.float32)
        entries.append(
            {
                "key": key,
                "vector_id": vid,
                "kind": "paired_mean_diff_pooled_polarity",
                "layer": int(layer),
                "label_for_c": "paired_xy_control",
                "c": 0.0,
                "sigma_train": 1.0,
                "slug": domain["slug"],
                "polarity": domain["polarity"],
                "set_id": domain.get("set_id"),
                "norm_raw": float(norm_raw),
                "mean_abs_d": float(np.mean(np.linalg.norm(g - x, axis=1))),
                "n_train_pairs": int(train_mask.sum()),
                "n_val_families": len(val_fams),
                "n_test_families": len(test_fams),
                **maybe_cos_with_probe(w, layer, probe_bank),
            }
        )
    return arrays, entries


def mmlu_items_for_titles(profile: dict, bank: dict[int, dict], titles: list[str]) -> list[dict]:
    from steering.mmlu_eval import expand_domain_profile

    wanted = set(titles)
    scoped = {**profile, "domains": [d for d in profile["domains"] if d["soc_major_title"] in wanted]}
    if not scoped["domains"]:
        raise ValueError(f"MMLU profile has none of {titles}")
    return expand_domain_profile(scoped, bank)


def load_set_bundle(
    *,
    set_id: str,
    scale: str,
    sets_path: Path | None = None,
    catalog_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    doc = load_polarity_sets(sets_path)
    set_entry = get_set(doc, set_id)
    catalog = load_catalog(catalog_path or CATALOG_JSON)
    socs = catalog_domains_for_set(catalog, scale=scale, slugs=set_entry["slugs"])
    domain = make_pool_domain(set_entry, socs)
    return doc, set_entry, socs, domain
