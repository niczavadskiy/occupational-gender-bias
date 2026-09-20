"""Per-SOC XY-control: one v_raw and one steering grid per FDR-significant domain."""

from __future__ import annotations

from typing import Any

import numpy as np

from steering.build_h1_candidates import fmt_strength
from steering.contrastive.build_vectors import maybe_cos_with_probe, needed_layers
from steering.xy_control.algebra import family_split3, fit_v_raw
from steering.xy_control.domains import hypothesized_alpha_sign, slug_soc
from steering.xy_control.paired_sample import family_ids, filter_soc

MIN_FAMILIES = 2


def domain_family_ids(items: list[dict], title: str) -> list[int]:
    return family_ids(filter_soc(items, title))


def alpha_matches_prior(alpha: float, hypothesized: str) -> bool:
    if hypothesized == "negative":
        return float(alpha) < 0
    if hypothesized == "positive":
        return float(alpha) > 0
    return False


def resolve_family_layers(cfg: dict, spec) -> list[int]:
    named = cfg["layers"]
    vals = named[spec] if isinstance(spec, str) else spec
    return [int(v) for v in vals]


def vector_id_for(slug: str) -> str:
    return f"v_raw__{slug}"


def make_candidate(
    domain: dict[str, Any],
    *,
    layer: int,
    alpha: float,
) -> dict[str, Any]:
    slug = domain["slug"]
    return {
        "id": f"soc_{slug}__vraw__L{layer}__add__a{fmt_strength(float(alpha))}",
        "family": f"soc_{slug}",
        "role": "candidate",
        "vector_id": vector_id_for(slug),
        "layers": [int(layer)],
        "intervention": "add",
        "alpha": float(alpha),
        "soc_major_title": domain["soc_major_title"],
        "slug": slug,
        "polarity": domain["polarity"],
        "hypothesized_alpha_sign": domain["hypothesized_alpha_sign"],
        "alpha_matches_prior": alpha_matches_prior(float(alpha), domain["hypothesized_alpha_sign"]),
    }


def expand_soc_candidates(
    domain: dict[str, Any],
    cfg: dict,
    *,
    smoke: bool = False,
    smoke_alphas: tuple[float, ...] = (-1.0, 1.0),
) -> list[dict[str, Any]]:
    if smoke:
        return [
            make_candidate(domain, layer=int(cfg["layers"]["anchor"]), alpha=a) for a in smoke_alphas
        ]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fam in cfg["families"]:
        if not fam.get("enabled", True):
            continue
        if fam.get("role") == "control":
            continue
        if "v_raw" not in fam.get("vectors", ["v_raw"]):
            continue
        layers = resolve_family_layers(cfg, fam["layers"])
        alphas = fam.get("alphas") or cfg["interventions"][0]["alphas"]
        for layer in layers:
            for alpha in alphas:
                cand = make_candidate(domain, layer=layer, alpha=alpha)
                if cand["id"] in seen:
                    continue
                seen.add(cand["id"])
                out.append(cand)
    return out


def plan_domain_splits(
    items: list[dict],
    domains: list[dict[str, Any]],
    *,
    seed: int,
    train_frac: float,
    val_frac: float,
) -> tuple[dict[str, dict[str, Any]], list[tuple[dict[str, Any], int]]]:
    """70/15/15 inside each SOC using all families of that domain."""
    planned: dict[str, dict[str, Any]] = {}
    skipped: list[tuple[dict[str, Any], int]] = []
    for domain in domains:
        fams = domain_family_ids(items, domain["soc_major_title"])
        split = split_domain_families(
            fams, seed=seed, train_frac=train_frac, val_frac=val_frac
        )
        if split is None:
            skipped.append((domain, len(fams)))
            continue
        train_fams, val_fams, test_fams = split
        planned[domain["slug"]] = {
            "soc_major_title": domain["soc_major_title"],
            "train_family_ids": sorted(train_fams),
            "val_family_ids": sorted(val_fams),
            "test_family_ids": sorted(test_fams),
            "n_families": len(fams),
            "n_train": len(train_fams),
            "n_eval": len(val_fams) + len(test_fams),
        }
    return planned, skipped


def train_items_from_plan(items: list[dict], planned: dict[str, dict[str, Any]]) -> list[dict]:
    train_ids: set[int] = set()
    for meta in planned.values():
        train_ids.update(int(x) for x in meta["train_family_ids"])
    out = [it for it in items if int(it["scenario_family_id"]) in train_ids]
    out.sort(key=lambda it: int(it["scenario_family_id"]))
    return out


def split_domain_families(
    family_ids: list[int],
    *,
    seed: int,
    train_frac: float,
    val_frac: float,
) -> tuple[set[int], set[int], set[int]] | None:
    uniq = sorted(set(int(x) for x in family_ids))
    if len(uniq) < MIN_FAMILIES:
        return None
    return family_split3(uniq, seed=seed, train_frac=train_frac, val_frac=val_frac)


def eval_ids_for_split(
    train: set[int],
    val: set[int],
    test: set[int],
    split: str,
) -> set[int]:
    if split == "all":
        return set(train) | set(val) | set(test)
    if split == "train":
        return set(train)
    if split == "val":
        return set(val)
    if split == "test":
        return set(test)
    if split == "heldout":
        held = set(val) | set(test)
        return held if held else set(train)
    raise ValueError(f"unknown split {split}")


def fit_domain_vectors(
    cfg: dict,
    H_g: dict[int, np.ndarray],
    H_xy: dict[int, np.ndarray],
    capture_rows: list[dict],
    domain: dict[str, Any],
    *,
    train_fams: set[int],
    val_fams: set[int],
    test_fams: set[int],
    probe_bank: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    probe_bank = probe_bank or {}
    title = domain["soc_major_title"]
    slug = domain["slug"]
    vid = vector_id_for(slug)
    layers = needed_layers(cfg)
    d_model = int(cfg["source"]["d_model"])
    train_mask = np.array(
        [
            r.get("soc_major_title") == title and int(r["scenario_family_id"]) in train_fams
            for r in capture_rows
        ]
    )
    if not train_mask.any():
        raise ValueError(f"{title}: empty train mask for v_raw")
    arrays: dict[str, np.ndarray] = {}
    entries: list[dict] = []
    for layer in layers:
        if layer not in H_g:
            continue
        g = H_g[layer].astype(np.float64)[train_mask]
        x = H_xy[layer].astype(np.float64)[train_mask]
        if g.shape != x.shape:
            raise ValueError(f"{title} L{layer}: gender {g.shape} vs XY {x.shape}")
        if g.shape[1] != d_model:
            raise ValueError(f"d_model mismatch: config {d_model}, HS L{layer} {g.shape[1]}")
        w, norm_raw = fit_v_raw(g - x)
        extra = {
            "norm_raw": float(norm_raw),
            "mean_abs_d": float(np.mean(np.linalg.norm(g - x, axis=1))),
            "n_train_pairs": int(train_mask.sum()),
            "n_val_families": len(val_fams),
            "n_test_families": len(test_fams),
            **maybe_cos_with_probe(w, layer, probe_bank),
        }
        key = f"{vid}__L{layer}"
        arrays[key] = w.astype(np.float32)
        entries.append(
            {
                "key": key,
                "vector_id": vid,
                "kind": "paired_mean_diff",
                "layer": int(layer),
                "label_for_c": "paired_xy_control",
                "c": 0.0,
                "sigma_train": 1.0,
                "soc_major_title": title,
                "slug": slug,
                "polarity": domain["polarity"],
                **extra,
            }
        )
    return arrays, entries


def skip_row(
    *,
    scale: str,
    title: str,
    reason: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "scale": scale,
        "soc_major_title": title,
        "slug": slug_soc(title) if title else "",
        "steer": False,
        "reason": reason,
        "polarity": "",
        "hypothesized_alpha_sign": hypothesized_alpha_sign(""),
    }
    if extra:
        row.update(extra)
    return row
