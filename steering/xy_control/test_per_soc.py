"""CPU tests for per-SOC vector fit and candidate expansion."""

from __future__ import annotations

import sys

import numpy as np
import yaml

from steering.xy_control.algebra import family_split3
from steering.xy_control.build_vectors import CONFIG_BY_SCALE
from steering.xy_control.domains import load_catalog
from steering.xy_control.paired_sample import load_pooled_items
from steering.xy_control.per_soc import (
    alpha_matches_prior,
    domain_family_ids,
    expand_soc_candidates,
    fit_domain_vectors,
    plan_domain_splits,
    skip_row,
    split_domain_families,
    vector_id_for,
)


def _cfg(scale: str) -> dict:
    return yaml.safe_load(CONFIG_BY_SCALE[scale].read_text(encoding="utf-8"))


def test_pooled_unique() -> None:
    items = load_pooled_items()
    fids = [int(it["scenario_family_id"]) for it in items]
    assert len(items) == 951
    assert len(fids) == len(set(fids))
    assert sum(len(it["rows"]) for it in items) == 3804
    assert len(domain_family_ids(items, "Production Occupations")) == 108
    catalog = load_catalog()
    for scale in ("2b", "4b"):
        for domain in catalog["scales"][scale]["steer"]:
            n = len(domain_family_ids(items, domain["soc_major_title"]))
            assert n >= 2, domain["soc_major_title"]
    layouts = {
        (row["context_order"], row["position_variant"])
        for it in items
        for row in it["rows"]
    }
    assert layouts == {
        ("man_first", "p0"),
        ("man_first", "p1"),
        ("woman_first", "p0"),
        ("woman_first", "p1"),
    }


def test_candidates_per_significant_soc() -> None:
    catalog = load_catalog()
    cfg = _cfg("2b")
    ids: list[str] = []
    for domain in catalog["scales"]["2b"]["steer"]:
        cands = expand_soc_candidates(domain, cfg)
        assert cands
        assert all(c["vector_id"] == vector_id_for(domain["slug"]) for c in cands)
        assert all(c["soc_major_title"] == domain["soc_major_title"] for c in cands)
        ids.extend(c["id"] for c in cands)
        male_neg = [c for c in cands if c["alpha"] < 0]
        female_pos = [c for c in cands if c["alpha"] > 0]
        assert male_neg and female_pos
        if domain["polarity"] == "male":
            assert all(c["alpha_matches_prior"] for c in male_neg)
            assert not any(c["alpha_matches_prior"] for c in female_pos)
        else:
            assert all(c["alpha_matches_prior"] for c in female_pos)
            assert not any(c["alpha_matches_prior"] for c in male_neg)
    assert len(ids) == len(set(ids))
    smoke = expand_soc_candidates(catalog["scales"]["2b"]["steer"][0], cfg, smoke=True)
    assert [c["layers"][0] for c in smoke] == [cfg["layers"]["anchor"]] * 2
    assert {c["alpha"] for c in smoke} == {-1.0, 1.0}


def test_skip_non_significant() -> None:
    catalog = load_catalog()
    skip_titles = {d["soc_major_title"] for d in catalog["scales"]["2b"]["skip"]}
    steer_titles = {d["soc_major_title"] for d in catalog["scales"]["2b"]["steer"]}
    assert not (skip_titles & steer_titles)
    row = skip_row(scale="2b", title="Management Occupations", reason="not_significant")
    assert row["steer"] is False and row["reason"] == "not_significant"


def test_plan_uses_all_soc_families() -> None:
    items = [
        {"scenario_family_id": i, "soc_major_title": "Toy Occupations"}
        for i in range(10)
    ] + [{"scenario_family_id": 99, "soc_major_title": "Other Occupations"}]
    domains = [{"slug": "toy_occupations", "soc_major_title": "Toy Occupations"}]
    planned, skipped = plan_domain_splits(
        items, domains, seed=0, train_frac=0.7, val_frac=0.15
    )
    assert not skipped
    meta = planned["toy_occupations"]
    assert domain_family_ids(items, "Toy Occupations") == list(range(10))
    assert meta["n_families"] == 10
    assert meta["n_train"] + meta["n_eval"] == 10
    tiny = [{"slug": "toy_occupations", "soc_major_title": "Toy Occupations"}]
    _, skipped = plan_domain_splits(
        items[:1], tiny, seed=0, train_frac=0.7, val_frac=0.15
    )
    assert skipped and skipped[0][1] == 1
    fams = list(range(10))
    split = split_domain_families(fams, seed=0, train_frac=0.7, val_frac=0.15)
    assert split is not None
    train, val, test = split
    assert not (train & val) and not (train & test) and not (val & test)
    assert train | val | test == set(fams)
    assert split_domain_families([1], seed=0, train_frac=0.7, val_frac=0.15) is None
    a = family_split3(fams, seed=0, train_frac=0.7, val_frac=0.15)
    assert a == split


def test_fit_uses_only_that_soc_train() -> None:
    cfg = {
        "source": {"d_model": 4},
        "layers": {"core": [1], "edge": [1], "anchor": 1, "belt": [1]},
        "families": [
            {
                "enabled": True,
                "role": "candidate",
                "vectors": ["v_raw"],
                "layers": "core",
                "alphas": [1.0],
            }
        ],
    }
    domain = {
        "soc_major_title": "Toy Occupations",
        "slug": "toy_occupations",
        "polarity": "male",
        "hypothesized_alpha_sign": "negative",
    }
    rows = [
        {"soc_major_title": "Toy Occupations", "scenario_family_id": 1},
        {"soc_major_title": "Toy Occupations", "scenario_family_id": 1},
        {"soc_major_title": "Toy Occupations", "scenario_family_id": 2},
        {"soc_major_title": "Other Occupations", "scenario_family_id": 3},
    ]
    H_g = {
        1: np.array(
            [
                [2.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [9.0, 0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    }
    H_xy = {1: np.zeros((4, 4), dtype=np.float32)}
    arrays, entries = fit_domain_vectors(
        cfg,
        H_g,
        H_xy,
        rows,
        domain,
        train_fams={1},
        val_fams={2},
        test_fams=set(),
        probe_bank={},
    )
    key = "v_raw__toy_occupations__L1"
    v = arrays[key]
    assert np.allclose(v, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert entries[0]["n_train_pairs"] == 2
    assert alpha_matches_prior(-1.0, "negative")
    assert not alpha_matches_prior(1.0, "negative")


def test_significant_socs_exist_in_pooled() -> None:
    catalog = load_catalog()
    items = load_pooled_items()
    titles = {it["soc_major_title"] for it in items}
    for scale in ("2b", "4b"):
        for d in catalog["scales"][scale]["steer"]:
            assert d["soc_major_title"] in titles, d["soc_major_title"]


def main() -> int:
    tests = [
        test_pooled_unique,
        test_candidates_per_significant_soc,
        test_skip_non_significant,
        test_plan_uses_all_soc_families,
        test_fit_uses_only_that_soc_train,
        test_significant_socs_exist_in_pooled,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"OK  {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
