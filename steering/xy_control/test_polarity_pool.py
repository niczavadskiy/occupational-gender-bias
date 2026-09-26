"""CPU tests for pooled polarity helpers."""

from __future__ import annotations

from steering.xy_control.algebra import family_split3_by_soc
from steering.xy_control.polarity_pool import (
    expand_pool_candidates,
    get_set,
    load_polarity_sets,
    make_pool_candidate,
    make_pool_domain,
    plan_pool_split,
    pool_slug,
)


def test_pool_slug_and_candidate_id() -> None:
    domain = {
        "slug": "polarity_male",
        "soc_major_title": "pro-male",
        "polarity": "male",
        "hypothesized_alpha_sign": "negative",
        "set_id": "promale",
    }
    c = make_pool_candidate(domain, layer=16, alpha=-4.0)
    assert c["id"] == "pool_polarity_male__vraw__L16__add__am4"
    assert c["vector_id"] == "v_raw__polarity_male"
    assert c["alpha_matches_prior"] is True


def test_sets_json_loads() -> None:
    doc = load_polarity_sets()
    male = get_set(doc, "promale")
    female = get_set(doc, "profemale")
    assert male["n_soc"] == 8
    assert female["n_soc"] == 2
    assert pool_slug("male") == "polarity_male"


def test_sets_4b_json_loads() -> None:
    import yaml
    from pathlib import Path

    from steering.xy_control.polarity_pool import HERE

    path = HERE / "domains" / "polarity_sets_4b_v1.json"
    doc = load_polarity_sets(path)
    assert doc["scale"] == "4b"
    assert doc["probe_peak"]["layers"] == [24, 23, 22]
    male = get_set(doc, "promale")
    female = get_set(doc, "profemale")
    assert male["n_soc"] == 4
    assert female["n_soc"] == 6
    assert len(male["slugs"]) == 4
    cfg = yaml.safe_load((HERE / "configs" / "xy_control_4b_peak_prepeak_a6.yaml").read_text(encoding="utf-8"))
    assert cfg["layers"]["peak"] == 24
    assert cfg["families"][0]["layers"] == [24, 23, 22]
    _ = Path


def test_plan_pool_split_stratified() -> None:
    items = []
    for title, n in (("A", 20), ("B", 10)):
        for i in range(n):
            items.append({"soc_major_title": title, "scenario_family_id": hash(title) % 10_000 + i})
    split = plan_pool_split(items, seed=0, train_frac=0.7, val_frac=0.15)
    assert split["n_train"] + split["n_val"] + split["n_test"] == split["n_families"]
    assert set(split["train_family_ids"]) & set(split["val_family_ids"]) == set()
    # matches family_split3_by_soc
    t, v, te = family_split3_by_soc(items, seed=0, train_frac=0.7, val_frac=0.15)
    assert set(split["train_family_ids"]) == t
    assert set(split["val_family_ids"]) == v
    assert set(split["test_family_ids"]) == te


def test_expand_smoke() -> None:
    cfg = {
        "layers": {"anchor": 16, "belt": [14, 15, 16]},
        "families": [
            {
                "enabled": True,
                "role": "candidate",
                "vectors": ["v_raw"],
                "layers": [16, 15, 14],
                "alphas": [-1.0, 1.0],
            }
        ],
        "interventions": [{"alphas": [-1.0, 1.0]}],
    }
    domain = make_pool_domain(
        {"id": "promale", "label": "pro-male", "polarity": "male"},
        [{"slug": "a", "soc_major_title": "A", "polarity": "male"}],
    )
    cands = expand_pool_candidates(domain, cfg, smoke=True)
    assert len(cands) == 2
    assert all(c["layers"] == [16] for c in cands)
    full = expand_pool_candidates(domain, cfg, smoke=False)
    assert len(full) == 6


def main() -> int:
    test_pool_slug_and_candidate_id()
    test_sets_json_loads()
    test_sets_4b_json_loads()
    test_plan_pool_split_stratified()
    test_expand_smoke()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
