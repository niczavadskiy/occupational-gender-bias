"""CPU tests for polarity-pooled INLP protocol helpers."""

from __future__ import annotations

from steering.polarity_pool_inlp import (
    DEFAULT_CONFIG,
    clamp_ranks_to_k_found,
    expand_inlp_pool_candidates,
    get_set,
    load_polarity_sets,
    load_yaml,
    max_k_found,
    resolve_peak_layers,
    resolve_ranks,
)
from steering.xy_control.algebra import family_split3_by_soc
from steering.xy_control.polarity_pool import plan_pool_split


def test_peak_prepeak_layers() -> None:
    cfg = load_yaml(DEFAULT_CONFIG)
    layers = resolve_peak_layers(cfg)
    assert layers == [16, 15, 14], layers
    assert int(cfg["layers"]["peak"]) == 16
    assert cfg["layers"]["prepeak"] == [15, 14]


def test_candidate_grid_preferred() -> None:
    cfg = load_yaml(DEFAULT_CONFIG)
    cands = expand_inlp_pool_candidates(cfg, smoke=False)
    assert len(cands) == 12, len(cands)
    layers = {c["layer"] for c in cands}
    ranks = {c["rank"] for c in cands}
    assert layers == {16, 15, 14}
    assert ranks == {1, 4, 8, 16}
    assert cands[0]["id"] == "inlp__L16__k1__a1"
    smoke = expand_inlp_pool_candidates(cfg, smoke=True)
    assert len(smoke) == 1
    assert smoke[0]["layer"] == 16
    assert smoke[0]["rank"] == 1


def test_clamp_ranks_k1() -> None:
    assert clamp_ranks_to_k_found([1, 4, 8, 16], 1) == [1]
    assert clamp_ranks_to_k_found([4, 8, 16], 1) == [1]
    assert clamp_ranks_to_k_found([1, 4, 8, 16], 8) == [1, 4, 8]
    assert clamp_ranks_to_k_found([4, 8], 0) == [1]
    cfg = load_yaml(DEFAULT_CONFIG)
    cands = expand_inlp_pool_candidates(cfg, k_found=1)
    assert {c["rank"] for c in cands} == {1}
    assert len(cands) == 3


def test_max_k_found() -> None:
    meta = {
        "layers_detail": [
            {"layer": 16, "k_found": 1},
            {"layer": 15, "k_found": 1},
            {"layer": 14, "k_found": 1},
        ]
    }
    assert max_k_found(meta) == 1
    assert max_k_found(meta, [16, 15]) == 1


def test_sets_match_xy_slugs() -> None:
    doc = load_polarity_sets()
    male = get_set(doc, "promale")
    female = get_set(doc, "profemale")
    assert male["n_soc"] == 8
    assert female["n_soc"] == 2
    assert len(male["slugs"]) == 8
    assert doc["probe_peak"]["layers"] == [16, 15, 14]
    assert 1 in doc["candidate_grid"]["ranks"]


def test_sets_4b_peak_and_stems() -> None:
    from steering.polarity_pool_inlp import STEERING_DIR, sample_paths_for, sample_stem_for

    path = STEERING_DIR / "domains" / "inlp_polarity_sets_4b_v1.json"
    doc = load_polarity_sets(path)
    cfg = load_yaml(STEERING_DIR / "configs" / "inlp_polarity_pool_4b_peak_prepeak.yaml")
    assert doc["scale"] == "4b"
    assert resolve_peak_layers(cfg) == [24, 23, 22]
    male = get_set(doc, "promale")
    female = get_set(doc, "profemale")
    assert male["n_soc"] == 4
    assert female["n_soc"] == 6
    assert sample_stem_for(male) == "inlp_polarity_pool_4b_promale_v1"
    assert sample_paths_for(male)["val"].name == "inlp_polarity_pool_4b_promale_val_v1.json"
    assert male["subspace_tag"] == "polarity_pool_4b_promale_v1"
    xy_doc = load_polarity_sets(STEERING_DIR / "xy_control" / "domains" / "polarity_sets_4b_v1.json")
    assert get_set(xy_doc, "promale")["slugs"] == male["slugs"]
    assert get_set(xy_doc, "profemale")["slugs"] == female["slugs"]


def test_split_matches_xy_helper() -> None:
    items = []
    for title, n in (("A", 20), ("B", 10)):
        for i in range(n):
            items.append(
                {
                    "soc_major_title": title,
                    "scenario_family_id": hash(title) % 10_000 + i,
                }
            )
    split = plan_pool_split(items, seed=0, train_frac=0.7, val_frac=0.15)
    t, v, te = family_split3_by_soc(items, seed=0, train_frac=0.7, val_frac=0.15)
    assert set(split["train_family_ids"]) == t
    assert set(split["val_family_ids"]) == v
    assert set(split["test_family_ids"]) == te


def test_ranks_from_config() -> None:
    cfg = load_yaml(DEFAULT_CONFIG)
    assert resolve_ranks(cfg) == [1, 4, 8, 16]


def main() -> int:
    test_peak_prepeak_layers()
    test_candidate_grid_preferred()
    test_clamp_ranks_k1()
    test_max_k_found()
    test_sets_match_xy_slugs()
    test_sets_4b_peak_and_stems()
    test_split_matches_xy_helper()
    test_ranks_from_config()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
