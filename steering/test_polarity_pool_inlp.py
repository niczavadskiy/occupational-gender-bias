"""CPU tests for polarity-pooled INLP protocol helpers."""

from __future__ import annotations

from steering.polarity_pool_inlp import (
    DEFAULT_CONFIG,
    expand_inlp_pool_candidates,
    get_set,
    load_polarity_sets,
    load_yaml,
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


def test_candidate_grid_9() -> None:
    cfg = load_yaml(DEFAULT_CONFIG)
    cands = expand_inlp_pool_candidates(cfg, smoke=False)
    assert len(cands) == 9, len(cands)
    layers = {c["layer"] for c in cands}
    ranks = {c["rank"] for c in cands}
    assert layers == {16, 15, 14}
    assert ranks == {4, 8, 16}
    assert all(c["alpha"] == 1.0 for c in cands)
    assert cands[0]["id"] == "inlp__L16__k4__a1"
    smoke = expand_inlp_pool_candidates(cfg, smoke=True)
    assert len(smoke) == 1
    assert smoke[0]["layer"] == 16
    assert smoke[0]["rank"] == 4


def test_sets_match_xy_slugs() -> None:
    doc = load_polarity_sets()
    male = get_set(doc, "promale")
    female = get_set(doc, "profemale")
    assert male["n_soc"] == 8
    assert female["n_soc"] == 2
    assert len(male["slugs"]) == 8
    assert doc["probe_peak"]["layers"] == [16, 15, 14]
    assert doc["candidate_grid"]["n_candidates"] == 9


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
    assert resolve_ranks(cfg) == [4, 8, 16]


def main() -> int:
    test_peak_prepeak_layers()
    test_candidate_grid_9()
    test_sets_match_xy_slugs()
    test_split_matches_xy_helper()
    test_ranks_from_config()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
