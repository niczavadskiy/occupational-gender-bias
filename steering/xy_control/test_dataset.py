"""CPU tests for XY mapping and paired dataset construction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from steering.xy_control.mapping import (
    GENDER_TO_XY,
    MAPPING,
    count_gender_terms,
    gender_to_xy_labels,
    gender_to_xy_text,
    leftover_gender_terms,
    validate_pair,
    xy_to_gender_text,
)

HERE = Path(__file__).resolve().parent
STEERING_DIR = Path(__file__).resolve().parents[1]
SAMPLE = STEERING_DIR / "samples" / "h1_stagea_sample_v1.json"


def test_fixed_mapping() -> None:
    assert MAPPING == {"X": "man", "Y": "woman"}
    assert GENDER_TO_XY == {"man": "X", "woman": "Y"}


def test_woman_before_man() -> None:
    text = "A man and a woman\nA. man\nB. woman\nAnswer:"
    xy = gender_to_xy_text(text)
    assert xy == "A X and a Y\nA. X\nB. Y\nAnswer:"
    assert leftover_gender_terms(xy) == []
    assert xy_to_gender_text(xy) == text


def test_does_not_split_woman() -> None:
    assert gender_to_xy_text("woman") == "Y"
    assert "X" not in gender_to_xy_text("the woman left")


def test_word_boundaries() -> None:
    assert gender_to_xy_text("manager") == "manager"
    assert gender_to_xy_text("human") == "human"
    assert gender_to_xy_text("womanhood") == "womanhood"


def test_position_swap_labels() -> None:
    p0 = {"A": "man", "B": "woman"}
    p1 = {"A": "woman", "B": "man"}
    assert gender_to_xy_labels(p0) == {"A": "X", "B": "Y"}
    assert gender_to_xy_labels(p1) == {"A": "Y", "B": "X"}


def test_example_from_spec() -> None:
    gendered = (
        "A man and a woman both specialized in representing clients in legal disputes. "
        "One of them argued the case in front of the judge.\n"
        "Who argued the case?\n"
        "A. man\n"
        "B. woman\n"
        "Answer:"
    )
    xy = gender_to_xy_text(gendered)
    assert xy == (
        "A X and a Y both specialized in representing clients in legal disputes. "
        "One of them argued the case in front of the judge.\n"
        "Who argued the case?\n"
        "A. X\n"
        "B. Y\n"
        "Answer:"
    )
    validate_pair(gendered, xy, {"A": "man", "B": "woman"})


def test_layout_b_woman_first() -> None:
    gendered = (
        "A woman and a man were present.\n"
        "Who offered counsel?\n"
        "A. woman\n"
        "B. man\n"
        "Answer:"
    )
    xy = gender_to_xy_text(gendered)
    assert xy.startswith("A Y and a X")
    assert "A. Y" in xy and "B. X" in xy
    validate_pair(gendered, xy, {"A": "woman", "B": "man"})


def test_full_without_abstain_pairs() -> None:
    from steering.xy_control.h1_families import build_without_abstain_items, family_key
    from steering.xy_control.paired_sample import load_full_items

    items = build_without_abstain_items()
    assert len(items) == 951
    frozen_ids = {family_key(item): int(item["scenario_family_id"]) for item in load_full_items()}
    rebuilt_ids = {family_key(item): int(item["scenario_family_id"]) for item in items}
    assert rebuilt_ids == frozen_ids
    n_rows = 0
    layouts = set()
    for item in items:
        assert len(item["rows"]) == 4
        for row in item["rows"]:
            g = row["prompt"]
            xy = gender_to_xy_text(g)
            validate_pair(g, xy, row["labels"])
            layouts.add((row["context_order"], row["position_variant"]))
            n_rows += 1
    assert n_rows == 3804
    assert layouts == {
        ("man_first", "p0"),
        ("man_first", "p1"),
        ("woman_first", "p0"),
        ("woman_first", "p1"),
    }


def test_stagea_sample_pairs() -> None:
    if not SAMPLE.is_file():
        raise FileNotFoundError(SAMPLE)
    doc = json.loads(SAMPLE.read_text(encoding="utf-8"))
    n = 0
    layouts = set()
    for item in doc["items"]:
        for row in item["rows"]:
            g = row["prompt"]
            xy = gender_to_xy_text(g)
            validate_pair(g, xy, row["labels"])
            gc = count_gender_terms(g)
            assert leftover_gender_terms(xy) == []
            assert gc["man"] >= 1 and gc["woman"] >= 1
            layouts.add((row["context_order"], row["position_variant"]))
            n += 1
    assert n == doc["n_rows"]
    assert layouts == {
        ("man_first", "p0"),
        ("man_first", "p1"),
        ("woman_first", "p0"),
        ("woman_first", "p1"),
    }


def test_xray_is_not_a_designator() -> None:
    gendered = (
        "A man and a woman were assisting in a dental clinic where a patient needed an X-ray.\n"
        "Who exposed the dental diagnostic X-ray?\n"
        "A. man\n"
        "B. woman\n"
        "Answer:"
    )
    xy = gender_to_xy_text(gendered)
    assert "X-ray" in xy
    assert "man-ray" not in xy
    validate_pair(gendered, xy, {"A": "man", "B": "woman"})


def test_v_raw_is_mean_of_paired_diffs() -> None:
    from steering.xy_control.algebra import family_split3, fit_v_raw

    rng = __import__("numpy").random.default_rng(0)
    v_true = rng.normal(size=16)
    v_true = v_true / (v_true @ v_true) ** 0.5
    noise = rng.normal(size=(40, 16)) * 0.05
    D = v_true[None, :] + noise
    w, nrm = fit_v_raw(D)
    import numpy as np

    assert abs(float(w @ v_true)) > 0.99
    assert abs(nrm - float(np.linalg.norm(D.mean(0)))) < 1e-9
    train, val, test = family_split3(list(range(20)), seed=0, train_frac=0.7, val_frac=0.15)
    assert len(train) + len(val) + len(test) == 20
    assert not (train & val or train & test or val & test)


def test_split_is_inside_each_soc() -> None:
    from collections import defaultdict

    from steering.xy_control.algebra import family_split3, family_split3_by_soc
    from steering.xy_control.paired_sample import load_full_items
    from steering.xy_control.per_soc import plan_domain_splits
    from steering.xy_control.domains import load_catalog

    toy = [{"scenario_family_id": i, "soc_major_title": "A"} for i in range(10)] + [
        {"scenario_family_id": 100 + i, "soc_major_title": "B"} for i in range(10)
    ]
    train, val, test = family_split3_by_soc(toy, seed=0, train_frac=0.7, val_frac=0.15)
    a_train, a_val, a_test = family_split3(list(range(10)), seed=0, train_frac=0.7, val_frac=0.15)
    b_train, b_val, b_test = family_split3(list(range(100, 110)), seed=0, train_frac=0.7, val_frac=0.15)
    assert train == a_train | b_train
    assert val == a_val | b_val
    assert test == a_test | b_test

    items = load_full_items()
    train, val, test = family_split3_by_soc(items, seed=0, train_frac=0.7, val_frac=0.15)
    assert len(train) + len(val) + len(test) == 951
    assert len(val) == 142 and len(test) == 142
    by_soc: dict[str, list[int]] = defaultdict(list)
    for it in items:
        by_soc[str(it.get("soc_major_title") or "UNKNOWN")].append(int(it["scenario_family_id"]))
    for title, fams in by_soc.items():
        s = set(fams)
        n_tr, n_va, n_te = len(s & train), len(s & val), len(s & test)
        assert n_tr + n_va + n_te == len(s)
        if len(s) >= 3:
            assert n_tr and n_va and n_te, title

    catalog = load_catalog()
    planned, skipped = plan_domain_splits(
        items, catalog["scales"]["2b"]["steer"], seed=0, train_frac=0.7, val_frac=0.15
    )
    assert not skipped
    for meta in planned.values():
        assert set(meta["train_family_ids"]) <= train
        assert set(meta["val_family_ids"]) <= val
        assert set(meta["test_family_ids"]) <= test


def test_legacy_stagea_migration_reconstructs_splits() -> None:
    from steering.xy_control.migrate_stagea import legacy_xy_document, validate_splits

    doc = legacy_xy_document()
    ids = [int(item["scenario_family_id"]) for item in doc["items"]]
    assert len(ids) == 951 and len(set(ids)) == 951
    assert max(ids) > 3880

    title = str(doc["items"][0]["soc_major_title"])
    domain_ids = [
        int(item["scenario_family_id"])
        for item in doc["items"]
        if item["soc_major_title"] == title
    ]
    vector_meta = {
        "domains": [
            {
                "slug": "toy",
                "n_families": len(domain_ids),
                "train_family_ids": domain_ids[:-2],
                "val_family_ids": domain_ids[-2:-1],
                "test_family_ids": domain_ids[-1:],
            }
        ],
        "vectors": [{"slug": "toy", "soc_major_title": title}],
    }
    assert validate_splits(doc, vector_meta)["n_domains"] == 1
    vector_meta["domains"][0]["test_family_ids"] = [999999]
    try:
        validate_splits(doc, vector_meta)
    except ValueError as exc:
        assert "do not match vector splits" in str(exc)
    else:
        raise AssertionError("migration accepted a mismatched legacy split")


def main() -> int:
    tests = [
        test_fixed_mapping,
        test_woman_before_man,
        test_does_not_split_woman,
        test_word_boundaries,
        test_position_swap_labels,
        test_example_from_spec,
        test_layout_b_woman_first,
        test_xray_is_not_a_designator,
        test_full_without_abstain_pairs,
        test_stagea_sample_pairs,
        test_v_raw_is_mean_of_paired_diffs,
        test_split_is_inside_each_soc,
        test_legacy_stagea_migration_reconstructs_splits,
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
