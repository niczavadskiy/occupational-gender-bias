"""CPU tests for per-SOC XY-control Stage B/C selection."""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

from steering.xy_control.domains import load_catalog
from steering.xy_control.stagebc import (
    candidate_from_entry,
    make_stageb_shortlist,
    select_stagec_entry,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_shortlist_best_plus_prior() -> None:
    catalog = load_catalog()
    domain = catalog["scales"]["2b"]["steer"][0]
    slug = domain["slug"]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        best = f"soc_{slug}__vraw__L15__add__a2"
        prior = f"soc_{slug}__vraw__L15__add__am2"
        _write_csv(
            root / "summary.csv",
            [
                {
                    "scale": "2b",
                    "soc_major_title": domain["soc_major_title"],
                    "slug": slug,
                    "polarity": domain["polarity"],
                    "baseline_gender_gap": 0.2,
                    "best_id": best,
                    "best_prior_id": prior,
                }
            ],
        )
        _write_csv(
            root / slug / "ranking.csv",
            [
                {
                    "config_id": "baseline",
                    "layer": "",
                    "alpha": 0.0,
                    "alpha_matches_prior": "",
                    "gender_gap": 0.2,
                    "gap_abs_reduction": 0.0,
                    "slot_gap": 0.01,
                },
                {
                    "config_id": best,
                    "layer": 15,
                    "alpha": 2.0,
                    "alpha_matches_prior": False,
                    "gender_gap": 0.1,
                    "gap_abs_reduction": 0.1,
                    "slot_gap": 0.02,
                },
                {
                    "config_id": prior,
                    "layer": 15,
                    "alpha": -2.0,
                    "alpha_matches_prior": True,
                    "gender_gap": 0.12,
                    "gap_abs_reduction": 0.08,
                    "slot_gap": 0.015,
                },
            ],
        )
        doc = make_stageb_shortlist(root, catalog, "2b")
        assert [c["config_id"] for c in doc["domains"][0]["candidates"]] == [best, prior]
        assert candidate_from_entry(domain, doc["domains"][0]["candidates"][1])["id"] == prior


def test_stageb_selection_and_identity() -> None:
    entry = {
        "slug": "toy",
        "soc_major_title": "Toy Occupations",
        "polarity": "male",
    }
    rows = [
        {
            "config_id": "free",
            "layer": 15,
            "alpha": 2.0,
            "alpha_matches_prior": False,
            "stage_a_gap_abs_reduction": 0.2,
            "stage_a_slot_gap": 0.02,
            "cap_loss": 0.02,
        },
        {
            "config_id": "prior",
            "layer": 15,
            "alpha": -2.0,
            "alpha_matches_prior": True,
            "stage_a_gap_abs_reduction": 0.1,
            "stage_a_slot_gap": 0.01,
            "cap_loss": 0.02,
        },
    ]
    keep = select_stagec_entry(entry, rows, 0.03)
    assert keep["config_id"] == "prior"
    blocked = select_stagec_entry(entry, [{**rows[0], "cap_loss": 0.04}], 0.03)
    assert blocked["decision"] == "identity" and blocked["config_id"] == "baseline"


def main() -> int:
    tests = [test_shortlist_best_plus_prior, test_stageb_selection_and_identity]
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
