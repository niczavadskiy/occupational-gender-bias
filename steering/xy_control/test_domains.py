"""CPU tests for the frozen H1 SOC-FDR domain catalog."""

from __future__ import annotations

import sys

from steering.xy_control.build_domains import main as build_main
from steering.xy_control.domains import (
    hypothesized_alpha_sign,
    load_catalog,
    polarity_of,
    select_domain,
    slug_soc,
)


def test_catalog_rebuilds() -> None:
    assert build_main(["--verify"]) == 0


def test_union_rule_and_polarity() -> None:
    catalog = load_catalog()
    two = catalog["scales"]["2b"]
    four = catalog["scales"]["4b"]
    assert two["n_steer"] == 10 and two["n_male"] == 8 and two["n_female"] == 2
    assert four["n_steer"] == 10 and four["n_male"] == 4 and four["n_female"] == 6
    by_title = {d["soc_major_title"]: d for d in two["steer"]}
    edu = by_title["Educational Instruction and Library Occupations"]
    assert edu["gate"] == "choice_fdr" and edu["polarity"] == "male"
    care = by_title["Personal Care and Service Occupations"]
    assert care["gate"] == "prob_fdr" and care["polarity"] == "female"
    skip_titles = {d["soc_major_title"] for d in two["skip"]}
    assert "Management Occupations" in skip_titles
    four_by = {d["soc_major_title"]: d for d in four["steer"]}
    assert four_by["Construction and Extraction Occupations"]["polarity"] == "male"
    assert four_by["Educational Instruction and Library Occupations"]["polarity"] == "female"


def test_select_domain_union() -> None:
    title = "Toy Occupations"
    both = select_domain(
        title,
        {
            "choice": {"rejected_fdr": True, "effect": 0.2, "q_value": 0.01},
            "prob": {"rejected_fdr": True, "effect": -0.9, "q_value": 0.01},
        },
    )
    assert both is not None
    assert both["gate"] == "both" and both["polarity"] == "male"
    only_prob = select_domain(
        title,
        {
            "choice": {"rejected_fdr": False, "effect": 0.01, "q_value": 0.8},
            "prob": {"rejected_fdr": True, "effect": -0.05, "q_value": 0.01},
        },
    )
    assert only_prob is not None and only_prob["gate"] == "prob_fdr"
    assert only_prob["polarity"] == "female"
    none = select_domain(
        title,
        {
            "choice": {"rejected_fdr": False, "effect": 0.2, "q_value": 0.2},
            "prob": {"rejected_fdr": False, "effect": -0.2, "q_value": 0.2},
        },
    )
    assert none is None


def test_slug_and_alpha_prior() -> None:
    assert slug_soc("Computer and Mathematical Occupations") == "computer_and_mathematical_occupations"
    assert hypothesized_alpha_sign("male") == "negative"
    assert hypothesized_alpha_sign("female") == "positive"
    assert polarity_of(0.1) == "male"
    assert polarity_of(-0.1) == "female"


def main() -> int:
    tests = [
        test_catalog_rebuilds,
        test_union_rule_and_polarity,
        test_select_domain_union,
        test_slug_and_alpha_prior,
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
