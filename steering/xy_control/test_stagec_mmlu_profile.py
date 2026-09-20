"""CPU checks for the XY Stage C 250-question MMLU profile."""

from __future__ import annotations

import json
from pathlib import Path

from steering.xy_control.build_stagec_mmlu_profile import (
    DEFAULT_OUT,
    DEFAULT_VAL,
    fdr_titles,
    reserve_val_ids,
)

STEERING_DIR = Path(__file__).resolve().parents[1]


def test_stagec_xy250_profile_frozen() -> None:
    assert DEFAULT_OUT.is_file(), DEFAULT_OUT
    doc = json.loads(DEFAULT_OUT.read_text(encoding="utf-8"))
    titles = fdr_titles()
    assert doc["n_domains"] == len(titles) == 13
    assert doc["sampling"]["n_per_soc"] == 250
    assert doc["n_questions"] == 13 * 250
    seen: set[int] = set()
    for domain in doc["domains"]:
        assert domain["soc_major_title"] in titles
        assert domain["n"] == 250
        ids = [int(q) for q in domain["question_ids"]]
        assert len(ids) == len(set(ids)) == 250
        assert not (seen & set(ids))
        seen.update(ids)
    reserved = reserve_val_ids(DEFAULT_VAL, titles)
    assert not (seen & reserved)


def main() -> None:
    test_stagec_xy250_profile_frozen()
    print("ok")


if __name__ == "__main__":
    main()
