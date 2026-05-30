"""
Train / val / test split by scenario_family_id (= base_id = context × predicate).

All factorial rows of one family share the same split (no evidence leakage).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class GroupTVTSplit:
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    group_to_split: dict[int, SplitName]
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float

    def mask(self, name: SplitName) -> np.ndarray:
        if name == "train":
            return self.train_mask
        if name == "val":
            return self.val_mask
        return self.test_mask

    def split_name_per_row(self, family_ids: np.ndarray) -> np.ndarray:
        return np.array(
            [self.group_to_split[int(g)] for g in family_ids], dtype=object
        )

    def verify(self, family_ids: np.ndarray) -> None:
        for g in np.unique(family_ids):
            row_splits = self.split_name_per_row(family_ids[family_ids == g])
            if len(np.unique(row_splits)) != 1:
                raise ValueError(
                    f"scenario_family_id={g} spans splits: {np.unique(row_splits)}"
                )

    def summary(self) -> dict[str, Any]:
        counts = {
            s: sum(1 for v in self.group_to_split.values() if v == s)
            for s in ("train", "val", "test")
        }
        return {
            "group_key": "scenario_family_id (base_id = context×predicate)",
            "n_scenario_families": len(self.group_to_split),
            "families_per_split": counts,
            "seed": self.seed,
            "ratios": {
                "train": self.train_ratio,
                "val": self.val_ratio,
                "test": self.test_ratio,
            },
        }


def shared_split_dir(run_dir: Path) -> Path:
    return run_dir / "probes" / "_shared"


def split_json_path(run_dir: Path) -> Path:
    return shared_split_dir(run_dir) / "group_split_scenario_family.json"


def make_group_tvt_split(
    family_ids: np.ndarray,
    *,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 0,
) -> GroupTVTSplit:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {total}")

    unique_groups = np.unique(family_ids)
    rng = np.random.default_rng(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    if n_train + n_val > n:
        n_val = max(0, n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test < 1 and n >= 3:
        n_test = 1
        n_train = n - n_val - n_test

    train_g = {int(x) for x in shuffled[:n_train]}
    val_g = {int(x) for x in shuffled[n_train : n_train + n_val]}
    test_g = {int(x) for x in shuffled[n_train + n_val :]}

    group_to_split: dict[int, SplitName] = {}
    for g in train_g:
        group_to_split[g] = "train"
    for g in val_g:
        group_to_split[g] = "val"
    for g in test_g:
        group_to_split[g] = "test"

    missing = {int(g) for g in unique_groups} - set(group_to_split)
    if missing:
        raise RuntimeError(f"unassigned families: {missing}")

    per_row = np.array([group_to_split[int(g)] for g in family_ids], dtype=object)
    split = GroupTVTSplit(
        train_mask=per_row == "train",
        val_mask=per_row == "val",
        test_mask=per_row == "test",
        group_to_split=group_to_split,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    split.verify(family_ids)
    return split


def save_split(split: GroupTVTSplit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **split.summary(),
        "group_to_split": {str(k): v for k, v in split.group_to_split.items()},
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _parse_group_to_split(raw: dict) -> dict[int, SplitName]:
    out: dict[int, SplitName] = {}
    for k, v in raw.items():
        key = str(k)
        if "|" in key:
            raise ValueError(
                "Stale split uses base_id|evidence keys. Run with --force-split."
            )
        out[int(key)] = v
    return out


def load_split(path: Path, family_ids: np.ndarray) -> GroupTVTSplit:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)

    group_to_split = _parse_group_to_split(payload["group_to_split"])
    missing = {int(g) for g in np.unique(family_ids)} - set(group_to_split)
    if missing:
        raise ValueError(
            f"Split missing {len(missing)} families, e.g. {sorted(missing)[:3]}. "
            "--force-split to rebuild."
        )

    per_row = np.array([group_to_split[int(g)] for g in family_ids], dtype=object)
    split = GroupTVTSplit(
        train_mask=per_row == "train",
        val_mask=per_row == "val",
        test_mask=per_row == "test",
        group_to_split=group_to_split,
        seed=int(payload.get("seed", 0)),
        train_ratio=float(payload["ratios"]["train"]),
        val_ratio=float(payload["ratios"]["val"]),
        test_ratio=float(payload["ratios"]["test"]),
    )
    split.verify(family_ids)
    return split


def align_split(split: GroupTVTSplit, family_ids: np.ndarray) -> GroupTVTSplit:
    """Row masks for a batch subset using the same family→split assignment."""
    per_row = np.array([split.group_to_split[int(g)] for g in family_ids], dtype=object)
    aligned = GroupTVTSplit(
        train_mask=per_row == "train",
        val_mask=per_row == "val",
        test_mask=per_row == "test",
        group_to_split=split.group_to_split,
        seed=split.seed,
        train_ratio=split.train_ratio,
        val_ratio=split.val_ratio,
        test_ratio=split.test_ratio,
    )
    aligned.verify(family_ids)
    return aligned


def get_or_create_split(
    family_ids: np.ndarray,
    run_dir: Path,
    *,
    seed: int = 0,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    force: bool = False,
) -> GroupTVTSplit:
    """Build or load family-level split; masks match len(family_ids) rows."""
    path = split_json_path(run_dir)
    unique_families = np.unique(family_ids)

    if path.is_file() and not force:
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        group_to_split = _parse_group_to_split(payload["group_to_split"])
        missing = {int(g) for g in unique_families} - set(group_to_split)
        if missing:
            raise ValueError(
                f"Split file missing families {sorted(missing)[:5]}. --force-split."
            )
        base = GroupTVTSplit(
            train_mask=np.array([]),
            val_mask=np.array([]),
            test_mask=np.array([]),
            group_to_split=group_to_split,
            seed=int(payload.get("seed", 0)),
            train_ratio=float(payload["ratios"]["train"]),
            val_ratio=float(payload["ratios"]["val"]),
            test_ratio=float(payload["ratios"]["test"]),
        )
        return align_split(base, family_ids)

    split_full = make_group_tvt_split(
        unique_families,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    save_split(split_full, path)
    return align_split(split_full, family_ids)
