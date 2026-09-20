"""Load XY-control items. Default pool is the full without_abstain set (951×4)."""

from __future__ import annotations

from pathlib import Path

from steering.run_inlp_stagea import load_json

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FULL_SPLIT = "full"
LEGACY_SPLITS = ("stagea", "stageb", "test", "stagec")
POOLED_SPLITS = (FULL_SPLIT,)


def split_path(name: str, data_dir: Path | None = None) -> Path:
    return (data_dir or DATA_DIR) / f"xy_pairs_{name}_v1.json"


def load_full_items(data_dir: Path | None = None) -> list[dict]:
    path = split_path(FULL_SPLIT, data_dir)
    if not path.is_file():
        raise FileNotFoundError(f"{path} — python -m steering.xy_control.build_dataset")
    doc = load_json(path)
    items = []
    for item in doc["items"]:
        row = dict(item)
        row["xy_split"] = FULL_SPLIT
        items.append(row)
    items.sort(key=lambda it: int(it["scenario_family_id"]))
    return items


def load_pooled_items(
    data_dir: Path | None = None,
    *,
    splits: tuple[str, ...] | None = None,
) -> list[dict]:
    """Union of XY splits. Default is the full without_abstain file."""
    names = splits if splits is not None else POOLED_SPLITS
    if names == (FULL_SPLIT,) or names == ("full",):
        return load_full_items(data_dir)
    items: list[dict] = []
    seen: set[int] = set()
    for name in names:
        path = split_path(name, data_dir)
        if not path.is_file():
            raise FileNotFoundError(path)
        doc = load_json(path)
        for item in doc["items"]:
            fid = int(item["scenario_family_id"])
            if fid in seen:
                raise ValueError(f"duplicate scenario_family_id {fid} in {path.name}")
            seen.add(fid)
            row = dict(item)
            row["xy_split"] = name
            items.append(row)
    items.sort(key=lambda it: int(it["scenario_family_id"]))
    return items


def filter_soc(items: list[dict], title: str) -> list[dict]:
    return [it for it in items if it.get("soc_major_title") == title]


def family_ids(items: list[dict]) -> list[int]:
    return [int(it["scenario_family_id"]) for it in items]
