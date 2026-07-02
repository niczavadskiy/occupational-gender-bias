"""Load v1 H3 merged per_item.jsonl (baseline + highlight MAN/WOMAN)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.metrics.load import (
    load_per_item_jsonl,
    prepare_metrics_frame,
    resolve_run_dir,
)

H3_V1_ROW_COUNT = 45648
V1_BASE_ROW_COUNTS = frozenset({7608, 15216})
H3_V1_ROW_COUNTS = frozenset({H3_V1_ROW_COUNT})

EVIDENCE_ALIASES = {
    "no_evidence": "no_evidence",
    "man": "man",
    "woman": "woman",
    "evidence_supports_man": "man",
    "evidence_supports_woman": "woman",
}
EVIDENCE_LEVELS = ("no_evidence", "man", "woman")


def normalize_evidence_shift(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "evidence_shift" not in out.columns:
        raise ValueError("evidence_shift column required for h3_v1")
    out["evidence_shift"] = (
        out["evidence_shift"].astype(str).map(lambda x: EVIDENCE_ALIASES.get(x, x))
    )
    unknown = set(out["evidence_shift"].unique()) - set(EVIDENCE_LEVELS)
    if unknown:
        raise ValueError(f"unknown evidence_shift values: {sorted(unknown)}")
    return out


def load_h3_dataframe(run_dir: Path) -> pd.DataFrame:
    rows = load_per_item_jsonl(run_dir)
    df = pd.DataFrame(rows)
    n = len(df)
    if n not in H3_V1_ROW_COUNTS:
        raise ValueError(
            f"Unexpected row count {n} in {run_dir}. "
            f"Expected h3_v1 layout: {sorted(H3_V1_ROW_COUNTS)} "
            f"(45648 = no_evidence + man + woman highlights merged)."
        )
    return normalize_evidence_shift(df)


def load_prepared_h3(
    run: str | Path | None = None,
    *,
    position_variant: str | None = None,
    context_order: str | None = None,
) -> tuple[Path, pd.DataFrame, int]:
    run_dir = resolve_run_dir(run)
    df_raw = load_h3_dataframe(run_dir)
    n_raw = len(df_raw)
    df = prepare_metrics_frame(
        df_raw,
        position_variant=position_variant,
        context_order=context_order,
    )
    return run_dir, df, n_raw
