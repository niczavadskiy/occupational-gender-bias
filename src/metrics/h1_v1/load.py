"""Load v1 per_item.jsonl (7608 single-context, 15216 merged)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.metrics.load import (
    load_per_item_jsonl,
    prepare_metrics_frame,
    resolve_run_dir,
)

V1_ROW_COUNTS = frozenset({7608, 15216})


def load_v1_dataframe(run_dir: Path) -> pd.DataFrame:
    rows = load_per_item_jsonl(run_dir)
    df = pd.DataFrame(rows)
    n = len(df)
    if n not in V1_ROW_COUNTS:
        raise ValueError(
            f"Unexpected row count {n} in {run_dir}. "
            f"Expected v1 layouts: {sorted(V1_ROW_COUNTS)} "
            f"(7608 single context_order; 15216 merged man_first+woman_first)."
        )
    return df


def load_prepared_v1(
    run: str | Path | None = None,
    *,
    position_variant: str | None = None,
    context_order: str | None = None,
) -> tuple[Path, pd.DataFrame, int]:
    run_dir = resolve_run_dir(run)
    df_raw = load_v1_dataframe(run_dir)
    n_raw = len(df_raw)
    df = prepare_metrics_frame(
        df_raw,
        position_variant=position_variant,
        context_order=context_order,
    )
    return run_dir, df, n_raw
