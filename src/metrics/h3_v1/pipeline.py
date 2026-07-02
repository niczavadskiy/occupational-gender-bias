"""Orchestrate H3 v1 metrics pipeline."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.metrics.fdr import apply_fdr_families
from src.metrics.load import main_task_frame
from src.metrics.reliability import apply_reliability_labels

from src.metrics.h3_v1.families import h3_v1_fdr_families
from src.metrics.h3_v1.hypotheses import DEFAULT_MIN_STRATUM_N, run_h3_v1
from src.metrics.h3_v1.labels import add_derived_columns
from src.metrics.h3_v1.load import load_prepared_h3
from src.metrics.h3_v1.rates import build_evidence_flip_rates, build_rates_summary
from src.metrics.h3_v1.report import write_outputs


def make_out_dir(run_dir: Path, out: Path | None = None) -> Path:
    if out is not None:
        p = Path(out)
        p.mkdir(parents=True, exist_ok=True)
        return p
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    p = run_dir / "metrics_h3_v1" / stamp
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_h3_v1_pipeline(
    run: str | Path | None = None,
    *,
    fdr: float = 0.05,
    out_dir: Path | None = None,
    min_stratum_n: int = DEFAULT_MIN_STRATUM_N,
    include_soc_strata: bool = True,
    position_variant: str | None = None,
    context_order: str | None = None,
) -> Path:
    run_dir, df, n_rows_raw = load_prepared_h3(
        run,
        position_variant=position_variant,
        context_order=context_order,
    )
    df = add_derived_columns(df)
    df_main = main_task_frame(df)

    results = run_h3_v1(
        df_main,
        min_stratum_n=min_stratum_n,
        include_soc_strata=include_soc_strata,
    )

    apply_reliability_labels(results)
    by_id = {r.test_id: r for r in results}
    families = h3_v1_fdr_families(df_main, [r.test_id for r in results], by_id)
    apply_fdr_families(results, families, q=fdr)

    rates_summary = build_rates_summary(df_main)
    flip_rates = build_evidence_flip_rates(df_main)

    out = make_out_dir(run_dir, out_dir)
    write_outputs(
        out,
        run_dir=run_dir,
        results=results,
        rates_summary=rates_summary,
        flip_rates=flip_rates,
        df=df_main,
        fdr=fdr,
        min_stratum_n=min_stratum_n,
        n_rows_raw=n_rows_raw,
        n_rows_metrics=len(df_main),
        position_variant=df.attrs.get("metrics_position_variant"),
        context_order=df.attrs.get("metrics_context_order"),
    )
    return out
