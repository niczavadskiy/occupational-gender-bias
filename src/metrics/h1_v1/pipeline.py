"""Orchestrate H1 v1 metrics pipeline."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.metrics.fdr import apply_fdr_families
from src.metrics.load import main_task_frame
from src.metrics.reliability import apply_reliability_labels

from src.metrics.h1_v1.families import H1V1_PRIMARY_TEST_PREFIX, h1_v1_fdr_families
from src.metrics.h1_v1.hypotheses import DEFAULT_MIN_SOC_G, DEFAULT_MIN_STRATUM_N, run_h1_v1
from src.metrics.h1_v1.labels import add_derived_columns
from src.metrics.h1_v1.load import load_prepared_v1
from src.metrics.h1_v1.rates import build_rates_by_scenario, build_rates_summary
from src.metrics.h1_v1.report import write_outputs


def make_out_dir(run_dir: Path, out: Path | None = None) -> Path:
    if out is not None:
        p = Path(out)
        p.mkdir(parents=True, exist_ok=True)
        return p
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    p = run_dir / "metrics_h1_v1" / stamp
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_h1_v1_pipeline(
    run: str | Path | None = None,
    *,
    fdr: float = 0.05,
    out_dir: Path | None = None,
    min_stratum_n: int = DEFAULT_MIN_STRATUM_N,
    min_soc_g: int = DEFAULT_MIN_SOC_G,
    include_soc_strata: bool = True,
    include_onet_strata: bool = True,
    position_variant: str | None = None,
    context_order: str | None = None,
) -> Path:
    run_dir, df, n_rows_raw = load_prepared_v1(
        run,
        position_variant=position_variant,
        context_order=context_order,
    )
    df = add_derived_columns(df)
    df_main = main_task_frame(df)

    results = run_h1_v1(
        df_main,
        min_stratum_n=min_stratum_n,
        min_soc_g=min_soc_g,
        include_soc_strata=include_soc_strata,
        include_onet_strata=include_onet_strata,
    )

    apply_reliability_labels(results)
    families = h1_v1_fdr_families(df_main, [r.test_id for r in results])
    primary_ids = {r.test_id for r in results if r.test_id.startswith(H1V1_PRIMARY_TEST_PREFIX)}
    apply_fdr_families(results, families, q=fdr, exclude_test_ids=primary_ids)

    rates_summary = build_rates_summary(df_main)
    rates_scenario = build_rates_by_scenario(df_main)

    out = make_out_dir(run_dir, out_dir)
    write_outputs(
        out,
        run_dir=run_dir,
        results=results,
        rates_summary=rates_summary,
        rates_scenario=rates_scenario,
        df=df_main,
        fdr=fdr,
        min_stratum_n=min_stratum_n,
        min_soc_g=min_soc_g,
        n_rows_raw=n_rows_raw,
        n_rows_metrics=len(df_main),
        position_variant=df.attrs.get("metrics_position_variant"),
        context_order=df.attrs.get("metrics_context_order"),
    )
    return out
