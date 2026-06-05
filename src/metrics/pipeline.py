"""Orchestrate behavioral metrics pipeline."""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Any

from src.metrics.fdr import apply_fdr_families
from src.metrics.families import FDR_FAMILIES, H2_ENABLED, h1_fdr_families, merge_answerability_fdr_families
from src.metrics.hypotheses import resolve_h4_formats
from src.metrics.annotations import resolve_annotation_jsonl
from src.metrics.hypotheses import run_all, run_post_hoc_h
from src.metrics.labels import add_derived_columns
from src.metrics.load import (
    load_dataframe,
    main_task_frame,
    metrics_position_variant,
    prepare_metrics_frame,
    resolve_run_dir,
)
from src.metrics.rates import build_rates_by_family, build_rates_summary
from src.metrics.reliability import apply_reliability_labels
from src.metrics.report import write_outputs

DEFAULT_HYPOTHESES = [
    h for h in ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9") if h != "H2" or H2_ENABLED
]


def make_out_dir(run_dir: Path, out: Path | None = None) -> Path:
    if out is not None:
        p = Path(out)
        p.mkdir(parents=True, exist_ok=True)
        return p
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    p = run_dir / "metrics" / stamp
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_pipeline(
    run: str | Path | None = None,
    *,
    hypotheses: list[str] | None = None,
    fdr: float = 0.05,
    level: str = "family",
    out_dir: Path | None = None,
    annotation_path: Path | None = None,
    h4_formats: str | tuple[str, ...] | None = None,
    position_variant: str | None = None,
    context_order: str | None = None,
) -> Path:
    run_dir = run if isinstance(run, Path) else resolve_run_dir(run)
    hypo = hypotheses or list(DEFAULT_HYPOTHESES)
    if level not in ("row", "family"):
        raise ValueError(f"level must be row|family, got {level!r}")

    df_raw = load_dataframe(run_dir)
    df = prepare_metrics_frame(
        df_raw,
        position_variant=position_variant,
        context_order=context_order,
    )
    df = add_derived_columns(df)
    df_main = main_task_frame(df)

    n_ties = int(df_main["tie_prob"].sum())
    n_abstain = int(df_main["abstain"].sum())

    rates_summary = build_rates_summary(df_main)
    rates_family = build_rates_by_family(df_main, question_format="choice")

    ann_jsonl = resolve_annotation_jsonl(annotation_path) if "H2" in hypo and H2_ENABLED else None
    need_ans = any(h in hypo for h in ("H4", "H7", "H8", "H9"))
    h4_strata = resolve_h4_formats(h4_formats) if h4_formats is not None else None

    results = run_all(
        df,
        hypotheses=hypo,
        level=level,
        annotation_path=annotation_path,
        h4_formats=h4_strata,
    )
    post_hoc = run_post_hoc_h(df_main, level)

    apply_reliability_labels(results)
    apply_reliability_labels(post_hoc)

    families = copy.deepcopy(FDR_FAMILIES)
    if "H1" in hypo:
        for key in list(families):
            if key.startswith("h1_strata"):
                del families[key]
        families.update(h1_fdr_families(df_main))
    if need_ans:
        families = merge_answerability_fdr_families(families, h4_strata or (), hypo)
    apply_fdr_families(results, families, q=fdr)

    out = make_out_dir(run_dir, out_dir)
    write_outputs(
        out,
        run_dir=run_dir,
        results=results,
        post_hoc_results=post_hoc,
        rates_summary=rates_summary,
        rates_family=rates_family,
        level=level,
        fdr=fdr,
        hypotheses=hypo,
        sanity={"n_ties_prob_equal": n_ties, "n_choice_c_abstain": n_abstain},
        annotation_jsonl=str(ann_jsonl) if ann_jsonl else None,
        h4_formats=list(h4_strata) if h4_strata else None,
        position_variant=df.attrs.get("metrics_position_variant"),
        context_order=df.attrs.get("metrics_context_order"),
        n_rows_raw=len(df_raw),
        n_rows_metrics=len(df),
    )
    return out
