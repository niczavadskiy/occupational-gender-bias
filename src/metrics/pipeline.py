"""Orchestrate behavioral metrics pipeline."""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Any

from src.metrics.fdr import apply_fdr_families
from src.metrics.families import FDR_FAMILIES
from src.metrics.annotations import resolve_annotation_jsonl
from src.metrics.hypotheses import run_all, run_post_hoc_h
from src.metrics.labels import add_derived_columns
from src.metrics.load import load_dataframe, resolve_run_dir
from src.metrics.rates import build_rates_by_family, build_rates_summary
from src.metrics.reliability import apply_reliability_labels
from src.metrics.report import write_outputs

DEFAULT_HYPOTHESES = ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9"]


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
) -> Path:
    run_dir = resolve_run_dir(run)
    hypo = hypotheses or list(DEFAULT_HYPOTHESES)
    if level not in ("row", "family"):
        raise ValueError(f"level must be row|family, got {level!r}")

    df = load_dataframe(run_dir)
    df = add_derived_columns(df)

    n_ties = int(df["tie_prob"].sum())
    n_abstain = int(df["abstain"].sum())

    rates_summary = build_rates_summary(df)
    rates_family = build_rates_by_family(df, question_format="choice")

    ann_jsonl = resolve_annotation_jsonl(annotation_path) if "H2" in hypo else None

    results = run_all(
        df,
        hypotheses=hypo,
        level=level,
        annotation_path=annotation_path,
    )
    post_hoc = run_post_hoc_h(df, level)

    apply_reliability_labels(results)
    apply_reliability_labels(post_hoc)

    families = copy.deepcopy(FDR_FAMILIES)
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
    )
    return out
