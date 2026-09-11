"""Shared probe utilities: data loading, models, metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from probes.h10 import H10Batch, H10_LAYOUT_V3_MAIN, collect_h10_v3_main_group_ids, h10_split_json_path
from probes.h11 import EvidenceMode, H11Batch, build_h11_batch, h11_split_json_path
from probes.h12 import H12Batch
from probes.h13 import H13Batch, h13_split_json_path
from probes.v1_rep import V1RepBatch, v1_split_json_path
from probes.load_run import load_run
from probes.paths import DEFAULT_RUN_NAME
from probes.splits import GroupTVTSplit, get_or_create_split

REGRESSION_TARGETS = (
    "log_odds",
    "prob_margin",
    "abstain_logit",
    "log_prob_abstain",
    "h10_st_logit",
    "h10_log_prob_st",
    "h13_yes_logit",
    "gender_prob",
    "narrative_prob",
    "slot_prob",
)
CLASSIFICATION_TARGETS = (
    "choice",
    "choice_abstain",
    "h10_choice_stereotype",
    "gender_choice",
    "narrative_choice",
    "slot_choice",
)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", default=DEFAULT_RUN_NAME)
    parser.add_argument(
        "--evidence-mode",
        choices=[m.value for m in EvidenceMode],
        default=EvidenceMode.ALL.value,
    )
    parser.add_argument(
        "--target",
        choices=(
            "log_odds",
            "prob_margin",
            "choice",
            "abstain_logit",
            "log_prob_abstain",
            "choice_abstain",
            "h10_st_logit",
            "h10_log_prob_st",
            "h10_choice_stereotype",
            "h13_yes_logit",
        ),
        default="log_odds",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--force-split", action="store_true")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Override probe run output dir (default: probes/<script>/<timestamp>/)",
    )


def load_probe_bundle(
    run: str | None,
    evidence_mode: EvidenceMode,
    *,
    abstain_variant: str = "without_abstain",
    question_formats: tuple[str, ...] | None = None,
) -> tuple[Path, dict, H11Batch, np.ndarray]:
    from probes.h11 import H11_QUESTION_FORMATS

    qf = H11_QUESTION_FORMATS if question_formats is None else question_formats
    run_dir, meta, items, hs = load_run(run)
    batch = build_h11_batch(
        items,
        evidence_mode=evidence_mode,
        abstain_variant=abstain_variant,
        question_formats=qf,
    )
    # Materialize only selected rows as float32 (smaller peak memory than casting full hs).
    X_layers = np.asarray(hs[batch.indices], dtype=np.float32)
    return run_dir, meta, batch, X_layers


def _v3_split_options(batch: H10Batch | H11Batch | H12Batch | H13Batch, run_dir: Path) -> dict[str, Path | str]:
    if batch.layout != H10_LAYOUT_V3_MAIN:
        return {}
    if isinstance(batch, H13Batch):
        return {
            "split_path": h13_split_json_path(run_dir),
            "group_key": batch.group_key_label(),
        }
    if isinstance(batch, H11Batch):
        return {
            "split_path": h11_split_json_path(run_dir, batch.layout),
            "group_key": batch.group_key_label(),
        }
    return {
        "split_path": h10_split_json_path(run_dir, batch.layout),
        "group_key": batch.group_key_label(),
    }


def load_split_for_run(
    run_dir: Path,
    batch: H11Batch | H10Batch | H12Batch | H13Batch | V1RepBatch,
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    force: bool,
) -> GroupTVTSplit:
    """Split aligned to this batch's rows (group ids may repeat across evidence)."""
    extra: dict[str, Path | str] = {}
    if isinstance(batch, V1RepBatch):
        extra = {
            "split_path": v1_split_json_path(run_dir),
            "group_key": batch.group_key_label(),
        }
    elif isinstance(batch, (H10Batch, H11Batch, H12Batch, H13Batch)):
        extra = _v3_split_options(batch, run_dir)
    return get_or_create_split(
        batch.scenario_family_id,
        run_dir,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        force=force,
        **extra,
    )


def load_canonical_family_split(
    run_dir: Path,
    items_batch_all: H11Batch | H10Batch | H12Batch | H13Batch | V1RepBatch,
    **kwargs,
) -> GroupTVTSplit:
    """Ensure split file lists all probe groups (use ALL evidence batch once)."""
    unique = np.unique(items_batch_all.scenario_family_id)
    extra: dict[str, Path | str] = {}
    if isinstance(items_batch_all, V1RepBatch):
        extra = {
            "split_path": v1_split_json_path(run_dir),
            "group_key": items_batch_all.group_key_label(),
        }
    elif isinstance(items_batch_all, (H10Batch, H11Batch, H12Batch, H13Batch)):
        extra = _v3_split_options(items_batch_all, run_dir)
    return get_or_create_split(unique, run_dir, **kwargs, **extra)


def make_regressor(alpha: float = 1.0) -> Pipeline:
    return Pipeline(
        [("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))]
    )


def make_classifier(C: float = 1.0, max_iter: int = 10_000) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    class_weight="balanced",
                    random_state=0,
                ),
            ),
        ]
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) < 2:
        return {"r2": float("nan"), "pearson_r": float("nan"), "mse": float("nan")}
    r2 = float(r2_score(y_true, y_pred))
    corr = (
        float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_pred) > 0 else float("nan")
    )
    mse = float(np.mean((y_true - y_pred) ** 2))
    return {"r2": r2, "pearson_r": corr, "mse": mse}


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None
) -> dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    if y_proba is not None and len(np.unique(y_true)) >= 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            out["roc_auc"] = float("nan")
    else:
        out["roc_auc"] = float("nan")
    return out


def fit_probe(
    X: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    *,
    target: str,
    ridge_alpha: float,
    clf_C: float,
    max_iter: int,
) -> Pipeline:
    if target in REGRESSION_TARGETS:
        model = make_regressor(ridge_alpha)
    else:
        if len(np.unique(y[train_mask])) < 2:
            raise ValueError("classification requires >=2 classes in train")
        model = make_classifier(clf_C, max_iter)
    model.fit(X[train_mask], y[train_mask])
    return model


def _probe_coef_and_intercept(model: Pipeline, *, target: str) -> tuple[np.ndarray, float]:
    if target in CLASSIFICATION_TARGETS:
        step = model.named_steps["clf"]
        intercept = float(np.asarray(step.intercept_).ravel()[0])
    else:
        step = model.named_steps["ridge"]
        intercept = float(np.asarray(step.intercept_).ravel()[0]) if hasattr(step, "intercept_") else 0.0
    return np.asarray(step.coef_, dtype=np.float64).ravel(), intercept


def raw_space_logit_params(model: Pipeline, *, target: str) -> tuple[np.ndarray, float]:
    """Hyperplane in raw HS: decision_function(h) = h @ w_raw + b_raw.

    Inverts StandardScaler: ĥ = (h − μ) / σ, so
        w_raw,j = coef_j / σ_j
        b_raw   = intercept − w_raw · μ
    w_raw is *not* unit-normalized (unlike direction_w_from_probe).
    """
    coef, intercept = _probe_coef_and_intercept(model, target=target)
    scaler = model.named_steps["scaler"]
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    w_raw = coef / np.where(scale > 0, scale, 1.0)
    b_raw = intercept - float(w_raw @ mean)
    return w_raw, b_raw


def direction_w_from_probe(model: Pipeline, *, target: str) -> np.ndarray:
    """Unit-norm probe direction in raw hidden-state space (inverse scaler)."""
    w_raw, _ = raw_space_logit_params(model, target=target)
    norm = float(np.linalg.norm(w_raw))
    if norm < 1e-8:
        return np.full_like(w_raw, np.nan, dtype=np.float64)
    return w_raw / norm


def predict_probe(
    model: Pipeline,
    X: np.ndarray,
    eval_mask: np.ndarray,
    *,
    target: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    pred = model.predict(X[eval_mask])
    proba = None
    if target in CLASSIFICATION_TARGETS and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X[eval_mask])[:, 1]
    return pred, proba


def eval_split(
    X: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    *,
    target: str,
    ridge_alpha: float,
    clf_C: float,
    max_iter: int,
) -> dict[str, float]:
    model = fit_probe(
        X,
        y,
        train_mask,
        target=target,
        ridge_alpha=ridge_alpha,
        clf_C=clf_C,
        max_iter=max_iter,
    )
    pred, proba = predict_probe(model, X, eval_mask, target=target)
    y_eval = y[eval_mask]
    if target in CLASSIFICATION_TARGETS:
        return classification_metrics(y_eval.astype(int), pred.astype(int), proba)
    return regression_metrics(y_eval, pred)
