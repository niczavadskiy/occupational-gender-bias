"""
Orthogonal gender probing: remove narrative / slot linear components from HS,
then re-decode gender_choice.

Used by v1_rep_summary to report «чистый» gender-сигнал vs confounded raw probe.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from probes.core import (
    classification_metrics,
    direction_w_from_probe,
    eval_split,
    fit_probe,
    load_split_for_run,
)
from probes.v1_rep import V1_PROBE_ROOT, load_v1_rep_bundle, v1_pipeline_run_slug

DEFAULT_CLF_C = 1.0
DEFAULT_MAX_ITER = 10_000


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.full_like(v, np.nan)
    return v / n


def _orthonormal_basis(*vectors: np.ndarray) -> np.ndarray:
    """Columns of Q form orthonormal basis for span(vectors), shape (d, k)."""
    cols = []
    for v in vectors:
        v = v.astype(np.float64)
        if not np.isfinite(v).all():
            continue
        for q in cols:
            v = v - float(v @ q) * q
        n = float(np.linalg.norm(v))
        if n > 1e-8:
            cols.append(v / n)
    if not cols:
        return np.zeros((vectors[0].shape[0], 0), dtype=np.float64)
    return np.stack(cols, axis=1)


def _residualize(X: np.ndarray, Q: np.ndarray) -> np.ndarray:
    if Q.shape[1] == 0:
        return X.copy()
    return X - (X @ Q) @ Q.T


def _orthogonalize(w: np.ndarray, basis: np.ndarray) -> np.ndarray:
    w = w.astype(np.float64).copy()
    for j in range(basis.shape[1]):
        q = basis[:, j]
        w = w - float(w @ q) * q
    return _unit(w)


def _bacc_from_scores(
    scores: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
) -> float:
    """Balanced accuracy with threshold = midpoint of class means on train."""
    y = y.astype(int)
    s_tr = scores[train_mask]
    y_tr = y[train_mask]
    m0 = float(s_tr[y_tr == 0].mean()) if (y_tr == 0).any() else 0.0
    m1 = float(s_tr[y_tr == 1].mean()) if (y_tr == 1).any() else 0.0
    thr = 0.5 * (m0 + m1)
    pred = (scores[eval_mask] > thr).astype(int)
    return float(classification_metrics(y[eval_mask], pred, None)["balanced_accuracy"])


def _label_correlations(y_g: np.ndarray, y_n: np.ndarray, y_s: np.ndarray) -> dict[str, float]:
    def corr(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 2:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "gender_narrative": corr(y_g, y_n),
        "gender_slot": corr(y_g, y_s),
        "narrative_slot": corr(y_n, y_s),
    }


def compute_orthogonal_gender_analysis(
    run_dir: Path,
    *,
    layer: int | None = None,
    clf_C: float = DEFAULT_CLF_C,
    max_iter: int = DEFAULT_MAX_ITER,
    seed: int = 0,
) -> dict[str, Any]:
    """Gender decoding before/after removing narrative+slot directions at a fixed layer."""
    _run_dir, _meta, batch, X_layers = load_v1_rep_bundle(run_dir.name)
    split = load_split_for_run(
        run_dir,
        batch,
        seed=seed,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        force=False,
    )

    if layer is None:
        meta_path = (
            run_dir / "probes" / V1_PROBE_ROOT / v1_pipeline_run_slug("gender_choice") / "meta.json"
        )
        with meta_path.open(encoding="utf-8") as f:
            g_meta = json.load(f)
        layer = int(g_meta["stages"]["layer_scan"]["summary"]["best_layer"])

    X = X_layers[:, layer, :].astype(np.float64)
    y_g = batch.gender_choice.astype(np.float64)
    y_n = batch.narrative_choice.astype(np.float64)
    y_s = batch.slot_choice.astype(np.float64)
    train = split.train_mask
    test = split.test_mask
    val = split.val_mask

    kw = dict(target="gender_choice", ridge_alpha=1.0, clf_C=clf_C, max_iter=max_iter)

    raw_test = eval_split(X, y_g, train, test, **kw)
    raw_val = eval_split(X, y_g, train, val, **kw)

    model_n = fit_probe(X, y_n, train, target="narrative_choice", ridge_alpha=1.0, clf_C=clf_C, max_iter=max_iter)
    model_s = fit_probe(X, y_s, train, target="slot_choice", ridge_alpha=1.0, clf_C=clf_C, max_iter=max_iter)
    model_g = fit_probe(X, y_g, train, **kw)

    w_g = direction_w_from_probe(model_g, target="gender_choice")
    w_n = direction_w_from_probe(model_n, target="narrative_choice")
    w_s = direction_w_from_probe(model_s, target="slot_choice")

    Q_ns = _orthonormal_basis(w_n, w_s)
    cos = {
        "gender_narrative": float(w_g @ Q_ns[:, 0]) if Q_ns.shape[1] >= 1 else float("nan"),
        "gender_slot": float(w_g @ Q_ns[:, 1]) if Q_ns.shape[1] >= 2 else float("nan"),
        "narrative_slot": float(Q_ns[:, 0] @ Q_ns[:, 1]) if Q_ns.shape[1] >= 2 else 0.0,
    }

    w_g_perp = _orthogonalize(w_g, Q_ns)
    scores_perp = X @ w_g_perp
    perp_test_bacc = _bacc_from_scores(scores_perp, y_g.astype(int), train, test)
    perp_val_bacc = _bacc_from_scores(scores_perp, y_g.astype(int), train, val)

    # HS residualization: remove span{narrative, slot} then re-fit gender probe.
    X_res = _residualize(X, Q_ns)
    res_test = eval_split(X_res, y_g, train, test, **kw)
    res_val = eval_split(X_res, y_g, train, val, **kw)

    # Ablations: remove only narrative or only slot.
    Q_n = _orthonormal_basis(w_n)
    Q_s = _orthonormal_basis(w_s)
    res_n_test = eval_split(_residualize(X, Q_n), y_g, train, test, **kw)
    res_s_test = eval_split(_residualize(X, Q_s), y_g, train, test, **kw)

    label_corr = _label_correlations(y_g, y_n, y_s)

    return {
        "layer": layer,
        "n_total": int(len(y_g)),
        "n_test": int(test.sum()),
        "label_correlations": label_corr,
        "direction_cosines": cos,
        "raw": {
            "val_balanced_accuracy": raw_val["balanced_accuracy"],
            "test_balanced_accuracy": raw_test["balanced_accuracy"],
            "test_roc_auc": raw_test.get("roc_auc"),
        },
        "orthogonal_score": {
            "val_balanced_accuracy": perp_val_bacc,
            "test_balanced_accuracy": perp_test_bacc,
            "description": "score = h·w_g⊥, threshold from train class means",
        },
        "residual_hs_refit": {
            "val_balanced_accuracy": res_val["balanced_accuracy"],
            "test_balanced_accuracy": res_test["balanced_accuracy"],
            "test_roc_auc": res_test.get("roc_auc"),
            "confounds": "narrative + slot",
        },
        "ablation": {
            "remove_narrative_only": res_n_test["balanced_accuracy"],
            "remove_slot_only": res_s_test["balanced_accuracy"],
        },
        "slugs": {
            "gender": v1_pipeline_run_slug("gender_choice"),
            "narrative": v1_pipeline_run_slug("narrative_choice"),
            "slot": v1_pipeline_run_slug("slot_choice"),
        },
    }
