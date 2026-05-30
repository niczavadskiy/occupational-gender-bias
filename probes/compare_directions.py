"""
Pairwise cosine similarity between Ridge probe directions w per layer.

  python -m probes.compare_directions --evidence-mode no_evidence --target log_odds
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from probes.core import (
    add_common_args,
    direction_w_from_probe,
    fit_probe,
    load_canonical_family_split,
    load_probe_bundle,
    load_split_for_run,
)
from probes.h11 import EvidenceMode, build_h11_batch, target_vector
from probes.load_run import load_per_item
from probes.run_io import (
    ProbeTimer,
    build_probe_meta,
    h11_probe_fields,
    new_probe_run_dir,
    write_meta,
    write_results_json,
)


def fit_all_layer_directions(
    X_layers: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    *,
    target: str,
    ridge_alpha: float,
    clf_C: float,
    max_iter: int,
) -> np.ndarray:
    """Stack of unit-norm w vectors, shape (n_layers, d_model)."""
    n_layers = X_layers.shape[1]
    w_list: list[np.ndarray] = []
    for layer in range(n_layers):
        model = fit_probe(
            X_layers[:, layer, :],
            y,
            train_mask,
            target=target,
            ridge_alpha=ridge_alpha,
            clf_C=clf_C,
            max_iter=max_iter,
        )
        w_list.append(direction_w_from_probe(model, target=target))
    return np.stack(w_list, axis=0)


def cosine_matrix(w: np.ndarray) -> np.ndarray:
    """(n_layers, n_layers) cos(w_i, w_j); invalid layers (NaN w) -> NaN."""
    n = w.shape[0]
    out = np.full((n, n), np.nan, dtype=np.float64)
    valid = ~np.isnan(w).any(axis=1)
    idx = np.where(valid)[0]
    if len(idx):
        wv = w[idx]
        out[np.ix_(idx, idx)] = wv @ wv.T
    return out


def _cosine_summary(cos: np.ndarray, n_layers: int) -> dict[str, float | None]:
    upper = cos[np.triu_indices(n_layers, k=1)]
    finite = upper[np.isfinite(upper)]
    out: dict[str, float | None] = {
        "mean_cosine_w_off_diagonal": float(finite.mean()) if len(finite) else None,
        "min_cosine_w_off_diagonal": float(finite.min()) if len(finite) else None,
        "max_cosine_w_off_diagonal": float(finite.max()) if len(finite) else None,
        "layer0_vs_layer19_cosine_w": (
            float(cos[0, 19]) if n_layers > 19 and np.isfinite(cos[0, 19]) else None
        ),
    }
    return out


def pairwise_rows(cos: np.ndarray) -> list[dict[str, float | int]]:
    rows = []
    n = cos.shape[0]
    for i in range(n):
        for j in range(i, n):
            v = cos[i, j]
            rows.append(
                {
                    "layer_a": i,
                    "layer_b": j,
                    "cosine": float(v) if np.isfinite(v) else None,
                }
            )
    return rows


def projection_correlation_matrix(
    X_layers: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """corr(projection at layer i, projection at layer j) over all samples."""
    n_layers = w.shape[0]
    proj = np.stack([X_layers[:, L, :] @ w[L] for L in range(n_layers)], axis=1)
    return np.corrcoef(proj.T)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=10_000)
    args = parser.parse_args(argv)

    mode = EvidenceMode(args.evidence_mode)
    timer = ProbeTimer()

    try:
        run_dir, parent_meta, batch, X_layers = load_probe_bundle(args.run, mode)
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1

    batch_all = build_h11_batch(load_per_item(run_dir), EvidenceMode.ALL)
    load_canonical_family_split(
        run_dir,
        batch_all,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        force=args.force_split,
    )
    split = load_split_for_run(
        run_dir,
        batch,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        force=False,
    )
    y = target_vector(batch, args.target)

    w = fit_all_layer_directions(
        X_layers,
        y,
        split.train_mask,
        target=args.target,
        ridge_alpha=args.ridge_alpha,
        clf_C=args.C,
        max_iter=args.max_iter,
    )
    cos_w = cosine_matrix(w)
    cos_proj = projection_correlation_matrix(X_layers, w)
    pairs = pairwise_rows(cos_w)
    n_layers = int(w.shape[0])

    out_dir, probe_run_id = new_probe_run_dir(run_dir, "compare_directions", args.out_dir)
    stem = f"compare_directions_{mode.value}_{args.target}"
    csv_name = f"{stem}_cos_w_pairs.csv"
    matrix_csv = f"{stem}_cos_w_matrix.csv"

    with (out_dir / csv_name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer_a", "layer_b", "cosine"])
        writer.writeheader()
        writer.writerows(pairs)

    with (out_dir / matrix_csv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer"] + list(range(n_layers)))
        for i in range(n_layers):
            writer.writerow([i] + [float(cos_w[i, j]) for j in range(n_layers)])

    npz_name = f"{stem}_weights.npz"
    np.savez_compressed(
        out_dir / npz_name,
        w_raw=w.astype(np.float32),
        cosine_w=cos_w.astype(np.float32),
        projection_correlation=cos_proj.astype(np.float32),
        layers=np.arange(n_layers, dtype=np.int32),
    )

    results = {
        "n_layers": n_layers,
        "evidence_mode": mode.value,
        "target": args.target,
        "pairwise_cosine_w": pairs,
        "cosine_w_matrix": cos_w.tolist(),
        "projection_correlation_matrix": cos_proj.tolist(),
        "summary": _cosine_summary(cos_w, n_layers),
    }
    write_results_json(out_dir, results)

    extra = {
        **h11_probe_fields(batch, args.target, split),
        "summary": results["summary"],
    }
    meta = build_probe_meta(
        probe_script="compare_directions",
        probe_run_id=probe_run_id,
        parent_run_dir=run_dir,
        parent_meta=parent_meta,
        probe_runtime_s=timer.elapsed(),
        artifacts={
            csv_name: csv_name,
            matrix_csv: matrix_csv,
            npz_name: npz_name,
            "results.json": "results.json",
        },
        extra=extra,
    )
    print(f"Wrote {write_meta(out_dir, meta)}")
    print(
        f"mean off-diagonal cos(w): {results['summary']['mean_cosine_w_off_diagonal']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
