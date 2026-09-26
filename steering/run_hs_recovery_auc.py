"""
Method 3 — layer-wise HS / AUC recovery after a single-layer erase.

Intervene once (probe center or INLP rank-k center) on ``--intervene-layer``,
capture last-token residual on ``--read-layers``, then ask whether the linear
gender signal reappears downstream.

Metrics per read layer ℓ′ (held-out families):
  1. frozen_ŵ   — ROC-AUC of s = h·ŵ_ℓ′ (if vector bank has that layer)
                  + mean |s − c|
  2. mean_diff  — direction fit on *baseline* train HS only:
                  w = mean(h|y=1) − mean(h|y=0); same w scored on
                  baseline test and steered test (no refit on steered)

y = gender_choice from frozen sample ``baseline_choice``:
  1 iff labels[baseline_choice] == "man".

Interpretation (see analysis notes):
  AUC drops at intervene layer and stays low below  → incomplete erase / no recovery
  AUC drops at intervene layer, rises again below → representational recovery (world B)
  AUC stays high at intervene layer               → erase missed the linear feature

Does NOT prove causality of the recovered feature (use sequential second-hit for that).

Examples:

    # INLP keep-case (2B Stage C): L15 k16, read L15–24
    python -m steering.run_hs_recovery_auc --model Qwen/Qwen3.5-2B-Base \\
        --device cuda --dtype float32 --mode inlp --intervene-layer 15 --rank 16 \\
        --subspaces steering/subspaces/inlp_gender_choice_v1.npz \\
        --read-layers 15,16,17,18,19,20,21,22,23,24 --tag hs_rec_l15k16

    # Probe center on L15 (needs h1_vectors_v1.npz)
    python -m steering.run_hs_recovery_auc --model Qwen/Qwen3.5-2B-Base \\
        --device cuda --dtype float32 --mode probe --intervene-layer 15 \\
        --vector-id w_gender_perp --alpha 1 --tag hs_rec_probe_l15

    # Smoke: 2 families, few layers
    python -m steering.run_hs_recovery_auc --model Qwen/Qwen3.5-2B-Base \\
        --device cuda --dtype float32 --mode probe --intervene-layer 15 \\
        --n-items 2 --read-layers 15,16,18 --tag hs_rec_smoke
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from steering.intervene import (
    InterventionSpec,
    Scorer,
    SubspaceSpec,
    SubspaceTrace,
    capture_last_token,
    load_model,
    steered,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STEERING_DIR = Path(__file__).resolve().parent

DEFAULT_SAMPLE = STEERING_DIR / "samples" / "h1_stagea_sample_v1.json"
DEFAULT_SUBSPACES = STEERING_DIR / "subspaces" / "inlp_gender_choice_v1.npz"
DEFAULT_VECTORS = (
    STEERING_DIR / "vectors" / "h1_vectors_v1.npz",
    STEERING_DIR / "vectors" / "slot_vectors_v1.npz",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def pick_items(items: list[dict], n: int | None) -> list[dict]:
    if n is None or n >= len(items):
        return list(items)
    by_soc: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_soc[it["soc_major_title"]].append(it)
    picked: list[dict] = []
    seen: set[int] = set()
    for soc in sorted(by_soc):
        fam = by_soc[soc][0]
        picked.append(fam)
        seen.add(id(fam))
        if len(picked) >= n:
            return picked[:n]
    for it in items:
        if id(it) in seen:
            continue
        picked.append(it)
        if len(picked) >= n:
            break
    return picked[:n]


def gender_choice_y(row: dict) -> int:
    """Frozen gender_choice from sample baseline_choice (layout semantics)."""
    choice = row.get("baseline_choice")
    if choice is None:
        raise SystemExit(f"row {row.get('id')}: нет baseline_choice — нужен sample со скором baseline")
    labels = row["labels"]
    if choice not in labels:
        raise SystemExit(f"row {row.get('id')}: baseline_choice={choice!r} нет в labels {labels}")
    return 1 if labels[choice] == "man" else 0


def slot_choice_y(row: dict) -> int:
    choice = row.get("baseline_choice")
    return 1 if choice == "A" else 0


def gender_choice_y_from_live(labels: dict[str, str], choice: str) -> int:
    if choice not in labels:
        raise SystemExit(f"live choice={choice!r} нет в labels {labels}")
    return 1 if labels[choice] == "man" else 0


def slot_choice_y_from_live(choice: str) -> int:
    return 1 if choice == "A" else 0


def gender_choice_y_from_live(labels: dict[str, str], choice: str) -> int:
    if choice not in labels:
        raise SystemExit(f"live choice={choice!r} нет в labels {labels}")
    return 1 if labels[choice] == "man" else 0


def slot_choice_y_from_live(choice: str) -> int:
    return 1 if choice == "A" else 0


def load_vector_bank(paths: list[Path]) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    bank: dict[str, np.ndarray] = {}
    calib: dict[str, dict] = {}
    for path in paths:
        if not path.is_file():
            continue
        meta_path = path.with_suffix(".json")
        with np.load(path) as z:
            for k in z.files:
                bank[k] = np.asarray(z[k], dtype=np.float32)
        if meta_path.is_file():
            for e in load_json(meta_path).get("vectors", []):
                calib[e["key"]] = e
    return bank, calib


def vector_key(vector_id: str, layer: int) -> str:
    return f"{vector_id}__L{layer}"


def roc_auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Mann–Whitney U / ROC-AUC. NaN if one class missing."""
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.int32)
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # P(score_pos > score_neg) + 0.5 P(equal)
    # efficient via ranks
    order = np.argsort(scores)
    ranks = np.empty_like(scores, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # average ranks for ties
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        if j > i:
            avg = 0.5 * (ranks[order[i]] + ranks[order[j]])
            ranks[order[i : j + 1]] = avg
        i = j + 1
    sum_ranks_pos = float(ranks[y == 1].sum())
    n_pos, n_neg = len(pos), len(neg)
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def mean_diff_direction(H: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = y.astype(int)
    pos = H[y == 1]
    neg = H[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("mean_diff: оба класса нужны в train")
    w = pos.mean(axis=0) - neg.mean(axis=0)
    n = float(np.linalg.norm(w))
    if n < 1e-12:
        return w
    return (w / n).astype(np.float64)


def family_split(
    family_ids: list[int],
    *,
    seed: int,
    train_frac: float,
) -> tuple[set[int], set[int]]:
    uniq = sorted(set(family_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_train = max(1, int(round(len(uniq) * train_frac)))
    n_train = min(n_train, len(uniq) - 1) if len(uniq) > 1 else 1
    train = set(uniq[:n_train])
    test = set(uniq[n_train:]) if len(uniq) > 1 else set(uniq)
    return train, test


def build_probe_spec(
    bank: dict[str, np.ndarray],
    calib: dict[str, dict],
    *,
    vector_id: str,
    layer: int,
    kind: str,
    alpha: float,
) -> InterventionSpec:
    key = vector_key(vector_id, layer)
    if key not in bank:
        raise SystemExit(
            f"нет вектора {key} в bank (есть {sum(1 for k in bank if k.startswith(vector_id))} ключей {vector_id}). "
            f"Соберите npz: python -m steering.build_h1_vectors"
        )
    meta = calib.get(key, {})
    return InterventionSpec(
        layer=layer,
        w=np.ascontiguousarray(bank[key], dtype=np.float32),
        kind=kind,
        alpha=alpha if kind == "center" else None,
        c=float(meta.get("c", 0.0)),
        sigma=float(meta.get("sigma_train", 1.0)),
    )


def build_inlp_spec(
    arrays: dict[str, np.ndarray],
    *,
    layer: int,
    rank: int,
    alpha: float,
    kind: str,
) -> SubspaceSpec:
    W_key = f"L{layer}__W"
    c_key = f"L{layer}__centers"
    if W_key not in arrays or c_key not in arrays:
        raise SystemExit(f"в subspaces нет {W_key}/{c_key}; ключи: {sorted(arrays)[:12]}…")
    W = arrays[W_key]
    c = arrays[c_key]
    if W.shape[1] < rank:
        raise SystemExit(f"rank {rank} > доступных {W.shape[1]} в {W_key}")
    return SubspaceSpec(
        layer=layer,
        W=np.ascontiguousarray(W[:, :rank], dtype=np.float32),
        c=np.ascontiguousarray(c[:rank], dtype=np.float32),
        kind=kind,
        alpha=alpha,
        label=f"inlp__L{layer}__k{rank}__a{alpha:g}",
    )


def row_margins(out: dict, labels: dict[str, str]) -> dict[str, float]:
    man = next(s for s, sem in labels.items() if sem == "man")
    woman = next(s for s, sem in labels.items() if sem == "woman")
    p_man = float(out.get(f"prob_constrained_{man}", float("nan")))
    p_woman = float(out.get(f"prob_constrained_{woman}", float("nan")))
    p_A = float(out.get("prob_constrained_A", float("nan")))
    p_B = float(out.get("prob_constrained_B", float("nan")))
    denom_g = p_man + p_woman
    denom_s = p_A + p_B
    return {
        "d_gender": float(out[f"logit_{man}"]) - float(out[f"logit_{woman}"]),
        "d_slot": float(out["logit_A"]) - float(out["logit_B"]),
        "p_man_norm": p_man / denom_g if denom_g > 0 else float("nan"),
        "p_A_norm": p_A / denom_s if denom_s > 0 else float("nan"),
        "choice": out["choice"],
    }


@torch.no_grad()
def capture_condition(
    model,
    scorer: Scorer,
    items: list[dict],
    specs: list[InterventionSpec | SubspaceSpec] | None,
    read_layers: list[int],
    *,
    log_every: int,
) -> tuple[list[dict], dict[int, list[np.ndarray]]]:
    """Returns meta rows + HS lists keyed by layer (same order as rows)."""
    hs: dict[int, list[np.ndarray]] = {L: [] for L in read_layers}
    rows: list[dict] = []
    total = sum(len(i["rows"]) for i in items)
    done = 0
    t0 = time.time()
    trace = SubspaceTrace() if specs and isinstance(specs[0], SubspaceSpec) else None

    for item in items:
        for src in item["rows"]:
            ctx = steered(model, specs, trace) if specs else contextlib.nullcontext()
            with ctx:
                with capture_last_token(model, read_layers) as store:
                    out = scorer.score(src["prompt"], src["valid_labels"])
                    for L in read_layers:
                        h = store[L][0, -1].float().detach().cpu().numpy().astype(np.float32)
                        hs[L].append(h)
            margins = row_margins(out, src["labels"])
            # Frozen H1 samples carry baseline_choice; polarity-pool samples from
            # xy_pairs do not — use this forward's choice (live baseline).
            if src.get("baseline_choice") is not None:
                y_g = gender_choice_y(src)
                y_s = slot_choice_y(src)
            else:
                y_g = gender_choice_y_from_live(src["labels"], margins["choice"])
                y_s = slot_choice_y_from_live(margins["choice"])
            row = {
                "id": src["id"],
                "scenario_family_id": item["scenario_family_id"],
                "soc_major_title": item["soc_major_title"],
                "y_gender": y_g,
                "y_slot": y_s,
                **margins,
            }
            if trace is not None and specs:
                L0 = specs[0].layer
                if L0 in trace.before:
                    row["hook_s_before"] = trace.before[L0].tolist()
                    row["hook_s_after"] = trace.after[L0].tolist()
                    row["hook_delta_norm"] = trace.delta_norm.get(L0)
                    row["hook_h_norm"] = trace.h_norm.get(L0)
            rows.append(row)
            done += 1
            if log_every and done % log_every == 0:
                rate = done / max(time.time() - t0, 1e-6)
                eta = (total - done) / max(rate, 1e-9) / 60
                print(f"    {done}/{total}  {rate:.2f} row/s  ETA {eta:.1f} min", flush=True)
    return rows, hs


def stack_hs(hs_lists: dict[int, list[np.ndarray]]) -> dict[int, np.ndarray]:
    return {L: np.stack(vs, axis=0) for L, vs in hs_lists.items()}


def summarize_layer(
    *,
    layer: int,
    H_base: np.ndarray,
    H_steer: np.ndarray,
    y: np.ndarray,
    families: np.ndarray,
    train_fams: set[int],
    test_fams: set[int],
    bank: dict[str, np.ndarray],
    calib: dict[str, dict],
    vector_id: str,
) -> dict:
    train = np.array([fid in train_fams for fid in families])
    test = np.array([fid in test_fams for fid in families])
    out: dict = {
        "layer": layer,
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "n_pos_test": int(y[test].sum()),
        "n_neg_test": int((1 - y[test]).sum()),
    }

    key = vector_key(vector_id, layer)
    if key in bank:
        w = bank[key].astype(np.float64)
        c = float(calib.get(key, {}).get("c", 0.0))
        s_b = H_base @ w
        s_s = H_steer @ w
        out["frozen"] = {
            "vector_key": key,
            "c": c,
            "auc_baseline_test": roc_auc(s_b[test], y[test]),
            "auc_steered_test": roc_auc(s_s[test], y[test]),
            "auc_baseline_all": roc_auc(s_b, y),
            "auc_steered_all": roc_auc(s_s, y),
            "mean_abs_s_c_baseline": float(np.mean(np.abs(s_b - c))),
            "mean_abs_s_c_steered": float(np.mean(np.abs(s_s - c))),
            "mean_abs_delta_s": float(np.mean(np.abs(s_s - s_b))),
        }
    else:
        out["frozen"] = None

    w_md: np.ndarray | None = None
    try:
        w_md = mean_diff_direction(H_base[train], y[train])
        s_b = H_base @ w_md
        s_s = H_steer @ w_md
        cos_fr = None
        if key in bank:
            wf = bank[key].astype(np.float64)
            cos_fr = float(w_md @ wf / (np.linalg.norm(wf) + 1e-12))
        out["mean_diff"] = {
            "auc_baseline_test": roc_auc(s_b[test], y[test]),
            "auc_steered_test": roc_auc(s_s[test], y[test]),
            "auc_baseline_train": roc_auc(s_b[train], y[train]),
            "mean_abs_score_baseline_test": float(np.mean(np.abs(s_b[test]))),
            "mean_abs_score_steered_test": float(np.mean(np.abs(s_s[test]))),
            "cos_with_frozen": cos_fr,
        }
    except ValueError as e:
        out["mean_diff"] = {"error": str(e)}

    try:
        w_refit = mean_diff_direction(H_steer[train], y[train])
        block = {
            "auc_steered_test": roc_auc(H_steer[test] @ w_refit, y[test]),
            "cos_with_baseline_md": None,
        }
        if w_md is not None:
            block["cos_with_baseline_md"] = float(w_refit @ w_md)
        out["mean_diff_refit_steered"] = block
    except ValueError as e:
        out["mean_diff_refit_steered"] = {"error": str(e)}

    return out


def preference_delta(rows_base: list[dict], rows_steer: list[dict]) -> dict:
    d_b = np.array([r["d_gender"] for r in rows_base], dtype=np.float64)
    d_s = np.array([r["d_gender"] for r in rows_steer], dtype=np.float64)
    # family-level |D| reduction needs family grouping — light summary only
    return {
        "mean_abs_d_gender_baseline": float(np.mean(np.abs(d_b))),
        "mean_abs_d_gender_steered": float(np.mean(np.abs(d_s))),
        "mean_abs_d_gender_reduction": float(np.mean(np.abs(d_b)) - np.mean(np.abs(d_s))),
        "mean_abs_theta_dev_baseline": float(
            np.mean(np.abs(np.array([r["p_man_norm"] for r in rows_base]) - 0.5))
        ),
        "mean_abs_theta_dev_steered": float(
            np.mean(np.abs(np.array([r["p_man_norm"] for r in rows_steer]) - 0.5))
        ),
    }


def write_by_layer_csv(path: Path, layers_out: list[dict]) -> None:
    headers = [
        "layer",
        "n_test",
        "frozen_auc_base",
        "frozen_auc_steer",
        "frozen_mean_abs_sc_base",
        "frozen_mean_abs_sc_steer",
        "md_auc_base_test",
        "md_auc_steer_test",
        "md_refit_auc_steer_test",
        "delta_auc_md",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for row in layers_out:
            fr = row.get("frozen") or {}
            md = row.get("mean_diff") or {}
            rf = row.get("mean_diff_refit_steered") or {}
            auc_b = md.get("auc_baseline_test")
            auc_s = md.get("auc_steered_test")
            w.writerow(
                {
                    "layer": row["layer"],
                    "n_test": row["n_test"],
                    "frozen_auc_base": fr.get("auc_baseline_test", ""),
                    "frozen_auc_steer": fr.get("auc_steered_test", ""),
                    "frozen_mean_abs_sc_base": fr.get("mean_abs_s_c_baseline", ""),
                    "frozen_mean_abs_sc_steer": fr.get("mean_abs_s_c_steered", ""),
                    "md_auc_base_test": auc_b if auc_b is not None else "",
                    "md_auc_steer_test": auc_s if auc_s is not None else "",
                    "md_refit_auc_steer_test": rf.get("auc_steered_test", ""),
                    "delta_auc_md": (
                        ""
                        if auc_b is None or auc_s is None or not np.isfinite(auc_b) or not np.isfinite(auc_s)
                        else float(auc_s) - float(auc_b)
                    ),
                }
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    ap.add_argument("--n-items", type=int, default=None, help="семей (default = все в sample)")
    ap.add_argument("--mode", choices=["probe", "inlp"], default="inlp")
    ap.add_argument("--intervene-layer", type=int, default=15)
    ap.add_argument("--read-layers", default="15,16,17,18,19,20,21,22,23,24")
    ap.add_argument("--vector-id", default="w_gender_perp", help="для --mode probe и frozen AUC")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--kind", default="center", choices=["center", "project_out"])
    ap.add_argument("--rank", type=int, default=16, help="INLP rank-k")
    ap.add_argument("--subspaces", type=Path, default=DEFAULT_SUBSPACES)
    ap.add_argument("--vectors", type=Path, nargs="*", default=None, help="npz bank paths")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "results" / "steering")
    ap.add_argument("--tag", default="hs_recovery_v1")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-hs", action="store_true", help="сохранить HS npz (тяжело)")
    args = ap.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    read_layers = parse_int_list(args.read_layers)
    if args.intervene_layer not in read_layers:
        read_layers = sorted(set(read_layers) | {args.intervene_layer})

    sample = load_json(args.sample)
    items = pick_items(sample["items"], args.n_items)
    n_rows = sum(len(i["rows"]) for i in items)

    vec_paths = list(args.vectors) if args.vectors else list(DEFAULT_VECTORS)
    bank, calib = load_vector_bank(vec_paths)

    specs: list[InterventionSpec | SubspaceSpec]
    intervene_meta: dict
    if args.mode == "probe":
        if not bank:
            raise SystemExit(
                "пустой vector bank — нужны "
                + ", ".join(str(p) for p in vec_paths)
                + " (python -m steering.build_h1_vectors)"
            )
        spec = build_probe_spec(
            bank,
            calib,
            vector_id=args.vector_id,
            layer=args.intervene_layer,
            kind=args.kind,
            alpha=args.alpha,
        )
        specs = [spec]
        intervene_meta = {
            "mode": "probe",
            "vector_id": args.vector_id,
            "layer": args.intervene_layer,
            "kind": args.kind,
            "alpha": args.alpha,
        }
    else:
        if not args.subspaces.is_file():
            raise SystemExit(
                f"нет subspaces {args.subspaces} — положите inlp_gender_choice_v1.npz "
                f"в steering/subspaces/ (sha из Stage C: 3545e531…)"
            )
        arrays = {k: np.asarray(v) for k, v in np.load(args.subspaces).items()}
        spec = build_inlp_spec(
            arrays,
            layer=args.intervene_layer,
            rank=args.rank,
            alpha=args.alpha,
            kind=args.kind,
        )
        specs = [spec]
        intervene_meta = {
            "mode": "inlp",
            "subspaces": str(args.subspaces),
            "layer": args.intervene_layer,
            "rank": args.rank,
            "kind": args.kind,
            "alpha": args.alpha,
            "label": spec.label,
        }

    print(
        f"hs_recovery [{args.tag}]: {len(items)} families / {n_rows} rows · "
        f"intervene {intervene_meta} · read {read_layers}"
    )
    print(f"  vector bank keys: {len(bank)}")

    print(f"\n[1] Загрузка {args.model}...")
    t_load = time.time()
    model, tokenizer = load_model(args.model, dtype=args.dtype, device=args.device)
    for p in model.parameters():
        p.requires_grad_(False)
    scorer = Scorer(model, tokenizer)
    print(f"  готово за {time.time() - t_load:.1f}s")

    out_dir = args.out_root / "hs_recovery" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[2] Baseline capture...")
    t0 = time.time()
    rows_base, hs_base_lists = capture_condition(
        model, scorer, items, None, read_layers, log_every=args.log_every
    )
    print(f"  baseline {time.time() - t0:.1f}s")

    print("\n[3] Steered capture...")
    t1 = time.time()
    rows_steer, hs_steer_lists = capture_condition(
        model, scorer, items, specs, read_layers, log_every=args.log_every
    )
    print(f"  steered {time.time() - t1:.1f}s")

    H_base = stack_hs(hs_base_lists)
    H_steer = stack_hs(hs_steer_lists)
    y = np.array([r["y_gender"] for r in rows_base], dtype=np.int32)
    families = np.array([r["scenario_family_id"] for r in rows_base], dtype=np.int64)
    train_fams, test_fams = family_split(
        families.tolist(), seed=args.split_seed, train_frac=args.train_frac
    )

    layers_out = []
    for L in read_layers:
        layers_out.append(
            summarize_layer(
                layer=L,
                H_base=H_base[L],
                H_steer=H_steer[L],
                y=y,
                families=families,
                train_fams=train_fams,
                test_fams=test_fams,
                bank=bank,
                calib=calib,
                vector_id=args.vector_id,
            )
        )

    pref = preference_delta(rows_base, rows_steer)
    summary = {
        "schema": "steering.hs_recovery_auc/v1",
        "tag": args.tag,
        "datetime": datetime.now().isoformat(),
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "sample": str(args.sample),
        "n_families": len(items),
        "n_rows": n_rows,
        "intervene": intervene_meta,
        "read_layers": read_layers,
        "label": {
            "id": "gender_choice",
            "definition": "1 iff labels[baseline_choice]=='man' (frozen in sample)",
        },
        "split": {
            "seed": args.split_seed,
            "train_frac": args.train_frac,
            "n_train_families": len(train_fams),
            "n_test_families": len(test_fams),
        },
        "preference_rowlevel": pref,
        "layers": layers_out,
        "read": (
            "Look at mean_diff.auc_*_test (and frozen if present). "
            "Recovery ≈ auc_steered rises again on layers > intervene while "
            "auc at intervene layer is suppressed vs baseline. "
            "mean_diff_refit_steered high while frozen/md-steered low ⇒ "
            "signal moved to a new linear direction (not the baseline axis)."
        ),
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_by_layer_csv(out_dir / "by_layer.csv", layers_out)

    # compact per-row preference only (not full HS)
    with (out_dir / "per_row_pref.jsonl").open("w", encoding="utf-8") as f:
        for rb, rs in zip(rows_base, rows_steer, strict=True):
            f.write(
                json.dumps(
                    {
                        "id": rb["id"],
                        "scenario_family_id": rb["scenario_family_id"],
                        "y_gender": rb["y_gender"],
                        "baseline": {
                            "d_gender": rb["d_gender"],
                            "p_man_norm": rb["p_man_norm"],
                            "choice": rb["choice"],
                        },
                        "steered": {
                            "d_gender": rs["d_gender"],
                            "p_man_norm": rs["p_man_norm"],
                            "choice": rs["choice"],
                            "hook_delta_norm": rs.get("hook_delta_norm"),
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    if args.save_hs:
        np.savez_compressed(
            out_dir / "hs_baseline.npz",
            **{f"L{L}": H_base[L] for L in read_layers},
            y=y,
            family=families,
        )
        np.savez_compressed(
            out_dir / "hs_steered.npz",
            **{f"L{L}": H_steer[L] for L in read_layers},
            y=y,
            family=families,
        )

    # console table
    print("\n=== by layer (mean_diff AUC test · baseline → steered) ===")
    print(f"{'L':>4}  {'AUC_base':>9}  {'AUC_steer':>9}  {'Δ':>8}  {'frozen|s-c| b→s':>22}")
    for row in layers_out:
        md = row.get("mean_diff") or {}
        fr = row.get("frozen") or {}
        ab = md.get("auc_baseline_test", float("nan"))
        as_ = md.get("auc_steered_test", float("nan"))
        dlt = as_ - ab if np.isfinite(ab) and np.isfinite(as_) else float("nan")
        if fr:
            sc = f"{fr['mean_abs_s_c_baseline']:.4f}→{fr['mean_abs_s_c_steered']:.4f}"
        else:
            sc = "—"
        print(f"{row['layer']:4d}  {ab:9.4f}  {as_:9.4f}  {dlt:8.4f}  {sc:>22}")

    print(f"\npreference |d_gender| mean: {pref['mean_abs_d_gender_baseline']:.4f} → "
          f"{pref['mean_abs_d_gender_steered']:.4f}  "
          f"(Δ={pref['mean_abs_d_gender_reduction']:.4f})")
    print(f"wrote {out_dir / 'summary.json'}")
    print(f"wrote {out_dir / 'by_layer.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
