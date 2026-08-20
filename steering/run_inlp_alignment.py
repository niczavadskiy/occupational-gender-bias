"""
Phase A.5: выравнивание INLP-направлений с выходным градиентом.

Дёшево и предсказывает исход Phase B до GPU-сетки. Для каждого слоя считается
g = ∇_h d, где d ∈ {d_gender = z_man − z_woman, d_slot = z_A − z_B}, и:

    cos(w_j, g)                            по каждому направлению INLP
    rho_k = ||W_kᵀ g||² / ||g||²           доля квадрата нормы градиента,
                                           захваченная rank-k подпространством

Референс — то же самое для random-подпространства из Phase A: у случайного
rank-k ожидается rho_k ≈ k/d (для d=2048 это 0.0005·k). Если INLP-подпространство
даёт rho_k того же порядка, оно ортогонально каузальному направлению и Phase B
почти обязан дать Case A/D; заметно большее rho_k — предпосылка Case B.

Требует градиентов, поэтому GPU и float32:

    python -m steering.run_inlp_alignment --model Qwen/Qwen3.5-2B-Base \\
        --device cuda --dtype float32 --tag inlp_align_v1

Smoke (4 строки):

    python -m steering.run_inlp_alignment --n-items 1 --tag inlp_align_smoke

Выход: results/steering/inlp_alignment/<tag>/{summary.json,directions.csv,per_row.jsonl}
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from steering.run_alignment_recovery import baseline_with_grads, cosine, load_json, pick_items

REPO_ROOT = Path(__file__).resolve().parents[1]
STEERING_DIR = Path(__file__).resolve().parent
DEFAULT_SUBSPACES = STEERING_DIR / "subspaces" / "inlp_gender_choice_v1.npz"
MARGINS = ("d_gender", "d_slot")


def subspace_rho(g: np.ndarray, W: np.ndarray) -> np.ndarray:
    """rho_k для k=1..K: кумулятивная доля ||g||², лежащая в span(w_1..w_k)."""
    ng2 = float(g @ g)
    if ng2 < 1e-24:
        return np.full(W.shape[1], np.nan)
    proj = W.T @ g
    return np.cumsum(proj * proj) / ng2


def median(xs: list[float]) -> float:
    arr = np.array([x for x in xs if np.isfinite(x)], dtype=float)
    return float(np.median(arr)) if len(arr) else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--subspaces", type=Path, default=DEFAULT_SUBSPACES)
    ap.add_argument("--sample", type=Path, default=STEERING_DIR / "samples" / "h1_stagea_sample_v1.json")
    ap.add_argument("--n-items", type=int, default=25, help="семей из sample (×4 строки)")
    ap.add_argument("--layers", default=None, help="Слои через запятую (default: все из подпространств)")
    ap.add_argument("--random-seed-key", default="0", help="seed random-подпространства для референса")
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "results" / "steering")
    ap.add_argument("--tag", default="inlp_align_v1")
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    from steering.intervene import load_model

    sub_meta = load_json(args.subspaces.with_suffix(".json"))
    with np.load(args.subspaces) as z:
        arrays = {k: np.asarray(z[k], dtype=np.float64) for k in z.files}

    by_layer = {int(L["layer"]): L for L in sub_meta["layers_detail"]}
    layers = (
        [int(x) for x in args.layers.split(",") if x.strip()] if args.layers else sorted(by_layer)
    )
    bases: dict[int, dict[str, np.ndarray]] = {}
    for layer in layers:
        entry = {"inlp": arrays[f"L{layer}__W"]}
        rand_key = f"L{layer}__W_random_s{args.random_seed_key}"
        if rand_key in arrays:
            entry["random"] = arrays[rand_key]
        bases[layer] = entry

    sample = load_json(args.sample)
    items = pick_items(sample["items"], args.n_items)
    n_rows = sum(len(i["rows"]) for i in items)
    d_model = int(sub_meta["d_model"])

    print(
        f"INLP alignment [{args.tag}]: {len(items)} семей / {n_rows} строк, "
        f"слои {layers}, dtype={args.dtype}"
    )
    for layer in layers:
        ks = {name: W.shape[1] for name, W in bases[layer].items()}
        print(f"  L{layer}: {ks}, ожидание для random rho_k ≈ k/{d_model}")

    print(f"\n[1] Загрузка {args.model}...")
    t_load = time.time()
    model, tokenizer = load_model(args.model, dtype=args.dtype, device=args.device)
    for p in model.parameters():
        p.requires_grad_(False)
    tok_ids = {
        label.strip(): tokenizer(label, add_special_tokens=False).input_ids[0]
        for label in (" A", " B", " C")
    }
    device = next(model.parameters()).device
    print(f"  готово за {time.time() - t_load:.1f}s  tokens {tok_ids}")

    out_dir = args.out_root / "inlp_alignment" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # cos по направлению, rho по ранку — накапливаем по строкам.
    cos_acc: dict[tuple[int, str, str, int], list[float]] = {}
    rho_acc: dict[tuple[int, str, str, int], list[float]] = {}
    g_norms: dict[tuple[int, str], list[float]] = {}

    t0 = time.time()
    done = 0
    with (out_dir / "per_row.jsonl").open("w", encoding="utf-8") as fout:
        for item in items:
            for src in item["rows"]:
                margins, _h, g_gender, g_slot, _h_all = baseline_with_grads(
                    model,
                    tokenizer,
                    src["prompt"],
                    src["labels"],
                    tok_ids,
                    device,
                    list(layers),
                    list(layers),
                )
                grads = {"d_gender": g_gender, "d_slot": g_slot}
                row_block: dict = {}
                for layer in layers:
                    layer_block: dict = {}
                    for margin in MARGINS:
                        g = grads[margin][layer]
                        g_norms.setdefault((layer, margin), []).append(float(np.linalg.norm(g)))
                        for name, W in bases[layer].items():
                            cos_j = [cosine(g, W[:, j]) for j in range(W.shape[1])]
                            rho_k = subspace_rho(g, W)
                            for j, cv in enumerate(cos_j, start=1):
                                cos_acc.setdefault((layer, margin, name, j), []).append(cv)
                            for k, rv in enumerate(rho_k, start=1):
                                rho_acc.setdefault((layer, margin, name, k), []).append(float(rv))
                            layer_block[f"{name}::{margin}"] = {
                                "cos": cos_j,
                                "rho_cumulative": [float(x) for x in rho_k],
                            }
                    layer_block["||g_gender||"] = float(np.linalg.norm(grads["d_gender"][layer]))
                    layer_block["||g_slot||"] = float(np.linalg.norm(grads["d_slot"][layer]))
                    row_block[str(layer)] = layer_block

                fout.write(
                    json.dumps(
                        {
                            "id": src["id"],
                            "scenario_family_id": item["scenario_family_id"],
                            "position_variant": src["position_variant"],
                            "context_order": src["context_order"],
                            "baseline": margins,
                            "alignment": row_block,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                done += 1
                if args.log_every and done % args.log_every == 0:
                    rate = done / max(time.time() - t0, 1e-9)
                    print(
                        f"    {done}/{n_rows}  {rate:.2f} row/s  "
                        f"ETA {(n_rows - done) / max(rate, 1e-9) / 60:.1f} min",
                        flush=True,
                    )

    directions: list[dict] = []
    for (layer, margin, name, j), vals in sorted(cos_acc.items()):
        directions.append(
            {
                "layer": layer,
                "margin": margin,
                "basis": name,
                "index": j,
                "median_cos": median(vals),
                "median_abs_cos": median([abs(v) for v in vals]),
                "median_rho_cumulative": median(rho_acc[(layer, margin, name, j)]),
                "rho_random_expectation": j / d_model,
                "n": len(vals),
            }
        )

    summary = {
        "schema": "steering.inlp_alignment/v1",
        "tag": args.tag,
        "datetime": datetime.now().isoformat(),
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "subspaces_file": args.subspaces.name,
        "subspaces_sha256": sub_meta.get("arrays_sha256"),
        "target": sub_meta["target"],
        "n_items": len(items),
        "n_rows": n_rows,
        "layers": layers,
        "d_model": d_model,
        "gradient_norms": {
            f"L{layer}::{margin}": median(vals) for (layer, margin), vals in sorted(g_norms.items())
        },
        "directions": directions,
        "read": (
            "rho_k(inlp) ≈ rho_k(random) ≈ k/d → подпространство ортогонально каузальному "
            "направлению, ожидается Case A/D. rho_k(inlp) >> k/d → предпосылка Case B."
        ),
        "runtime_s": round(time.time() - t0, 1),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (out_dir / "directions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(directions[0].keys()))
        writer.writeheader()
        writer.writerows(directions)

    print(f"\n=== INLP alignment [{args.tag}] {summary['runtime_s']:.0f}s → {out_dir} ===")
    for layer in layers:
        print(f"\n  L{layer}  ||g_gender||={summary['gradient_norms'].get(f'L{layer}::d_gender', float('nan')):.4g}")
        for margin in MARGINS:
            for name in bases[layer]:
                rows = [
                    d
                    for d in directions
                    if d["layer"] == layer and d["margin"] == margin and d["basis"] == name
                ]
                if not rows:
                    continue
                head = "  ".join(f"w{d['index']}:{d['median_abs_cos']:.3f}" for d in rows[:8])
                last = rows[-1]
                print(
                    f"    {name:<7} {margin:<9} |cos| {head}"
                    f"   → rho_{last['index']}={last['median_rho_cumulative']:.4f} "
                    f"(random {last['rho_random_expectation']:.4f})"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
