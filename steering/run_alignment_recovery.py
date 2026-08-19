"""
Эксп.1–3: probe-normal vs output-gradient, first-order Δd, downstream recovery.

На 25 семей из h1_stagea_sample_v1.json (100 строк), слои L16 и L23, float32.

  1. alignment: cos(g, ŵ) и ρ = (g·ŵ)² / ||g||², где
     d_gender = z(man_slot) − z(woman_slot),  d_slot = z_A − z_B
  2. Δd: shift ±β=1 и center α=1 vs gᵀ Δh
  3. recovery: s_{l'} после интервенции на native-слое (gender@L16, slot@L23)

GPU:

    python -m steering.run_alignment_recovery --model Qwen/Qwen3.5-2B-Base \\
        --device cuda --dtype float32 --tag mech_v1

Smoke (4 строки):

    python -m steering.run_alignment_recovery --model Qwen/Qwen3.5-2B-Base \\
        --device cuda --dtype float32 --n-items 1 --tag mech_smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from steering.intervene import (
    InterventionSpec,
    ProjectionTrace,
    capture_last_token,
    load_model,
    steered,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STEERING_DIR = Path(__file__).resolve().parent

ALIGN_LAYERS = (16, 23)
READ_LAYERS = tuple(range(16, 25))
VECTOR_IDS = ("w_gender", "w_gender_perp", "w_slot", "w_slot_perp", "w_random")
PRIMARY = {"w_gender": "d_gender", "w_gender_perp": "d_gender", "w_slot": "d_slot", "w_slot_perp": "d_slot"}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pick_items(items: list[dict], n: int) -> list[dict]:
    """По одной семье на SOC, затем добор из исходного порядка до n."""
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


def load_vector_bank() -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """slot_vectors закрывает L16–24; h1_vectors перекрывает H1-ключи (perp, L14–20)."""
    bank: dict[str, np.ndarray] = {}
    calib: dict[str, dict] = {}
    for name in ("slot_vectors_v1.npz", "h1_vectors_v1.npz"):
        path = STEERING_DIR / "vectors" / name
        if not path.is_file():
            continue
        meta = load_json(path.with_suffix(".json"))
        with np.load(path) as z:
            for k in z.files:
                bank[k] = np.asarray(z[k], dtype=np.float32)
        for e in meta["vectors"]:
            calib[e["key"]] = e
    return bank, calib


def vector_key(vector_id: str, layer: int) -> str:
    if vector_id == "w_random":
        return f"w_random__L{layer}__s0"
    return f"{vector_id}__L{layer}"


def last_token(store: dict[int, torch.Tensor], layer: int) -> torch.Tensor:
    return store[layer][0, -1].float()


def slot_for(labels: dict[str, str], semantic: str) -> str:
    for slot, sem in labels.items():
        if sem == semantic:
            return slot
    raise KeyError(f"{semantic} not in {labels}")


def margins_from_logits(
    last_logits: torch.Tensor,
    tok_ids: dict[str, int],
    labels: dict[str, str],
) -> dict[str, float | torch.Tensor]:
    z_a = last_logits[tok_ids["A"]]
    z_b = last_logits[tok_ids["B"]]
    man_slot = slot_for(labels, "man")
    woman_slot = slot_for(labels, "woman")
    z_man = last_logits[tok_ids[man_slot]]
    z_woman = last_logits[tok_ids[woman_slot]]
    d_gender = z_man - z_woman
    d_slot = z_a - z_b
    pair = torch.stack([z_a, z_b])
    probs = torch.softmax(pair, dim=0)
    p_a, p_b = probs[0], probs[1]
    p_man = p_a if man_slot == "A" else p_b
    p_woman = 1.0 - p_man
    d_logp = torch.log(p_man.clamp_min(1e-12)) - torch.log(p_woman.clamp_min(1e-12))
    return {
        "logit_A": z_a,
        "logit_B": z_b,
        "d_gender": d_gender,
        "d_slot": d_slot,
        "p_man": p_man,
        "p_woman": p_woman,
        "d_logp_gender": d_logp,
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(a @ b / (na * nb))


def rho(g: np.ndarray, w: np.ndarray) -> float:
    ng = float(np.linalg.norm(g))
    if ng < 1e-12:
        return float("nan")
    return float((g @ w) ** 2 / (ng * ng))


def f32(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().float().cpu())
    return float(x)


def spec_from(
    bank: dict[str, np.ndarray],
    calib: dict[str, dict],
    vector_id: str,
    layer: int,
    kind: str,
    *,
    alpha: float | None = None,
    beta: float | None = None,
) -> InterventionSpec | None:
    key = vector_key(vector_id, layer)
    if key not in bank:
        return None
    meta = calib[key]
    return InterventionSpec(
        layer=layer,
        w=bank[key],
        kind=kind,
        alpha=alpha,
        beta=beta,
        c=float(meta["c"]) if kind == "center" else 0.0,
        sigma=float(meta["sigma_train"]),
    )


def planned_interventions(
    bank: dict[str, np.ndarray],
    calib: dict[str, dict],
    layers: tuple[int, ...],
) -> list[tuple[str, str, InterventionSpec]]:
    """(vector_id, label, spec). label уникален в плане."""
    out: list[tuple[str, str, InterventionSpec]] = []
    for layer in layers:
        vecs = ["w_gender", "w_gender_perp", "w_slot", "w_slot_perp"]
        if layer == 16:
            vecs.append("w_random")
        for vid in vecs:
            for kind, kw, tag in (
                ("shift", {"beta": 1.0}, "shift_bp1"),
                ("shift", {"beta": -1.0}, "shift_bm1"),
                ("center", {"alpha": 1.0}, "center_a1"),
            ):
                spec = spec_from(bank, calib, vid, layer, kind, **kw)
                if spec is None:
                    continue
                out.append((vid, f"{vid}__L{layer}__{tag}", spec))
    return out


def recovery_layers(spec: InterventionSpec, vid: str) -> list[int] | None:
    if spec.kind == "center" and spec.alpha == 1.0 and vid == "w_gender" and spec.layer == 16:
        return list(READ_LAYERS)
    if spec.kind == "shift" and spec.beta == 1.0 and vid == "w_gender" and spec.layer == 16:
        return list(READ_LAYERS)
    if spec.kind == "center" and spec.alpha == 1.0 and vid == "w_slot" and spec.layer == 23:
        return [23, 24]
    if spec.kind == "shift" and spec.beta == 1.0 and vid == "w_slot" and spec.layer == 23:
        return [23, 24]
    return None


def projection_map(
    store: dict[int, torch.Tensor],
    bank: dict[str, np.ndarray],
    layers: list[int],
    vector_ids: tuple[str, ...] = ("w_gender", "w_slot"),
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {vid: {} for vid in vector_ids}
    for layer in layers:
        if layer not in store:
            continue
        h = last_token(store, layer).detach().cpu().numpy()
        for vid in vector_ids:
            key = vector_key(vid, layer)
            if key not in bank:
                continue
            out[vid][str(layer)] = float(h @ bank[key].astype(np.float64))
    return out


@torch.no_grad()
def forward_logits(model, tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**inputs)
    return out.logits[0, -1].float()


def baseline_with_grads(
    model,
    tokenizer,
    prompt: str,
    labels: dict[str, str],
    tok_ids: dict[str, int],
    device: torch.device,
    capture_layers: list[int],
    grad_layers: list[int],
) -> tuple[dict, dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """margins, h[grad], g_gender, g_slot, h_all[capture]."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.enable_grad():
        with capture_last_token(model, capture_layers, retain_grad_layers=grad_layers) as store:
            out = model(**inputs)
            last_logits = out.logits[0, -1].float()
            m = margins_from_logits(last_logits, tok_ids, labels)

            h = {L: last_token(store, L).detach().cpu().numpy().astype(np.float64) for L in grad_layers}

            m["d_gender"].backward(retain_graph=True)
            g_gender = {}
            for L in grad_layers:
                grad = store[L].grad
                if grad is None:
                    g_gender[L] = np.zeros_like(h[L])
                else:
                    g_gender[L] = grad[0, -1].float().detach().cpu().numpy().astype(np.float64)
                    store[L].grad.zero_()

            m["d_slot"].backward()
            g_slot = {}
            for L in grad_layers:
                grad = store[L].grad
                if grad is None:
                    g_slot[L] = np.zeros_like(h[L])
                else:
                    g_slot[L] = grad[0, -1].float().detach().cpu().numpy().astype(np.float64)

            captured_h_all = {
                L: last_token(store, L).detach().cpu().numpy().astype(np.float64)
                for L in capture_layers
                if L in store
            }

    margins = {k: f32(v) for k, v in m.items()}
    return margins, h, g_gender, g_slot, captured_h_all


def median(xs: list[float]) -> float:
    arr = np.array([x for x in xs if np.isfinite(x)], dtype=float)
    return float(np.median(arr)) if len(arr) else float("nan")


def mean_abs(xs: list[float]) -> float:
    arr = np.array([x for x in xs if np.isfinite(x)], dtype=float)
    return float(np.mean(np.abs(arr))) if len(arr) else float("nan")


def corr(a: list[float], b: list[float]) -> float:
    x = np.array(a, dtype=float)
    y = np.array(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    if float(np.std(x[mask])) < 1e-15 or float(np.std(y[mask])) < 1e-15:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def summarize(rows: list[dict], layers: tuple[int, ...]) -> dict:
    align: dict[str, dict] = {}
    for L in layers:
        block: dict[str, dict] = {}
        for vid in VECTOR_IDS:
            for margin in ("d_gender", "d_slot"):
                cos = [r["alignment"][str(L)][vid][margin]["cos"] for r in rows if vid in r["alignment"].get(str(L), {})]
                rh = [r["alignment"][str(L)][vid][margin]["rho"] for r in rows if vid in r["alignment"].get(str(L), {})]
                if not cos:
                    continue
                block[f"{vid}::{margin}"] = {
                    "median_cos": median(cos),
                    "median_abs_cos": median([abs(c) for c in cos]),
                    "mean_abs_cos": mean_abs(cos),
                    "median_rho": median(rh),
                    "n": len(cos),
                }
        gn = [r["alignment"][str(L)]["||g_gender||"] for r in rows]
        sn = [r["alignment"][str(L)]["||g_slot||"] for r in rows]
        block["||g_gender||_median"] = median(gn)
        block["||g_slot||_median"] = median(sn)
        align[str(L)] = block

    by_cfg: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        for iv in r["interventions"]:
            by_cfg[iv["id"]].append(iv)

    interventions = {}
    for cfg_id, ivs in sorted(by_cfg.items()):
        primary = PRIMARY.get(ivs[0]["vector_id"], "d_gender")
        lin = [x[f"delta_{primary}_lin"] for x in ivs]
        obs = [x[f"delta_{primary}_obs"] for x in ivs]
        interventions[cfg_id] = {
            "n": len(ivs),
            "primary_margin": primary,
            "mean_abs_delta_obs": mean_abs(obs),
            "mean_abs_delta_lin": mean_abs(lin),
            "corr_lin_obs": corr(lin, obs),
            "mean_abs_delta_p_man": mean_abs([x["delta_p_man"] for x in ivs]),
            "mean_abs_delta_s": mean_abs([x["delta_s"] for x in ivs]),
        }

    recovery = {}
    for r in rows:
        for rec_id, rec in r.get("recovery", {}).items():
            recovery.setdefault(rec_id, defaultdict(list))
            base = rec.get("baseline", {})
            steered_s = rec.get("steered", {})
            for vid, layers_s in steered_s.items():
                for L, s in layers_s.items():
                    b = base.get(vid, {}).get(L)
                    if b is None:
                        continue
                    recovery[rec_id][f"{vid}__L{L}"].append(abs(s - b))
    recovery_sum = {
        rid: {k: {"median_abs_delta_s": median(vs), "mean_abs_delta_s": mean_abs(vs)} for k, vs in d.items()}
        for rid, d in recovery.items()
    }

    return {
        "n_rows": len(rows),
        "alignment": align,
        "interventions": interventions,
        "recovery": recovery_sum,
        "read": (
            "H-B: median |cos(w,g)| ≈ 0 and mean |Δd_obs| ≈ 0. "
            "H-A: |cos| notable, Δd_lin≈Δd_obs locally, |Δs| decays toward later layers. "
            "H-C: |cos| moderate, ρ small, |Δd_lin| small even when ||g|| is large."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--sample", type=Path, default=STEERING_DIR / "samples" / "h1_stagea_sample_v1.json")
    ap.add_argument("--n-items", type=int, default=25, help="семей из sample (×4 строки)")
    ap.add_argument("--layers", default="16,23")
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "results" / "steering")
    ap.add_argument("--tag", default="mech_v1")
    ap.add_argument("--log-every", type=int, default=5)
    args = ap.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    layers = tuple(int(x) for x in args.layers.split(",") if x.strip())
    sample = load_json(args.sample)
    items = pick_items(sample["items"], args.n_items)
    n_rows = sum(len(i["rows"]) for i in items)
    bank, calib = load_vector_bank()
    plan = planned_interventions(bank, calib, layers)
    capture_layers = sorted(set(READ_LAYERS) | set(layers))

    print(
        f"mech [{args.tag}]: {len(items)} families / {n_rows} rows × "
        f"(1 baseline+grad + {len(plan)} interventions)  dtype={args.dtype}"
    )
    print(f"  vectors in bank: {len(bank)}; planned interventions: {len(plan)}")

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

    out_dir = args.out_root / "mech" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "per_row.jsonl"
    t0 = time.time()
    rows_out: list[dict] = []
    done = 0

    with jsonl_path.open("w", encoding="utf-8") as fout:
        for item in items:
            for src in item["rows"]:
                labels = src["labels"]
                margins, h, g_gender, g_slot, h_all = baseline_with_grads(
                    model,
                    tokenizer,
                    src["prompt"],
                    labels,
                    tok_ids,
                    device,
                    capture_layers,
                    list(layers),
                )
                s_base = {}
                for vid in ("w_gender", "w_slot"):
                    s_base[vid] = {}
                    for L, vec in h_all.items():
                        key = vector_key(vid, L)
                        if key in bank:
                            s_base[vid][str(L)] = float(vec @ bank[key].astype(np.float64))

                if done == 0 and all(float(np.linalg.norm(g_gender[L])) < 1e-18 for L in g_gender):
                    print("  WARNING: ∇h d_gender = 0 — retain_grad не сработал, alignment бесполезен")
                alignment = {}
                for L in layers:
                    block: dict = {
                        "||g_gender||": float(np.linalg.norm(g_gender[L])),
                        "||g_slot||": float(np.linalg.norm(g_slot[L])),
                    }
                    for vid in VECTOR_IDS:
                        key = vector_key(vid, L)
                        if key not in bank:
                            continue
                        w = bank[key].astype(np.float64)
                        block[vid] = {
                            "d_gender": {
                                "cos": cosine(g_gender[L], w),
                                "rho": rho(g_gender[L], w),
                            },
                            "d_slot": {
                                "cos": cosine(g_slot[L], w),
                                "rho": rho(g_slot[L], w),
                            },
                        }
                    alignment[str(L)] = block

                interventions = []
                recovery: dict[str, dict] = {}
                for vid, cfg_id, spec in plan:
                    rec_layers = recovery_layers(spec, vid)
                    trace = ProjectionTrace()
                    with steered(model, [spec], trace):
                        if rec_layers:
                            with capture_last_token(model, rec_layers) as rec_store:
                                last_logits = forward_logits(model, tokenizer, src["prompt"], device)
                                s_steer = projection_map(rec_store, bank, rec_layers)
                        else:
                            last_logits = forward_logits(model, tokenizer, src["prompt"], device)
                            s_steer = None
                    m2 = {k: f32(v) for k, v in margins_from_logits(last_logits, tok_ids, labels).items()}
                    s_b = trace.before.get(spec.layer)
                    s_a = trace.after.get(spec.layer)
                    if spec.kind == "shift":
                        delta_scale = float(spec.beta) * float(spec.sigma)
                    elif spec.kind == "center":
                        s_h = float(h[spec.layer] @ spec.w.astype(np.float64))
                        delta_scale = -float(spec.alpha) * (s_h - spec.c)
                    else:
                        delta_scale = 0.0
                    dh = delta_scale * spec.w.astype(np.float64)
                    rec = {
                        "id": cfg_id,
                        "vector_id": vid,
                        "layer": spec.layer,
                        "kind": spec.kind,
                        "alpha": spec.alpha,
                        "beta": spec.beta,
                        "s_before": s_b,
                        "s_after": s_a,
                        "delta_s": None if s_b is None or s_a is None else s_a - s_b,
                        "d_gender": m2["d_gender"],
                        "d_slot": m2["d_slot"],
                        "p_man": m2["p_man"],
                        "delta_d_gender_obs": m2["d_gender"] - margins["d_gender"],
                        "delta_d_slot_obs": m2["d_slot"] - margins["d_slot"],
                        "delta_p_man": m2["p_man"] - margins["p_man"],
                        "delta_d_gender_lin": float(g_gender[spec.layer] @ dh),
                        "delta_d_slot_lin": float(g_slot[spec.layer] @ dh),
                    }
                    interventions.append(rec)
                    if s_steer is not None:
                        recovery[cfg_id] = {"baseline": s_base, "steered": s_steer}

                row = {
                    "id": src["id"],
                    "scenario_family_id": item["scenario_family_id"],
                    "soc_major_title": item["soc_major_title"],
                    "position_variant": src["position_variant"],
                    "context_order": src["context_order"],
                    "labels": labels,
                    "baseline": margins,
                    "alignment": alignment,
                    "interventions": interventions,
                    "recovery": recovery,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows_out.append(row)
                done += 1
                if args.log_every and done % args.log_every == 0:
                    rate = done / max(time.time() - t0, 1e-6)
                    eta = (n_rows - done) / max(rate, 1e-9) / 60
                    print(f"    {done}/{n_rows}  {rate:.2f} row/s  ETA {eta:.1f} min", flush=True)

    summary = summarize(rows_out, layers)
    meta = {
        "schema": "steering.mech_alignment_recovery/v1",
        "tag": args.tag,
        "datetime": datetime.now().isoformat(),
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "n_items": len(items),
        "n_rows": n_rows,
        "layers": list(layers),
        "n_interventions_per_row": len(plan),
        "intervention_ids": [p[1] for p in plan],
        "runtime_s": round(time.time() - t0, 1),
        "hf_home": os.environ.get("HF_HOME"),
        "summary": summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # compact alignment table
    csv_path = out_dir / "alignment_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["layer", "vector", "margin", "median_cos", "median_abs_cos", "mean_abs_cos", "median_rho", "n"],
        )
        w.writeheader()
        for L, block in summary["alignment"].items():
            for key, st in block.items():
                if not isinstance(st, dict) or "median_cos" not in st:
                    continue
                vid, margin = key.split("::")
                w.writerow({"layer": L, "vector": vid, "margin": margin, **st})

    iv_csv = out_dir / "delta_summary.csv"
    with iv_csv.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "id",
            "n",
            "primary_margin",
            "mean_abs_delta_obs",
            "mean_abs_delta_lin",
            "corr_lin_obs",
            "mean_abs_delta_p_man",
            "mean_abs_delta_s",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cfg_id, st in summary["interventions"].items():
            w.writerow({"id": cfg_id, **st})

    print(f"\n=== mech [{args.tag}] {meta['runtime_s']:.0f}s → {out_dir} ===")
    print("alignment |cos| medians:")
    for L, block in summary["alignment"].items():
        print(f"  L{L}  ||g_gender||={block['||g_gender||_median']:.4g}  ||g_slot||={block['||g_slot||_median']:.4g}")
        for key, st in block.items():
            if not isinstance(st, dict) or "median_abs_cos" not in st:
                continue
            print(f"    {key:<28} |cos|={st['median_abs_cos']:.3f}  ρ={st['median_rho']:.3f}")
    print("Δd (primary):")
    for cfg_id, st in summary["interventions"].items():
        if "shift_bp1" not in cfg_id and "center_a1" not in cfg_id:
            continue
        print(
            f"    {cfg_id:<42} |Δd_obs|={st['mean_abs_delta_obs']:.4g}  "
            f"corr(lin,obs)={st['corr_lin_obs']:.3f}  |Δp_man|={st['mean_abs_delta_p_man']:.4g}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
