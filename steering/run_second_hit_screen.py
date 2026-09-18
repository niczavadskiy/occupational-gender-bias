"""
Sequential multi-site erase — Phase 1: second-hit screen under L15 gender INLP.

Fixed base stack S0 = INLP gender L15 k16 center α=1.
For each candidate second site (gender / slot probe or INLP on L16–24):
  - alone:   only second site
  - stacked: S0 + second site
Compare preference vs true baseline → R_gender, |θ−0.5|, R_slot.
Shortlist layers where stacked beats S0 alone (synergy on gender and/or slot).

Examples:

    python -m steering.run_second_hit_screen --model Qwen/Qwen3.5-2B-Base \\
        --device cuda --dtype float32 --tag second_hit_v1

    # smoke
    python -m steering.run_second_hit_screen --model Qwen/Qwen3.5-2B-Base \\
        --device cuda --n-items 2 --layers 16,20,23 --tag second_hit_smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from steering.intervene import (
    InterventionSpec,
    Scorer,
    SubspaceSpec,
    load_model,
    steered,
)
from steering.run_hs_recovery_auc import (
    build_inlp_spec,
    build_probe_spec,
    load_vector_bank,
    pick_items,
)
from steering.run_inlp_stagea import behavioral_metrics, load_json, row_margins

REPO_ROOT = Path(__file__).resolve().parents[1]
STEERING_DIR = Path(__file__).resolve().parent

DEFAULT_SAMPLE = STEERING_DIR / "samples" / "h1_stagea_sample_v1.json"
DEFAULT_SUBSPACES = STEERING_DIR / "subspaces" / "inlp_gender_choice_v1.npz"
DEFAULT_VECTORS = (
    STEERING_DIR / "vectors" / "h1_vectors_v1.npz",
    STEERING_DIR / "vectors" / "slot_vectors_v1.npz",
)


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def run_rows(scorer: Scorer, model, items: list[dict], specs: list, *, label: str, log_every: int) -> list[dict]:
    rows: list[dict] = []
    total = sum(len(i["rows"]) for i in items)
    t0 = time.time()
    for item in items:
        for src in item["rows"]:
            with steered(model, specs if specs else []):
                out = scorer.score(src["prompt"], list(src["valid_labels"]))
            labels = src["labels"]
            rows.append(
                {
                    "id": src["id"],
                    "scenario_family_id": item["scenario_family_id"],
                    "soc_major_title": item["soc_major_title"],
                    "profession": item.get("profession"),
                    "position_variant": src["position_variant"],
                    "context_order": src["context_order"],
                    "labels": labels,
                    "choice": out["choice"],
                    "logit_A": out["logit_A"],
                    "logit_B": out["logit_B"],
                    **row_margins(out, labels),
                    "baseline_choice_recorded": src.get("baseline_choice"),
                }
            )
            if log_every and len(rows) % log_every == 0:
                rate = len(rows) / max(time.time() - t0, 1e-9)
                eta = (total - len(rows)) / max(rate, 1e-9) / 60
                print(f"    [{label}] {len(rows)}/{total}  {rate:.2f} row/s  ETA {eta:.1f} min", flush=True)
    return rows


def try_probe(
    bank: dict,
    calib: dict,
    vector_id: str,
    layer: int,
    kind: str,
    alpha: float,
) -> InterventionSpec | None:
    key = f"{vector_id}__L{layer}"
    if key not in bank:
        return None
    try:
        return build_probe_spec(bank, calib, vector_id=vector_id, layer=layer, kind=kind, alpha=alpha)
    except SystemExit:
        return None


def try_inlp(arrays: dict, layer: int, rank: int, alpha: float) -> SubspaceSpec | None:
    if f"L{layer}__W" not in arrays:
        return None
    try:
        return build_inlp_spec(arrays, layer=layer, rank=rank, alpha=alpha, kind="center")
    except SystemExit:
        return None


def build_candidates(
    *,
    layers: list[int],
    bank: dict,
    calib: dict,
    inlp_arrays: dict,
    gender_vec: str,
    slot_vec: str,
    inlp_rank: int,
    alpha: float,
    include_inlp: bool,
    include_project_out: bool,
) -> list[dict]:
    """Each candidate: id, axis, layer, mode, alone_specs (without base)."""
    out: list[dict] = []
    gender_ids = []
    for v in (gender_vec, "w_gender_perp", "w_gender"):
        if v not in gender_ids:
            gender_ids.append(v)
    slot_ids = []
    for v in (slot_vec, "w_slot_perp", "w_slot"):
        if v not in slot_ids:
            slot_ids.append(v)

    for L in layers:
        added_gender_probe = False
        for gvid in gender_ids:
            if added_gender_probe:
                break
            sp = try_probe(bank, calib, gvid, L, "center", alpha)
            if sp is not None:
                out.append(
                    {
                        "id": f"gender_probe__{gvid}__L{L}__center__a{alpha:g}".replace(".", "p"),
                        "axis": "gender",
                        "layer": L,
                        "mode": "probe_center",
                        "vector_id": gvid,
                        "spec": sp,
                    }
                )
                added_gender_probe = True
                if include_project_out:
                    sp_po = try_probe(bank, calib, gvid, L, "project_out", alpha)
                    if sp_po is not None:
                        out.append(
                            {
                                "id": f"gender_probe__{gvid}__L{L}__project_out",
                                "axis": "gender",
                                "layer": L,
                                "mode": "probe_project_out",
                                "vector_id": gvid,
                                "spec": sp_po,
                            }
                        )

        if include_inlp:
            sp2 = try_inlp(inlp_arrays, L, inlp_rank, alpha)
            if sp2 is not None:
                out.append(
                    {
                        "id": f"gender_inlp__L{L}__k{inlp_rank}__a{alpha:g}".replace(".", "p"),
                        "axis": "gender",
                        "layer": L,
                        "mode": "inlp_center",
                        "rank": inlp_rank,
                        "spec": sp2,
                    }
                )

        added_slot = False
        for svid in slot_ids:
            if added_slot:
                break
            sp = try_probe(bank, calib, svid, L, "center", alpha)
            if sp is not None:
                out.append(
                    {
                        "id": f"slot_probe__{svid}__L{L}__center__a{alpha:g}".replace(".", "p"),
                        "axis": "slot",
                        "layer": L,
                        "mode": "probe_center",
                        "vector_id": svid,
                        "spec": sp,
                    }
                )
                added_slot = True
                if include_project_out:
                    sp_po = try_probe(bank, calib, svid, L, "project_out", alpha)
                    if sp_po is not None:
                        out.append(
                            {
                                "id": f"slot_probe__{svid}__L{L}__project_out",
                                "axis": "slot",
                                "layer": L,
                                "mode": "probe_project_out",
                                "vector_id": svid,
                                "spec": sp_po,
                            }
                        )

        if L in (16, 20, 23):
            key0 = f"w_random__L{L}__s0"
            if key0 in bank:
                meta = calib.get(key0, {})
                sp = InterventionSpec(
                    layer=L,
                    w=np.ascontiguousarray(bank[key0], dtype=np.float32),
                    kind="project_out",
                    c=float(meta.get("c", 0.0)),
                    sigma=float(meta.get("sigma_train", 1.0)),
                )
                out.append(
                    {
                        "id": f"random__L{L}__project_out__s0",
                        "axis": "random",
                        "layer": L,
                        "mode": "random_project_out",
                        "spec": sp,
                    }
                )
    return out


def metric_row(tag: str, meta: dict, metrics: dict) -> dict:
    g = metrics["gender_axis"]["reduction"]
    s = metrics["slot_axis"]["reduction"]
    th = metrics["theta_prob_axis"]["reduction"]
    return {
        "id": tag,
        **meta,
        "R_gender": g["mean"],
        "R_gender_ci_lo": g["ci_lo"],
        "R_gender_ci_hi": g["ci_hi"],
        "frac_R_gender_pos": g["fraction_positive"],
        "R_slot": s["mean"],
        "abs_theta_before": metrics["theta_prob_axis"]["mean_abs_before"],
        "abs_theta_after": metrics["theta_prob_axis"]["mean_abs_after"],
        "R_theta": th["mean"],
        "flip_rate": metrics["flip_rate"],
        "|D|_before": metrics["gender_axis"]["mean_abs_before"],
        "|D|_after": metrics["gender_axis"]["mean_abs_after"],
    }


def _write_ranking_csv(path: Path, ranking: list[dict]) -> None:
    if not ranking:
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in ranking:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        for r in ranking:
            w.writerow(r)


def build_shortlist(ranking: list[dict], *, synergy_min: float) -> list[dict]:
    shortlist = []
    s0_slot = ranking[0]["R_slot"] if ranking else 0.0
    for row in ranking:
        if row.get("condition") != "stacked_on_S0":
            continue
        syn = float(row.get("synergy_R_gender", 0) or 0)
        ok = False
        reason = []
        if row["axis"] == "gender" and syn >= synergy_min and row["R_gender_ci_lo"] > 0:
            ok = True
            reason.append(f"gender_synergy={syn:.4f}")
        if row["axis"] == "slot":
            if row["R_slot"] - s0_slot >= synergy_min:
                ok = True
                reason.append(f"slot_R_gain={row['R_slot'] - s0_slot:.4f}")
            if syn >= synergy_min:
                ok = True
                reason.append(f"gender_synergy={syn:.4f}")
        if row["axis"] == "random":
            continue
        if ok:
            shortlist.append(
                {
                    "id": row["id"].replace("__stack", ""),
                    "axis": row["axis"],
                    "layer": row["layer"],
                    "mode": row["mode"],
                    "R_gender_stack": row["R_gender"],
                    "synergy_R_gender": syn,
                    "R_slot_stack": row["R_slot"],
                    "reason": "; ".join(reason),
                    "next": "conditional_inlp_stage_a" if row["axis"] == "gender" else "stacked_slot_stage_a",
                }
            )
    shortlist.sort(key=lambda x: (-float(x["synergy_R_gender"]), -float(x["R_slot_stack"])))
    return shortlist


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    ap.add_argument("--n-items", type=int, default=None)
    ap.add_argument("--subspaces", type=Path, default=DEFAULT_SUBSPACES)
    ap.add_argument("--base-layer", type=int, default=15)
    ap.add_argument("--base-rank", type=int, default=16)
    ap.add_argument("--base-alpha", type=float, default=1.0)
    ap.add_argument("--layers", default="16,17,18,19,20,21,22,23,24")
    ap.add_argument("--gender-vec", default="w_gender_perp")
    ap.add_argument("--slot-vec", default="w_slot_perp")
    ap.add_argument("--second-alpha", type=float, default=1.0)
    ap.add_argument("--inlp-rank", type=int, default=16)
    ap.add_argument("--no-inlp-second", action="store_true")
    ap.add_argument("--no-project-out", action="store_true")
    ap.add_argument("--skip-alone", action="store_true", help="only stacked vs base (faster)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "results" / "steering")
    ap.add_argument("--tag", default="second_hit_v1")
    ap.add_argument("--log-every", type=int, default=40)
    ap.add_argument("--synergy-min", type=float, default=0.005, help="min R_stack - R_base to shortlist")
    ap.add_argument(
        "--finalize-from",
        type=Path,
        default=None,
        help="только пересобрать shortlist/summary из ranking.csv (без модели)",
    )
    args = ap.parse_args(argv)

    if args.finalize_from is not None:
        return finalize_from_dir(args.finalize_from, synergy_min=args.synergy_min, tag=args.tag)

    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    layers = parse_int_list(args.layers)
    sample = load_json(args.sample)
    items = pick_items(sample["items"], args.n_items)
    bank, calib = load_vector_bank(list(DEFAULT_VECTORS))
    if not args.subspaces.is_file():
        raise SystemExit(f"MISSING {args.subspaces}")
    inlp_arrays = {k: np.asarray(v, dtype=np.float32) for k, v in np.load(args.subspaces).items()}
    base_spec = build_inlp_spec(
        inlp_arrays,
        layer=args.base_layer,
        rank=args.base_rank,
        alpha=args.base_alpha,
        kind="center",
    )

    candidates = build_candidates(
        layers=layers,
        bank=bank,
        calib=calib,
        inlp_arrays=inlp_arrays,
        gender_vec=args.gender_vec,
        slot_vec=args.slot_vec,
        inlp_rank=args.inlp_rank,
        alpha=args.second_alpha,
        include_inlp=not args.no_inlp_second,
        include_project_out=not args.no_project_out,
    )
    print(
        f"second_hit [{args.tag}]: base=L{args.base_layer}k{args.base_rank} · "
        f"{len(items)} fam · {len(candidates)} candidates · layers {layers}"
    )
    if not candidates:
        raise SystemExit("нет кандидатов — проверьте vector banks / layers")

    print(f"\n[1] load {args.model}...")
    t0 = time.time()
    model, tokenizer = load_model(args.model, dtype=args.dtype, device=args.device)
    for p in model.parameters():
        p.requires_grad_(False)
    scorer = Scorer(model, tokenizer)
    print(f"  ready in {time.time() - t0:.1f}s")

    out_dir = args.out_root / "second_hit" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[2] baseline...")
    base_rows = run_rows(scorer, model, items, [], label="baseline", log_every=args.log_every)

    print("\n[3] S0 = L15 INLP alone...")
    s0_rows = run_rows(scorer, model, items, [base_spec], label="S0", log_every=args.log_every)
    s0_metrics = behavioral_metrics(base_rows, s0_rows, n_boot=args.n_boot, seed=args.seed)
    ranking = [
        metric_row(
            "S0_L15_inlp",
            {"axis": "base", "layer": args.base_layer, "mode": "inlp", "condition": "alone"},
            s0_metrics,
        )
    ]
    R_S0 = ranking[0]["R_gender"]
    print(f"  S0 R_gender={R_S0:.4f}  |θ| {ranking[0]['abs_theta_before']:.4f}→{ranking[0]['abs_theta_after']:.4f}")

    alone_cache: dict[str, dict] = {}
    results_detail = []

    for i, cand in enumerate(candidates, 1):
        cid = cand["id"]
        print(f"\n[{3 + i}/{3 + len(candidates)}] {cid}")
        alone_m = None
        if not args.skip_alone:
            alone_rows = run_rows(
                scorer, model, items, [cand["spec"]], label=f"{cid}_alone", log_every=args.log_every
            )
            alone_m = behavioral_metrics(base_rows, alone_rows, n_boot=args.n_boot, seed=args.seed)
            alone_cache[cid] = metric_row(
                f"{cid}__alone",
                {
                    "axis": cand["axis"],
                    "layer": cand["layer"],
                    "mode": cand["mode"],
                    "condition": "alone",
                },
                alone_m,
            )
            ranking.append(alone_cache[cid])

        stack_rows = run_rows(
            scorer,
            model,
            items,
            [base_spec, cand["spec"]],
            label=f"{cid}_stack",
            log_every=args.log_every,
        )
        stack_m = behavioral_metrics(base_rows, stack_rows, n_boot=args.n_boot, seed=args.seed)
        stack_row = metric_row(
            f"{cid}__stack",
            {
                "axis": cand["axis"],
                "layer": cand["layer"],
                "mode": cand["mode"],
                "condition": "stacked_on_S0",
            },
            stack_m,
        )
        R_stack = stack_row["R_gender"]
        synergy = R_stack - R_S0
        stack_row["synergy_R_gender"] = synergy
        R_alone = alone_cache[cid]["R_gender"] if cid in alone_cache else float("nan")
        stack_row["R_alone"] = R_alone
        stack_row["synergy_vs_sum"] = (
            R_stack - R_S0 - R_alone if np.isfinite(R_alone) else float("nan")
        )
        ranking.append(stack_row)
        results_detail.append(
            {
                "candidate": {k: v for k, v in cand.items() if k != "spec"},
                "stacked": stack_row,
                "alone": alone_cache.get(cid),
            }
        )
        # checkpoint after each candidate (crash-safe)
        (out_dir / "detail.json").write_text(
            json.dumps(results_detail, indent=2, default=str) + "\n", encoding="utf-8"
        )
        _write_ranking_csv(out_dir / "ranking.csv", ranking)
        print(
            f"  stack R_gender={R_stack:.4f}  synergy vs S0={synergy:+.4f}  "
            f"R_slot={stack_row['R_slot']:.4f}"
        )

    # shortlist + artifacts (JSON first, then CSV — crash-safe)
    shortlist = build_shortlist(ranking, synergy_min=args.synergy_min)
    csv_path = out_dir / "ranking.csv"
    summary = {
        "schema": "steering.second_hit_screen/v1",
        "tag": args.tag,
        "datetime": datetime.now().isoformat(),
        "model": args.model,
        "base": {
            "layer": args.base_layer,
            "rank": args.base_rank,
            "alpha": args.base_alpha,
            "subspaces": str(args.subspaces),
            "R_gender": R_S0,
        },
        "layers": layers,
        "n_families": len(items),
        "n_candidates": len(candidates),
        "synergy_min": args.synergy_min,
        "shortlist": shortlist,
        "read": (
            "synergy_R_gender = R(S0+second) − R(S0). "
            "Shortlist gender if synergy≥min and CI_lo(R_stack)>0; "
            "slot if R_slot gain or gender synergy. "
            "Next: conditional INLP Stage A on shortlisted layers under S0."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "shortlist.json").write_text(
        json.dumps({"shortlist": shortlist, "base": summary["base"]}, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "detail.json").write_text(json.dumps(results_detail, indent=2, default=str) + "\n", encoding="utf-8")
    _write_ranking_csv(csv_path, ranking)

    print("\n=== shortlist ===")
    if not shortlist:
        print("  (empty — no second site beats S0 by synergy_min)")
    for s in shortlist:
        print(
            f"  L{s['layer']} {s['axis']:6s} {s['mode']:20s}  "
            f"synergy={s['synergy_R_gender']:+.4f}  R_slot={s['R_slot_stack']:.4f}  [{s['reason']}]"
        )
    print(f"\nwrote {csv_path}")
    print(f"wrote {out_dir / 'shortlist.json'}")
    return 0


def finalize_from_dir(out_dir: Path, *, synergy_min: float, tag: str) -> int:
    """Rebuild shortlist/summary from checkpoint ranking.csv (no GPU)."""
    csv_path = out_dir / "ranking.csv"
    if not csv_path.is_file():
        raise SystemExit(f"MISSING {csv_path} — нужен checkpoint после прогона")
    ranking: list[dict] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            for k in (
                "R_gender",
                "R_gender_ci_lo",
                "R_gender_ci_hi",
                "R_slot",
                "synergy_R_gender",
                "R_alone",
                "synergy_vs_sum",
                "layer",
            ):
                if k in row and row[k] not in ("", None):
                    try:
                        row[k] = float(row[k]) if k != "layer" else int(float(row[k]))
                    except ValueError:
                        pass
            ranking.append(row)
    if not ranking:
        raise SystemExit(f"пустой {csv_path}")
    shortlist = build_shortlist(ranking, synergy_min=synergy_min)
    s0 = next((r for r in ranking if r.get("condition") == "alone" and r.get("axis") == "base"), ranking[0])
    base = {
        "layer": int(s0.get("layer", 15)),
        "rank": 16,
        "alpha": 1.0,
        "R_gender": float(s0.get("R_gender", float("nan"))),
    }
    summary = {
        "schema": "steering.second_hit_screen/v1",
        "tag": tag,
        "datetime": datetime.now().isoformat(),
        "finalized_from": str(csv_path),
        "base": base,
        "synergy_min": synergy_min,
        "shortlist": shortlist,
        "n_ranking_rows": len(ranking),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "shortlist.json").write_text(
        json.dumps({"shortlist": shortlist, "base": base}, indent=2) + "\n", encoding="utf-8"
    )
    _write_ranking_csv(csv_path, ranking)
    print(f"finalized {out_dir}: {len(shortlist)} shortlist from {len(ranking)} ranking rows")
    for s in shortlist:
        print(
            f"  L{s['layer']} {s['axis']:6s} {s['mode']:20s}  "
            f"synergy={s['synergy_R_gender']:+.4f}  R_slot={s['R_slot_stack']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
