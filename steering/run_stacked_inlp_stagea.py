"""
Stacked INLP Stage A under fixed S0 (default L15 k16 gender INLP).

Every candidate is evaluated as **S0 + second-site**, plus reference runs:
  - true baseline (no hooks)
  - S0 alone

Second-site subspaces should be **conditional** (fit under S0), from
`build_conditional_inlp_subspace`, but any Stage-A-shaped npz works.

Examples:

    python -m steering.run_stacked_inlp_stagea \\
        --model Qwen/Qwen3.5-2B-Base --device cuda --dtype float32 \\
        --second-subspaces steering/subspaces/inlp_cond_gender_choice_cond_under_l15_v1.npz \\
        --layers 16,20 --ranks 8,16,32 --tag stacked_gender_v1

    # smoke
    python -m steering.run_stacked_inlp_stagea --device cuda --limit-items 2 \\
        --second-subspaces ... --layers 20 --ranks 8 --tag smoke
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

from steering.intervene import Scorer, SubspaceSpec, SubspaceTrace, load_model, steered
from steering.run_hs_recovery_auc import build_inlp_spec
from steering.run_inlp_stagea import (
    behavioral_metrics,
    build_configs,
    load_json,
    parse_float_list,
    parse_int_list,
    row_margins,
    spec_for,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STEERING_DIR = Path(__file__).resolve().parent
DEFAULT_BASE = STEERING_DIR / "subspaces" / "inlp_gender_choice_v1.npz"
DEFAULT_SAMPLE = STEERING_DIR / "samples" / "h1_stagea_sample_v1.json"


def run_rows(
    scorer: Scorer,
    model,
    items: list[dict],
    specs: list[SubspaceSpec],
    *,
    label: str,
    log_every: int,
    trace_layer: int | None = None,
) -> list[dict]:
    trace = SubspaceTrace() if trace_layer is not None else None
    rows: list[dict] = []
    total = sum(len(i["rows"]) for i in items)
    t0 = time.time()
    for item in items:
        for src in item["rows"]:
            if trace is not None:
                trace.clear()
            with steered(model, specs if specs else [], trace):
                out = scorer.score(src["prompt"], list(src["valid_labels"]))
            labels = src["labels"]
            row = {
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
            if trace is not None and trace_layer is not None and specs:
                # hook check on the *second* site if present, else first
                L = trace_layer
                s_b = trace.before.get(L)
                s_a = trace.after.get(L)
                if s_b is not None and s_a is not None:
                    # find matching spec
                    for sp in specs:
                        if sp.layer == L:
                            err = np.abs(
                                s_a - sp.expected_s_after(np.asarray(s_b, dtype=np.float64))
                            )
                            row["projection_error_max"] = float(err.max())
                            row["projection_error_mean"] = float(err.mean())
                            break
            rows.append(row)
            if log_every and len(rows) % log_every == 0:
                rate = len(rows) / max(time.time() - t0, 1e-9)
                eta = (total - len(rows)) / max(rate, 1e-9) / 60
                print(f"    [{label}] {len(rows)}/{total}  {rate:.2f} row/s  ETA {eta:.1f} min", flush=True)
    return rows


def ranking_row(tag: str, meta: dict, metrics: dict, *, primary: str) -> dict:
    g = metrics["gender_axis"]["reduction"]
    s = metrics["slot_axis"]["reduction"]
    th = metrics["theta_prob_axis"]["reduction"]
    primary_block = g if primary == "gender" else s
    return {
        "id": tag,
        **meta,
        "R_primary": primary_block["mean"],
        "R_primary_ci_lo": primary_block["ci_lo"],
        "R_primary_ci_hi": primary_block["ci_hi"],
        "frac_R_primary_pos": primary_block["fraction_positive"],
        "R_gender": g["mean"],
        "R_gender_ci_lo": g["ci_lo"],
        "R_slot": s["mean"],
        "abs_theta_before": metrics["theta_prob_axis"]["mean_abs_before"],
        "abs_theta_after": metrics["theta_prob_axis"]["mean_abs_after"],
        "R_theta": th["mean"],
        "flip_rate": metrics["flip_rate"],
        "|D|_before": metrics["gender_axis"]["mean_abs_before"],
        "|D|_after": metrics["gender_axis"]["mean_abs_after"],
        "|S|_before": metrics["slot_axis"]["mean_abs_before"],
        "|S|_after": metrics["slot_axis"]["mean_abs_after"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    ap.add_argument("--limit-items", type=int, default=None)
    ap.add_argument("--base-subspaces", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--base-layer", type=int, default=15)
    ap.add_argument("--base-rank", type=int, default=16)
    ap.add_argument("--base-alpha", type=float, default=1.0)
    ap.add_argument("--second-subspaces", type=Path, required=True)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--ranks", default="8,16,32")
    ap.add_argument("--alphas", default="1.0")
    ap.add_argument("--no-random-control", action="store_true")
    ap.add_argument("--random-seeds", default="0")
    ap.add_argument("--primary-axis", choices=["gender", "slot", "auto"], default="auto")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "results" / "steering")
    ap.add_argument("--tag", default="stacked_inlp_v1")
    ap.add_argument("--log-every", type=int, default=40)
    ap.add_argument("--from-shortlist", type=Path, default=None, help="restrict layers to shortlist")
    args = ap.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    if not args.base_subspaces.is_file():
        raise SystemExit(f"MISSING {args.base_subspaces}")
    if not args.second_subspaces.is_file():
        raise SystemExit(f"MISSING {args.second_subspaces}")

    base_arrays = {k: np.asarray(v, dtype=np.float32) for k, v in np.load(args.base_subspaces).items()}
    base_spec = build_inlp_spec(
        base_arrays,
        layer=args.base_layer,
        rank=args.base_rank,
        alpha=args.base_alpha,
        kind="center",
    )

    second_meta = load_json(args.second_subspaces.with_suffix(".json"))
    with np.load(args.second_subspaces) as z:
        second_arrays = {k: np.asarray(z[k], dtype=np.float32) for k in z.files}

    primary = args.primary_axis
    if primary == "auto":
        tgt = str(second_meta.get("target", "gender_choice"))
        primary = "slot" if "slot" in tgt else "gender"

    layers = parse_int_list(args.layers)
    if args.from_shortlist is not None:
        sl = load_json(args.from_shortlist).get("shortlist", [])
        want_axis = "slot" if primary == "slot" else "gender"
        from_sl = sorted({int(e["layer"]) for e in sl if e.get("axis") == want_axis})
        if from_sl:
            layers = from_sl if layers is None else [L for L in layers if L in from_sl]
            print(f"layers from shortlist ({want_axis}): {layers}")

    configs = build_configs(
        second_meta,
        set(second_arrays),
        layers=layers,
        ranks=parse_int_list(args.ranks),
        alphas=parse_float_list(args.alphas) or [1.0],
        random_seeds=parse_int_list(args.random_seeds) or [0],
        include_random=not args.no_random_control,
    )
    if not configs:
        raise SystemExit("пустой план конфигураций — проверьте layers/ranks vs second subspaces")

    sample = load_json(args.sample)
    items = sample["items"][: args.limit_items] if args.limit_items else sample["items"]
    n_rows = sum(len(i["rows"]) for i in items)
    out_dir = args.out_root / "stacked_inlp_stage_a" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"stacked Stage A [{args.tag}]: S0=L{args.base_layer}k{args.base_rank} + "
        f"{len(configs)} second sites · {len(items)} fam · primary={primary}"
    )

    print(f"\n[1] load {args.model}...")
    t0 = time.time()
    model, tokenizer = load_model(args.model, dtype=args.dtype, device=args.device)
    for p in model.parameters():
        p.requires_grad_(False)
    scorer = Scorer(model, tokenizer)
    print(f"  ready in {time.time() - t0:.1f}s")

    print("\n[2] baseline...")
    base_rows = run_rows(scorer, model, items, [], label="baseline", log_every=args.log_every)

    print("\n[3] S0 alone...")
    s0_rows = run_rows(
        scorer, model, items, [base_spec], label="S0", log_every=args.log_every, trace_layer=args.base_layer
    )
    s0_m = behavioral_metrics(base_rows, s0_rows, n_boot=args.n_boot, seed=args.seed)
    ranking = [
        ranking_row(
            "S0_alone",
            {"role": "base", "basis": "inlp", "layer": args.base_layer, "rank": args.base_rank, "alpha": args.base_alpha},
            s0_m,
            primary=primary,
        )
    ]
    R_S0 = ranking[0]["R_primary"]
    print(f"  S0 R_primary={R_S0:.4f}")

    for i, cfg in enumerate(configs, 1):
        sp2 = spec_for(cfg, second_arrays)
        cid = f"S0+{cfg['id']}"
        print(f"\n[{3 + i}/{3 + len(configs)}] {cid}")
        stack_rows = run_rows(
            scorer,
            model,
            items,
            [base_spec, sp2],
            label=cid,
            log_every=args.log_every,
            trace_layer=sp2.layer,
        )
        m = behavioral_metrics(base_rows, stack_rows, n_boot=args.n_boot, seed=args.seed)
        row = ranking_row(
            cid,
            {
                "role": cfg["role"],
                "basis": cfg["basis"],
                "layer": cfg["layer"],
                "rank": cfg["rank"],
                "alpha": cfg["alpha"],
            },
            m,
            primary=primary,
        )
        row["synergy_vs_S0"] = row["R_primary"] - R_S0
        ranking.append(row)

        cdir = out_dir / cid.replace("+", "_")
        cdir.mkdir(parents=True, exist_ok=True)
        with (cdir / "per_row.jsonl").open("w", encoding="utf-8") as f:
            for r in stack_rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        (cdir / "metrics.json").write_text(
            json.dumps({"config_id": cid, "metrics": m, "ranking_row": row}, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(
            f"  R_primary={row['R_primary']:.4f}  synergy={row['synergy_vs_S0']:+.4f}  "
            f"R_gender={row['R_gender']:.4f}  R_slot={row['R_slot']:.4f}"
        )

    # shortlist: INLP (not random) with synergy > 0 and CI_lo > 0
    shortlist = []
    for row in ranking:
        if row.get("role") != "candidate":
            continue
        if float(row.get("synergy_vs_S0", 0)) < 0.005:
            continue
        if float(row.get("R_primary_ci_lo", -1)) <= 0:
            continue
        shortlist.append(
            {
                "id": row["id"],
                "layer": row["layer"],
                "rank": row["rank"],
                "alpha": row["alpha"],
                "R_primary": row["R_primary"],
                "synergy_vs_S0": row["synergy_vs_S0"],
                "R_gender": row["R_gender"],
                "R_slot": row["R_slot"],
            }
        )
    shortlist.sort(key=lambda x: (-float(x["synergy_vs_S0"]), -float(x["R_primary"])))

    csv_path = out_dir / "ranking.csv"
    keys = list(ranking[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in ranking:
            w.writerow(r)

    summary = {
        "schema": "steering.stacked_inlp_stage_a/v1",
        "tag": args.tag,
        "datetime": datetime.now().isoformat(),
        "model": args.model,
        "primary_axis": primary,
        "base": {
            "layer": args.base_layer,
            "rank": args.base_rank,
            "alpha": args.base_alpha,
            "subspaces": str(args.base_subspaces),
            "R_primary": R_S0,
        },
        "second_subspaces": str(args.second_subspaces),
        "second_target": second_meta.get("target"),
        "n_families": len(items),
        "n_rows": n_rows,
        "n_configs": len(configs),
        "shortlist": shortlist,
        "read": (
            "R_primary vs true baseline. synergy_vs_S0 = R(S0+second) − R(S0). "
            "Next: method-3 AUC under winning stack; then Stage B/C if keep."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "stageb_shortlist.json").write_text(
        json.dumps({"shortlist": shortlist, "base": summary["base"], "source": f"stacked Stage A [{args.tag}]"}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    print("\n=== shortlist (synergy ≥ 0.005, CI_lo>0) ===")
    if not shortlist:
        print("  (empty)")
    for s in shortlist[:10]:
        print(
            f"  L{s['layer']} k{s['rank']}  R={s['R_primary']:.4f}  "
            f"synergy={s['synergy_vs_S0']:+.4f}"
        )
    print(f"\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
