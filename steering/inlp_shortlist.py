"""
Автовыбор shortlist INLP: Stage A → B и Stage B → C.

A→B (из ranking.csv):
  1) candidates с ci_lo > min_ci_lo и mean_R > matched random (тот же layer×rank)
  2) winning layer = argmax max(R) среди прошедших фильтр
  3) на этом слое top ``max_candidates`` по R (опц. отсев по auc_after)
  4) + matching rand0 controls

B→C (из keep.json Stage B):
  candidates с cap_loss ≤ max (уже в keep) → schema stagec_keep

CLI:
  python -m steering.inlp_shortlist from-stage-a \\
    --ranking .../ranking.csv --primary-axis gender --out shortlist.json

  python -m steering.inlp_shortlist from-stage-b \\
    --keep .../keep.json --out stagec_keep.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

CONFIG_ID_RE = re.compile(
    r"^(?P<basis>inlp|rand(?P<seed>\d+))__L(?P<layer>\d+)__k(?P<rank>\d+)__a(?P<alpha>.+)$"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_config_id(config_id: str) -> dict[str, Any] | None:
    m = CONFIG_ID_RE.match(config_id)
    if not m:
        return None
    alpha_raw = m.group("alpha").replace("p", ".")
    try:
        alpha = float(alpha_raw)
    except ValueError:
        alpha = alpha_raw
    return {
        "id": config_id,
        "basis": m.group("basis") if m.group("basis") == "inlp" else f"random_s{m.group('seed')}",
        "layer": int(m.group("layer")),
        "rank": int(m.group("rank")),
        "alpha": alpha,
        "seed": int(m.group("seed")) if m.group("seed") is not None else None,
    }


def layers_ranks_alphas_from_ids(ids: list[str]) -> tuple[list[int], list[int], list[float]]:
    layers: set[int] = set()
    ranks: set[int] = set()
    alphas: set[float] = set()
    for cid in ids:
        p = parse_config_id(cid)
        if not p:
            continue
        layers.add(int(p["layer"]))
        ranks.add(int(p["rank"]))
        if isinstance(p["alpha"], (int, float)):
            alphas.add(float(p["alpha"]))
    return sorted(layers), sorted(ranks), sorted(alphas) or [1.0]


def _f(x: Any) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def read_ranking_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        cid = r.get("config_id") or r.get("id")
        if not cid or cid == "baseline":
            continue
        row = dict(r)
        row["config_id"] = cid
        row["role"] = r.get("role") or "candidate"
        for key in ("mean_R_gender", "mean_R_slot", "ci_lo", "ci_hi", "auc_after_removal", "flip_rate", "alpha"):
            if key in row:
                row[key] = _f(row[key])
        for key in ("layer", "rank"):
            if key in row and row[key] not in (None, ""):
                row[key] = int(float(row[key]))
        if row.get("layer") is None or row.get("rank") is None:
            parsed = parse_config_id(cid)
            if parsed:
                row["layer"] = parsed["layer"]
                row["rank"] = parsed["rank"]
                row["alpha"] = parsed["alpha"]
        out.append(row)
    return out


def primary_R(row: dict[str, Any], primary_axis: str) -> float | None:
    if primary_axis == "slot":
        return _f(row.get("mean_R_slot"))
    return _f(row.get("mean_R_gender"))


def matched_random_R(
    controls: list[dict[str, Any]],
    *,
    layer: int,
    rank: int,
    primary_axis: str,
) -> float | None:
    exact = [c for c in controls if c.get("layer") == layer and c.get("rank") == rank]
    pool = exact or [c for c in controls if c.get("layer") == layer]
    vals = [primary_R(c, primary_axis) for c in pool]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def select_stageb_from_ranking(
    ranking: list[dict[str, Any]],
    *,
    primary_axis: str = "gender",
    max_candidates: int = 3,
    min_ci_lo: float = 0.0,
    require_above_random: bool = True,
    exclude_auc_at_or_below: float | None = None,
    include_random_controls: bool = True,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Построить shortlist Stage B из строк ranking Stage A."""
    candidates = [r for r in ranking if r.get("role") == "candidate"]
    controls = [r for r in ranking if r.get("role") == "control"]

    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for r in candidates:
        R = primary_R(r, primary_axis)
        ci_lo = _f(r.get("ci_lo"))
        reasons: list[str] = []
        if R is None:
            reasons.append("missing_R")
        if ci_lo is None:
            reasons.append("missing_ci_lo")
        elif ci_lo <= min_ci_lo:
            reasons.append(f"ci_lo={ci_lo:.4g}<={min_ci_lo}")
        rnd = None
        if (
            require_above_random
            and R is not None
            and r.get("layer") is not None
            and r.get("rank") is not None
        ):
            rnd = matched_random_R(
                controls,
                layer=int(r["layer"]),
                rank=int(r["rank"]),
                primary_axis=primary_axis,
            )
            if rnd is not None and not (R > rnd):
                reasons.append(f"R={R:.4g}<=random={rnd:.4g}")
        if reasons:
            rejected.append({"config_id": r["config_id"], "reasons": reasons, "R": R, "ci_lo": ci_lo})
            continue
        eligible.append({**r, "_R": R, "_random_R": rnd})

    if not eligible:
        scored = []
        for r in candidates:
            R = primary_R(r, primary_axis)
            if R is None:
                continue
            scored.append({**r, "_R": R, "_random_R": None})
        scored.sort(key=lambda x: -float(x["_R"]))
        eligible = scored[:max_candidates]
        selection_note = "fallback: no CI/random passers — top by R only"
        by_layer: dict[int, list[dict]] = {}
        for r in eligible:
            by_layer.setdefault(int(r["layer"]), []).append(r)
        layer_score = {L: max(float(x["_R"]) for x in rows) for L, rows in by_layer.items()}
        winning_layer = max(layer_score, key=layer_score.get) if layer_score else None
        chosen = eligible
    else:
        selection_note = (
            "winning layer by max R among CI>min & above random; top ranks on that layer"
        )
        by_layer = {}
        for r in eligible:
            by_layer.setdefault(int(r["layer"]), []).append(r)
        layer_score = {L: max(float(x["_R"]) for x in rows) for L, rows in by_layer.items()}
        winning_layer = max(layer_score, key=layer_score.get)

        pool = list(by_layer[winning_layer])
        preferred = pool
        if exclude_auc_at_or_below is not None:
            preferred = [
                r
                for r in pool
                if r.get("auc_after_removal") is None
                or float(r["auc_after_removal"]) > float(exclude_auc_at_or_below)
            ]
            if not preferred:
                preferred = pool
                selection_note += f"; auc filter {exclude_auc_at_or_below} emptied pool — ignored"
            else:
                selection_note += f"; prefer auc_after > {exclude_auc_at_or_below}"

        preferred.sort(key=lambda x: -float(x["_R"]))
        chosen = preferred[:max_candidates]

    entries: list[dict[str, Any]] = []
    for r in chosen:
        entries.append({"id": r["config_id"], "role": "candidate"})
    if include_random_controls:
        for r in chosen:
            parsed = parse_config_id(r["config_id"])
            a_tok = "1"
            if parsed and parsed["alpha"] != 1.0:
                a = parsed["alpha"]
                a_tok = str(a).replace(".", "p") if isinstance(a, float) else str(a)
            ctrl_id = f"rand{random_seed}__L{int(r['layer'])}__k{int(r['rank'])}__a{a_tok}"
            entries.append({"id": ctrl_id, "role": "control"})

    layers, ranks, alphas = layers_ranks_alphas_from_ids([e["id"] for e in entries])
    return {
        "schema": "steering.inlp_stageb_shortlist/v1",
        "source": "auto from Stage A ranking",
        "primary_axis": primary_axis,
        "selection": {
            "rule": selection_note,
            "winning_layer": winning_layer,
            "layer_scores": {str(k): v for k, v in sorted(layer_score.items())},
            "max_candidates": max_candidates,
            "min_ci_lo": min_ci_lo,
            "require_above_random": require_above_random,
            "exclude_auc_at_or_below": exclude_auc_at_or_below,
            "chosen": [
                {
                    "id": r["config_id"],
                    "R": r["_R"],
                    "ci_lo": r.get("ci_lo"),
                    "auc_after": r.get("auc_after_removal"),
                    "random_R": r.get("_random_R"),
                }
                for r in chosen
            ],
            "n_eligible": len(eligible) if eligible else 0,
            "n_rejected": len(rejected),
        },
        "grid": {"layers": layers, "ranks": ranks, "alphas": alphas},
        "candidates": entries,
    }


def select_stagec_from_keep(
    keep_doc: dict[str, Any],
    *,
    hypothesis: str | None = None,
    model_scale: str | None = None,
) -> dict[str, Any]:
    """Stage C keep из Stage B keep.json (уже cap_loss-filtered)."""
    entries = []
    for c in keep_doc.get("candidates", []):
        if isinstance(c, str):
            cid, role = c, "candidate"
            flag = False
        else:
            cid = c.get("id") or c.get("config_id")
            role = c.get("role", "candidate")
            flag = bool(c.get("cap_loss_flag"))
        if not cid or role == "control" or flag:
            continue
        entries.append({"id": cid, "role": "candidate"})

    layers, ranks, alphas = layers_ranks_alphas_from_ids([e["id"] for e in entries])
    doc: dict[str, Any] = {
        "schema": "steering.inlp_stagec_keep/v1",
        "source": "auto from Stage B keep.json",
        "note": keep_doc.get("rule", "cap_loss keep"),
        "cap_loss_max": keep_doc.get("cap_loss_max"),
        "grid": {"layers": layers, "ranks": ranks, "alphas": alphas},
        "candidates": entries,
    }
    if hypothesis:
        doc["hypothesis"] = hypothesis
    if model_scale:
        doc["model_scale"] = model_scale
    return doc


def shortlist_entry_ids(doc: dict[str, Any]) -> list[str]:
    ids = []
    for c in doc.get("candidates", []):
        if isinstance(c, str):
            ids.append(c)
        else:
            cid = c.get("id") or c.get("config_id")
            if cid:
                ids.append(cid)
    return ids


def cmd_from_stage_a(args: argparse.Namespace) -> int:
    ranking = read_ranking_csv(args.ranking)
    primary = args.primary_axis
    if primary == "auto":
        primary = "slot" if args.prefer_slot else "gender"
    doc = select_stageb_from_ranking(
        ranking,
        primary_axis=primary,
        max_candidates=args.max_candidates,
        min_ci_lo=args.min_ci_lo,
        require_above_random=not args.no_require_above_random,
        exclude_auc_at_or_below=args.exclude_auc_at_or_below,
        include_random_controls=not args.no_random_controls,
        random_seed=args.random_seed,
    )
    if args.hypothesis:
        doc["hypothesis"] = args.hypothesis
    if args.model_scale:
        doc["model_scale"] = args.model_scale
    doc["source"] = f"auto from {args.ranking}"
    write_json(args.out, doc)
    print(f"wrote {args.out}")
    print(f"  winning_layer=L{doc['selection']['winning_layer']}  grid={doc['grid']}")
    for c in doc["candidates"]:
        print(f"  {c['role']:9} {c['id']}")
    return 0


def cmd_from_stage_b(args: argparse.Namespace) -> int:
    keep = load_json(args.keep)
    doc = select_stagec_from_keep(
        keep, hypothesis=args.hypothesis, model_scale=args.model_scale
    )
    doc["source"] = f"auto from {args.keep}"
    write_json(args.out, doc)
    print(f"wrote {args.out}")
    print(f"  grid={doc['grid']}  n={len(doc['candidates'])}")
    for c in doc["candidates"]:
        print(f"  {c['id']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("from-stage-a", help="ranking.csv → Stage B shortlist")
    a.add_argument("--ranking", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--primary-axis", choices=["gender", "slot", "auto"], default="auto")
    a.add_argument("--prefer-slot", action="store_true", help="with auto: use slot axis")
    a.add_argument("--max-candidates", type=int, default=3)
    a.add_argument("--min-ci-lo", type=float, default=0.0)
    a.add_argument("--no-require-above-random", action="store_true")
    a.add_argument(
        "--exclude-auc-at-or-below",
        type=float,
        default=None,
        help="prefer ranks with auc_after above this (e.g. 0.55)",
    )
    a.add_argument("--no-random-controls", action="store_true")
    a.add_argument("--random-seed", type=int, default=0)
    a.add_argument("--hypothesis", default=None)
    a.add_argument("--model-scale", default=None)
    a.set_defaults(func=cmd_from_stage_a)

    b = sub.add_parser("from-stage-b", help="Stage B keep.json → Stage C keep")
    b.add_argument("--keep", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--hypothesis", default=None)
    b.add_argument("--model-scale", default=None)
    b.set_defaults(func=cmd_from_stage_b)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
