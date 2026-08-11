"""
Разворачивание сетки H1 steering-кандидатов в замороженный список.

Вход:  steering/configs/h1_steering_candidates_v1.yaml
Выход: steering/candidates/h1_candidates_v1.json
       steering/candidates/h1_candidates_v1.md

Запуск:
    python -m steering.build_h1_candidates
    python -m steering.build_h1_candidates --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

STEERING_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = STEERING_DIR / "configs" / "h1_steering_candidates_v1.yaml"
DEFAULT_OUT_DIR = STEERING_DIR / "candidates"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_layers(spec: Any, layers_cfg: dict) -> list[int]:
    if isinstance(spec, str):
        if spec not in layers_cfg:
            raise ValueError(f"unknown layers key {spec!r}")
        val = layers_cfg[spec]
        if not isinstance(val, list):
            raise ValueError(f"layers.{spec} must be a list")
        return [int(x) for x in val]
    if isinstance(spec, list):
        return [int(x) for x in spec]
    raise ValueError(f"layers must be list or named belt/core/edge, got {spec!r}")


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    vec_ids = [v["id"] for v in cfg["vectors"]]
    if len(vec_ids) != len(set(vec_ids)):
        raise ValueError("duplicate vector id")
    vec_by_id = {v["id"]: v for v in cfg["vectors"]}

    int_ids = [i["id"] for i in cfg["interventions"]]
    if len(int_ids) != len(set(int_ids)):
        raise ValueError("duplicate intervention id")
    int_by_id = {i["id"]: i for i in cfg["interventions"]}

    for fam in cfg["families"]:
        for vid in fam["vectors"]:
            if vid not in vec_by_id:
                raise ValueError(f"family {fam['id']}: unknown vector {vid}")
        if fam["intervention"] not in int_by_id:
            raise ValueError(f"family {fam['id']}: unknown intervention {fam['intervention']}")
        resolve_layers(fam["layers"], cfg["layers"])

    return cfg


def strength_values(fam: dict, intervention: dict) -> list[tuple[str, float]]:
    """Список (param_name, value) для декартова произведения."""
    iid = intervention["id"]
    if iid == "shift":
        vals = fam.get("betas", intervention["betas"])
        return [("beta", float(v)) for v in vals]
    if "alphas" in fam:
        vals = fam["alphas"]
    else:
        vals = intervention.get("alphas", [1.0])
    return [("alpha", float(v)) for v in vals]


def expand_family(fam: dict, cfg: dict) -> list[dict]:
    if not fam.get("enabled", True):
        return []

    vec_by_id = {v["id"]: v for v in cfg["vectors"]}
    intervention = next(i for i in cfg["interventions"] if i["id"] == fam["intervention"])
    layers = resolve_layers(fam["layers"], cfg["layers"])
    multi = bool(fam.get("apply_to_all_layers_at_once", False))
    strengths = strength_values(fam, intervention)

    out: list[dict] = []
    for vid in fam["vectors"]:
        vec = vec_by_id[vid]
        seeds = vec.get("seeds", [None]) if vec["kind"] == "random_unit" else [None]
        for seed in seeds:
            for param_name, strength in strengths:
                layer_specs: list[list[int]] = [layers] if multi else [[L] for L in layers]
                for layer_list in layer_specs:
                    cand_id = make_candidate_id(
                        fam_id=fam["id"],
                        vec_short=vec["short"],
                        layers=layer_list,
                        multi=multi,
                        intervention=intervention["id"],
                        param_name=param_name,
                        strength=strength,
                        seed=seed,
                    )
                    row: dict[str, Any] = {
                        "id": cand_id,
                        "family": fam["id"],
                        "role": fam["role"],
                        "vector_id": vid,
                        "vector_kind": vec["kind"],
                        "vector_base": vec.get("base"),
                        "orthogonalize_against": list(vec.get("orthogonalize_against") or []),
                        "layers": layer_list,
                        "apply_to_all_layers_at_once": multi,
                        "intervention": intervention["id"],
                        "formula": intervention["formula"],
                        param_name: strength,
                        "reference_c": intervention.get("reference_c"),
                        "scale": intervention.get("scale"),
                    }
                    if seed is not None:
                        row["random_seed"] = int(seed)
                    out.append(row)
    return out


def fmt_strength(x: float) -> str:
    s = f"{x:g}"
    return s.replace("-", "m").replace(".", "p")


def make_candidate_id(
    *,
    fam_id: str,
    vec_short: str,
    layers: list[int],
    multi: bool,
    intervention: str,
    param_name: str,
    strength: float,
    seed: int | None,
) -> str:
    if multi:
        layer_part = "L" + "-".join(str(L) for L in layers)
    else:
        layer_part = f"L{layers[0]}"
    parts = [fam_id, vec_short, layer_part, intervention, f"{param_name[0]}{fmt_strength(strength)}"]
    if seed is not None:
        parts.append(f"s{seed}")
    return "__".join(parts)


def build_document(cfg: dict, cfg_path: Path) -> dict:
    candidates: list[dict] = []
    for fam in cfg["families"]:
        candidates.extend(expand_family(fam, cfg))

    ids = [c["id"] for c in candidates]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate candidate ids: {dupes[:5]}")

    by_role: dict[str, int] = {}
    by_family: dict[str, int] = {}
    for c in candidates:
        by_role[c["role"]] = by_role.get(c["role"], 0) + 1
        by_family[c["family"]] = by_family.get(c["family"], 0) + 1

    return {
        "schema": "steering.h1_candidates/v1",
        "hypothesis": cfg["hypothesis"],
        "map_version": cfg["version"],
        "map_config": cfg_path.name,
        "map_sha256": sha256_file(cfg_path),
        "source": cfg["source"],
        "hook": cfg["hook"],
        "vectors": cfg["vectors"],
        "interventions": cfg["interventions"],
        "layers": cfg["layers"],
        "baseline": cfg["baseline"],
        "screening": cfg["screening"],
        "n_candidates": len(candidates),
        "n_by_role": by_role,
        "n_by_family": dict(sorted(by_family.items())),
        "forwards_estimate": {
            "per_candidate": cfg["screening"]["forwards_per_candidate"],
            "total_with_baseline": (len(candidates) + 1) * cfg["screening"]["forwards_per_candidate"],
        },
        "candidates": candidates,
    }


def summary_markdown(doc: dict) -> str:
    lines = [
        f"# H1 steering candidates — `{doc['map_version']}`",
        "",
        f"- Конфиг: `{doc['map_config']}`",
        f"- Кандидатов: **{doc['n_candidates']}** "
        f"(candidate {doc['n_by_role'].get('candidate', 0)}, "
        f"control {doc['n_by_role'].get('control', 0)}) + baseline",
        f"- Forward estimate Stage A: **{doc['forwards_estimate']['total_with_baseline']}** "
        f"({doc['forwards_estimate']['per_candidate']} × n+1)",
        f"- Hook: `{doc['hook']['site']}`, `{doc['hook']['positions']}`",
        "",
        "## По семействам",
        "",
        "| family | n | role |",
        "| :--- | ---: | :--- |",
    ]
    role_by_fam = {c["family"]: c["role"] for c in doc["candidates"]}
    for fam, n in doc["n_by_family"].items():
        lines.append(f"| `{fam}` | {n} | {role_by_fam.get(fam, '?')} |")

    lines += [
        "",
        "## Primary metric (Stage A)",
        "",
        f"- `{doc['screening']['primary_metric']['id']}`: "
        f"{doc['screening']['primary_metric']['formula']}",
        f"- keep_top: **{doc['screening']['keep_top']}**",
        "",
        "Полный список id — в JSON (`candidates[].id`).",
        "",
    ]
    return "\n".join(lines)


def dump_json(path: Path, doc: dict) -> str:
    text = json.dumps(doc, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    ver = cfg["version"]
    doc = build_document(cfg, args.config)
    md = summary_markdown(doc)

    json_path = args.out_dir / f"h1_candidates_{ver}.json"
    md_path = args.out_dir / f"h1_candidates_{ver}.md"

    if args.verify:
        expected = json.dumps(doc, ensure_ascii=False, indent=2) + "\n"
        bad = []
        if not json_path.exists() or json_path.read_text(encoding="utf-8") != expected:
            bad.append(json_path.name)
        if not md_path.exists() or md_path.read_text(encoding="utf-8") != md:
            bad.append(md_path.name)
        if bad:
            print("MISMATCH: " + ", ".join(bad))
            return 1
        print("OK — кандидаты совпадают с пересборкой")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(json_path, doc)
    md_path.write_text(md, encoding="utf-8")

    print(
        f"h1 {ver}: {doc['n_candidates']} candidates "
        f"(candidate {doc['n_by_role'].get('candidate', 0)}, "
        f"control {doc['n_by_role'].get('control', 0)})"
    )
    for fam, n in doc["n_by_family"].items():
        print(f"  {fam:<22} {n:>3}")
    print(f"  -> {json_path.name}")
    print(f"  -> {md_path.name}")
    print(f"  Stage A forwards ≈ {doc['forwards_estimate']['total_with_baseline']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
