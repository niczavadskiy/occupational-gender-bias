"""Freeze Gemma steering peaks + FDR polarity sets from H1 catalog.

Run from repo root:
  python experiments/_freeze_gemma_steering.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from steering.xy_control.build_domains import build_catalog, catalog_markdown  # noqa: E402
from steering.xy_control.domains import CATALOG_JSON, CATALOG_MD  # noqa: E402

PEAKS = {
    "gemma3_1b": {
        "gender": {"peak": 9, "layers": [9, 8, 7], "source": "gender_choice L9"},
        "slot": {"peak": 20, "layers": [20, 19, 18], "source": "slot_choice L20"},
        "model": "google/gemma-3-1b-pt",
        "d_model": 1152,
        "n_hs": 27,
    },
    "gemma3_4b": {
        "gender": {"peak": 24, "layers": [24, 23, 22], "source": "gender_choice L24"},
        "slot": {"peak": 31, "layers": [31, 30, 29], "source": "slot_choice L31"},
        "model": "google/gemma-3-4b-pt",
        "d_model": 2560,
        "n_hs": 35,
    },
}


def _replace_peaks(text: str, peak: int, m1: int, m2: int) -> str:
    # Longer tokens first.
    out = text
    out = out.replace("PEAK_M2", str(m2))
    out = out.replace("PEAK_M1", str(m1))
    out = out.replace("PEAK−2", str(m2))
    out = out.replace("PEAK−1", str(m1))
    out = out.replace("PEAK", str(peak))
    out = re.sub(r"status:\s*STUB_FILL_PEAK", "status: READY", out)
    out = re.sub(r"version:\s*v1_peak_prepeak_stub", "version: v1_peak_prepeak", out)
    out = re.sub(
        r"version:\s*v1_peak_prepeak_a6_stub", "version: v1_peak_prepeak_a6", out
    )
    return out


def fill_yaml(path: Path, peak: int) -> None:
    if not path.is_file():
        print(f"  miss yaml {path}")
        return
    m1, m2 = peak - 1, peak - 2
    text = path.read_text(encoding="utf-8")
    if "PEAK" not in text and "STUB_FILL_PEAK" not in text:
        print(f"  skip (already filled) {path.relative_to(REPO)}")
        return
    path.write_text(_replace_peaks(text, peak, m1, m2), encoding="utf-8")
    print(f"  filled {path.relative_to(REPO)} → L{peak}/{m1}/{m2}")


def write_slot_yaml(scale: str, meta: dict) -> None:
    peak = meta["slot"]["peak"]
    layers = meta["slot"]["layers"]
    exp = REPO / "experiments" / (
        "gemma3-1b-pt" if scale == "gemma3_1b" else "gemma3-4b-pt"
    )
    path = exp / "steering" / "configs" / f"inlp_slot_prob_{scale}_v1.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    hs_note = (
        f"0 = embeddings, 1..{meta['n_hs'] - 1} = block output"
    )
    body = f"""# Classical slot_prob INLP · {meta['model']}
# Belt = slot_choice probe peak + two layers below.

version: v1_peak_prepeak
hypothesis: inlp_slot_prob
scale: {scale}
created: 2026-09-28
status: READY
target: slot_prob

source:
  model_id: {meta['model']}
  d_model: {meta['d_model']}
  n_hidden_states: {meta['n_hs']}
  hidden_state_index: "{hs_note}"
  label_note: >-
    Classical INLP on slot last-token HS (slot_prob / p_A). Peak from
    slot_choice classification probe ({meta['slot']['source']}).

hook:
  site: residual_stream_hidden_state
  positions: last_prompt_token
  intervention: "h' = h − α W_k (W_kᵀ h − c)"

probe:
  ridge_alpha: 1.0

inlp:
  k_max: 32
  chance:
    corr_threshold: 0.15

layers:
  peak: {peak}
  prepeak: [{layers[1]}, {layers[2]}]
  belt: [{layers[0]}, {layers[1]}, {layers[2]}]
  core: [{layers[0]}, {layers[1]}, {layers[2]}]
  anchor: {peak}

steering:
  kind: center
  ranks: [1, 4, 8, 16]
  alphas: [1.0]
  random_control: true
  random_seeds: [0]

split:
  train_frac: 0.70
  val_frac: 0.15
  test_frac: 0.15
  seed: 0
  stratify_by: soc_major_title

families:
  - id: main_center_peak_prepeak
    enabled: true
    role: candidate
    layers: [{layers[0]}, {layers[1]}, {layers[2]}]
    ranks: [1, 4, 8, 16]
    alphas: [1.0]
    description: slot peak + peak−1 + peak−2 × ranks; Stage A clamps to k_found.
"""
    path.write_text(body, encoding="utf-8")
    print(f"  wrote {path.relative_to(REPO)}")


def fill_polarity_sets(catalog: dict) -> None:
    for scale, meta in PEAKS.items():
        block = catalog["scales"][scale]
        male = [d["slug"] for d in block["steer"] if d["polarity"] == "male"]
        female = [d["slug"] for d in block["steer"] if d["polarity"] == "female"]
        gpeak = meta["gender"]
        exp = REPO / "experiments" / (
            "gemma3-1b-pt" if scale == "gemma3_1b" else "gemma3-4b-pt"
        )
        xy_paths = [
            exp / "steering" / "domains" / f"polarity_sets_{scale}_v1.json",
            exp / "steering" / "xy_control" / "domains" / f"polarity_sets_{scale}_v1.json",
        ]
        inlp_path = exp / "steering" / "domains" / f"inlp_polarity_sets_{scale}_v1.json"

        xy_doc = {
            "schema": "steering.xy_control_polarity_sets/v1",
            "version": "v2_pooled",
            "scale": scale,
            "status": "READY",
            "catalog": "steering/xy_control/domains/h1_soc_fdr_v1.json",
            "config": f"experiments/{exp.name}/steering/xy_control/configs/xy_control_{scale}_peak_prepeak_a6.yaml",
            "protocol": {
                "id": "pooled_polarity_one_v",
                "note": (
                    "One v_raw per polarity set. Empty sets are skipped at runtime."
                ),
            },
            "probe_peak": {
                "source": gpeak["source"],
                "peak": gpeak["peak"],
                "layers": gpeak["layers"],
                "note": "gender_choice probe peak; also peak−1, peak−2",
            },
            "note": (
                f"FDR-significant {scale} SOCs split by H1 polarity. "
                f"male={len(male)} female={len(female)}. Skip empty sets."
            ),
            "sets": [
                {
                    "id": "promale",
                    "label": "pro-male",
                    "polarity": "male",
                    "tag_stage_a": f"xy_{scale}_peak_prepeak_a6_pool_promale",
                    "tag_stage_b": f"xy_{scale}_peak_prepeak_a6_pool_promale_b",
                    "tag_stage_c": f"xy_{scale}_peak_prepeak_a6_pool_promale_c",
                    "n_soc": len(male),
                    "slugs": male,
                },
                {
                    "id": "profemale",
                    "label": "pro-female",
                    "polarity": "female",
                    "tag_stage_a": f"xy_{scale}_peak_prepeak_a6_pool_profemale",
                    "tag_stage_b": f"xy_{scale}_peak_prepeak_a6_pool_profemale_b",
                    "tag_stage_c": f"xy_{scale}_peak_prepeak_a6_pool_profemale_c",
                    "n_soc": len(female),
                    "slugs": female,
                },
            ],
        }
        for p in xy_paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                json.dumps(xy_doc, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"  wrote {p.relative_to(REPO)}")

        inlp_doc = {
            "schema": "steering.inlp_polarity_sets/v1",
            "version": "v1_pooled",
            "scale": scale,
            "status": "READY",
            "catalog": "steering/xy_control/domains/h1_soc_fdr_v1.json",
            "xy_sets": f"experiments/{exp.name}/steering/domains/polarity_sets_{scale}_v1.json",
            "config": f"experiments/{exp.name}/steering/configs/inlp_polarity_pool_{scale}_peak_prepeak.yaml",
            "xy_dataset": "steering/xy_control/data/xy_pairs_full_v1.json",
            "protocol": {
                "id": "pooled_polarity_one_subspace",
                "note": (
                    "One INLP subspace W per polarity set. Empty sets skipped."
                ),
            },
            "probe_peak": {
                "source": gpeak["source"],
                "peak": gpeak["peak"],
                "layers": gpeak["layers"],
                "note": "matched to XY peak_prepeak gender belt",
            },
            "candidate_grid": {
                "layers": gpeak["layers"],
                "ranks": [1, 4, 8, 16],
                "alphas": [1.0],
                "n_candidates_preferred": 12,
                "note": "3×4×1 preferred; Stage A clamps ranks to k_found",
            },
            "note": (
                f"FDR-significant {scale} SOCs. Matched arm to XY polarity pool. "
                "Gender only (no slot in pool protocols)."
            ),
            "sets": [
                {
                    "id": "promale",
                    "label": "pro-male",
                    "polarity": "male",
                    "sample_stem": f"inlp_polarity_pool_{scale}_promale_v1",
                    "tag_stage_a": f"inlp_{scale}_peak_prepeak_pool_promale",
                    "tag_stage_b": f"inlp_{scale}_peak_prepeak_pool_promale_b",
                    "tag_stage_c": f"inlp_{scale}_peak_prepeak_pool_promale_c",
                    "subspace_tag": f"polarity_pool_{scale}_promale_v1",
                    "n_soc": len(male),
                    "slugs": male,
                },
                {
                    "id": "profemale",
                    "label": "pro-female",
                    "polarity": "female",
                    "sample_stem": f"inlp_polarity_pool_{scale}_profemale_v1",
                    "tag_stage_a": f"inlp_{scale}_peak_prepeak_pool_profemale",
                    "tag_stage_b": f"inlp_{scale}_peak_prepeak_pool_profemale_b",
                    "tag_stage_c": f"inlp_{scale}_peak_prepeak_pool_profemale_c",
                    "subspace_tag": f"polarity_pool_{scale}_profemale_v1",
                    "n_soc": len(female),
                    "slugs": female,
                },
            ],
        }
        inlp_path.write_text(
            json.dumps(inlp_doc, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"  wrote {inlp_path.relative_to(REPO)}")


def main() -> int:
    print("=== rebuild H1 SOC-FDR catalog (incl. Gemma) ===")
    catalog = build_catalog()
    catalog = json.loads(json.dumps(catalog))  # NaN → null via default? keep manual
    # jsonable floats already ok from build_domains path — use build_domains main pieces
    from steering.xy_control.build_domains import _jsonable

    catalog = _jsonable(catalog)
    md = catalog_markdown(catalog)
    CATALOG_JSON.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    CATALOG_MD.write_text(md, encoding="utf-8")
    for scale, block in catalog["scales"].items():
        print(
            f"  {scale}: steer {block['n_steer']} "
            f"(male {block['n_male']}, female {block['n_female']})"
        )

    print("=== fill PEAK YAMLs + slot configs ===")
    for scale, meta in PEAKS.items():
        exp = REPO / "experiments" / (
            "gemma3-1b-pt" if scale == "gemma3_1b" else "gemma3-4b-pt"
        )
        g = meta["gender"]["peak"]
        for rel in (
            f"configs/inlp_gender_prob_{scale}_v1.yaml",
            f"configs/inlp_polarity_pool_{scale}_peak_prepeak.yaml",
            f"configs/xy_control_{scale}_peak_prepeak_a6.yaml",
            f"xy_control/configs/xy_control_{scale}_peak_prepeak_a6.yaml",
        ):
            fill_yaml(exp / "steering" / rel, g)
        write_slot_yaml(scale, meta)

    print("=== polarity set JSON ===")
    fill_polarity_sets(catalog)
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
