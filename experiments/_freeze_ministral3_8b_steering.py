"""Freeze Ministral 3 8B steering peaks + FDR polarity sets.

Run from occupational-gender-bias root (after H1 metrics + SOC CSV on disk):

  python experiments/_freeze_ministral3_8b_steering.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from steering.xy_control.build_domains import build_catalog, catalog_markdown  # noqa: E402
from steering.xy_control.domains import CATALOG_JSON, CATALOG_MD, SOC_CSV  # noqa: E402

SCALE = "ministral3_8b"
EXP = REPO / "experiments" / "ministral3-8b-base"
MODEL = "mistralai/Ministral-3-8B-Base-2512"
D_MODEL = 4096
N_HS = 35
N_LAYERS = 34

PEAKS = {
    "gender": {
        "peak": 29,
        "layers": [29, 28, 27],
        "source": "gender_choice L29",
    },
    "slot": {
        "peak": 28,
        "layers": [28, 27, 26],
        "source": "slot_choice L28",
    },
}


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {path.relative_to(REPO)}")


def write_gender_yaml() -> None:
    peak = PEAKS["gender"]["peak"]
    layers = PEAKS["gender"]["layers"]
    path = EXP / "steering" / "configs" / f"inlp_gender_prob_{SCALE}_v1.yaml"
    path.write_text(
        f"""# Classical gender_prob INLP · {MODEL}
# Belt = gender_choice probe peak + two layers below.

version: v1_peak_prepeak
hypothesis: inlp_gender_prob
scale: {SCALE}
created: 2026-10-01
status: READY
dtype: bfloat16

source:
  model_id: {MODEL}
  d_model: {D_MODEL}
  n_layers: {N_LAYERS}
  n_hidden_states: {N_HS}
  hidden_state_index: "0 = embeddings, 1..{N_LAYERS} = block output"
  label_note: >-
    Classical INLP on gendered last-token HS (gender_prob). Peak from
    {PEAKS['gender']['source']} (steering PEAK = choice).

hook:
  site: residual_stream_hidden_state
  positions: last_prompt_token
  intervention: "h' = h − α W_k (W_kᵀ h − c)"

target: gender_prob

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
  note: gender_choice peak L{peak}.

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
    description: peak + peak−1 + peak−2 × ranks; Stage A clamps to k_found.
""",
        encoding="utf-8",
    )
    print(f"  wrote {path.relative_to(REPO)}")


def write_slot_yaml() -> None:
    peak = PEAKS["slot"]["peak"]
    layers = PEAKS["slot"]["layers"]
    path = EXP / "steering" / "configs" / f"inlp_slot_prob_{SCALE}_v1.yaml"
    path.write_text(
        f"""# Classical slot_prob INLP · {MODEL}
# Belt = slot_choice probe peak + two layers below.

version: v1_peak_prepeak
hypothesis: inlp_slot_prob
scale: {SCALE}
created: 2026-10-01
status: READY
dtype: bfloat16
target: slot_prob

source:
  model_id: {MODEL}
  d_model: {D_MODEL}
  n_layers: {N_LAYERS}
  n_hidden_states: {N_HS}
  hidden_state_index: "0 = embeddings, 1..{N_LAYERS} = block output"
  label_note: >-
    Classical INLP on slot last-token HS (slot_prob / p_A). Peak from
    {PEAKS['slot']['source']}.

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
""",
        encoding="utf-8",
    )
    print(f"  wrote {path.relative_to(REPO)}")


def write_pool_inlp_yaml() -> None:
    peak = PEAKS["gender"]["peak"]
    layers = PEAKS["gender"]["layers"]
    path = EXP / "steering" / "configs" / f"inlp_polarity_pool_{SCALE}_peak_prepeak.yaml"
    path.write_text(
        f"""# Polarity-pooled INLP · {MODEL}

version: v1_peak_prepeak
hypothesis: inlp_polarity_pool
scale: {SCALE}
created: 2026-10-01
status: READY
dtype: bfloat16
protocol: pooled_polarity_one_subspace

source:
  model_id: {MODEL}
  d_model: {D_MODEL}
  n_layers: {N_LAYERS}
  n_hidden_states: {N_HS}
  hidden_state_index: "0 = embeddings, 1..{N_LAYERS} = block output"
  fit_sample: steering/xy_control/data/xy_pairs_full_v1.json

hook:
  site: residual_stream_hidden_state
  positions: last_prompt_token
  intervention: "h' = h − α W_k (W_kᵀ h − c)"

target: gender_prob

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
  rule: family_split3_by_soc then pool

capability:
  stage_b_profile: steering/profiles/mmlu_pro_domain_val_v1.json
  stage_c_profile: steering/profiles/mmlu_pro_domain_test_v1.json
  cap_loss_max: 0.03

families:
  - id: main_center_peak_prepeak
    enabled: true
    role: candidate
    layers: [{layers[0]}, {layers[1]}, {layers[2]}]
    ranks: [1, 4, 8, 16]
    alphas: [1.0]
""",
        encoding="utf-8",
    )
    print(f"  wrote {path.relative_to(REPO)}")


def write_xy_yaml() -> None:
    peak = PEAKS["gender"]["peak"]
    layers = PEAKS["gender"]["layers"]
    body = f"""# Stage A XY-control polarity pool · {MODEL}
# |α|≤6 grid matched to Qwen/Gemma peak_prepeak_a6.

version: v1_peak_prepeak_a6
hypothesis: xy_control
scale: {SCALE}
created: 2026-10-01
status: READY
dtype: bfloat16

source:
  model_id: {MODEL}
  d_model: {D_MODEL}
  n_layers: {N_LAYERS}
  n_hidden_states: {N_HS}
  hidden_state_index: "0 = embeddings, 1..{N_LAYERS} = block output"
  mapping: {{X: man, "Y": woman}}
  fit_sample: data/xy_pairs_full_v1.json
  eval_sample: data/xy_pairs_full_v1.json
  label: paired_xy_control
  split:
    train_frac: 0.70
    val_frac: 0.15
    test_frac: 0.15
    seed: 0
    stratify_by: soc_major_title

hook:
  site: residual_stream_hidden_state
  positions: last_prompt_token

vectors:
  - id: v_raw
    short: vraw
    role: candidate
    kind: paired_mean_diff

interventions:
  - id: add
    formula: "h' = h − α·v̂"
    alphas: [-6.0, -4.0, -3.0, -2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]

layers:
  belt: [{layers[2]}, {layers[1]}, {layers[0]}]
  core: [{layers[2]}, {layers[1]}, {layers[0]}]
  peak: {peak}
  prepeak: [{layers[1]}, {layers[2]}]
  anchor: {peak}
  note: gender_choice probe peak L{peak}.

families:
  - id: main_add_peak_prepeak
    enabled: true
    role: candidate
    vectors: [v_raw]
    layers: [{layers[0]}, {layers[1]}, {layers[2]}]
    intervention: add
    alphas: [-6.0, -4.0, -3.0, -2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]

baseline:
  id: baseline

screening:
  stage: A
  primary_metric:
    id: gender_gap
    formula: "mean_i Δ_i, Δ_i = mean_layouts (log P(man) − log P(woman))"
    direction: reduce_abs
  keep_top: 10
"""
    for rel in (
        EXP / "steering" / "configs" / f"xy_control_{SCALE}_peak_prepeak_a6.yaml",
        EXP / "steering" / "xy_control" / "configs" / f"xy_control_{SCALE}_peak_prepeak_a6.yaml",
    ):
        rel.parent.mkdir(parents=True, exist_ok=True)
        rel.write_text(body, encoding="utf-8")
        print(f"  wrote {rel.relative_to(REPO)}")


def fill_polarity_sets(catalog: dict) -> tuple[int, int]:
    block = catalog["scales"][SCALE]
    male = [d["slug"] for d in block["steer"] if d["polarity"] == "male"]
    female = [d["slug"] for d in block["steer"] if d["polarity"] == "female"]
    gpeak = PEAKS["gender"]

    xy_doc = {
        "schema": "steering.xy_control_polarity_sets/v1",
        "version": "v2_pooled",
        "scale": SCALE,
        "status": "READY",
        "catalog": "steering/xy_control/domains/h1_soc_fdr_v1.json",
        "config": f"experiments/ministral3-8b-base/steering/xy_control/configs/xy_control_{SCALE}_peak_prepeak_a6.yaml",
        "protocol": {
            "id": "pooled_polarity_one_v",
            "note": "One v_raw per polarity set. Empty sets are skipped at runtime.",
        },
        "probe_peak": {
            "source": gpeak["source"],
            "peak": gpeak["peak"],
            "layers": gpeak["layers"],
            "note": "gender_choice probe peak; also peak−1, peak−2",
        },
        "note": (
            f"FDR-significant {SCALE} SOCs split by H1 polarity. "
            f"male={len(male)} female={len(female)}."
        ),
        "sets": [
            {
                "id": "promale",
                "label": "pro-male",
                "polarity": "male",
                "tag_stage_a": f"xy_{SCALE}_peak_prepeak_a6_pool_promale",
                "tag_stage_b": f"xy_{SCALE}_peak_prepeak_a6_pool_promale_b",
                "tag_stage_c": f"xy_{SCALE}_peak_prepeak_a6_pool_promale_c",
                "n_soc": len(male),
                "slugs": male,
            },
            {
                "id": "profemale",
                "label": "pro-female",
                "polarity": "female",
                "tag_stage_a": f"xy_{SCALE}_peak_prepeak_a6_pool_profemale",
                "tag_stage_b": f"xy_{SCALE}_peak_prepeak_a6_pool_profemale_b",
                "tag_stage_c": f"xy_{SCALE}_peak_prepeak_a6_pool_profemale_c",
                "n_soc": len(female),
                "slugs": female,
            },
        ],
    }
    for p in (
        EXP / "steering" / "domains" / f"polarity_sets_{SCALE}_v1.json",
        EXP / "steering" / "xy_control" / "domains" / f"polarity_sets_{SCALE}_v1.json",
    ):
        _write_json(p, xy_doc)

    inlp_doc = {
        "schema": "steering.inlp_polarity_sets/v1",
        "version": "v1_pooled",
        "scale": SCALE,
        "status": "READY",
        "catalog": "steering/xy_control/domains/h1_soc_fdr_v1.json",
        "xy_sets": f"experiments/ministral3-8b-base/steering/domains/polarity_sets_{SCALE}_v1.json",
        "config": f"experiments/ministral3-8b-base/steering/configs/inlp_polarity_pool_{SCALE}_peak_prepeak.yaml",
        "xy_dataset": "steering/xy_control/data/xy_pairs_full_v1.json",
        "protocol": {
            "id": "pooled_polarity_one_subspace",
            "note": "One INLP subspace W per polarity set. Empty sets skipped.",
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
            f"FDR-significant {SCALE} SOCs. Matched arm to XY polarity pool. "
            "Gender only (no slot in pool protocols)."
        ),
        "sets": [
            {
                "id": "promale",
                "label": "pro-male",
                "polarity": "male",
                "sample_stem": f"inlp_polarity_pool_{SCALE}_promale_v1",
                "tag_stage_a": f"inlp_{SCALE}_peak_prepeak_pool_promale",
                "tag_stage_b": f"inlp_{SCALE}_peak_prepeak_pool_promale_b",
                "tag_stage_c": f"inlp_{SCALE}_peak_prepeak_pool_promale_c",
                "subspace_tag": f"polarity_pool_{SCALE}_promale_v1",
                "n_soc": len(male),
                "slugs": male,
            },
            {
                "id": "profemale",
                "label": "pro-female",
                "polarity": "female",
                "sample_stem": f"inlp_polarity_pool_{SCALE}_profemale_v1",
                "tag_stage_a": f"inlp_{SCALE}_peak_prepeak_pool_profemale",
                "tag_stage_b": f"inlp_{SCALE}_peak_prepeak_pool_profemale_b",
                "tag_stage_c": f"inlp_{SCALE}_peak_prepeak_pool_profemale_c",
                "subspace_tag": f"polarity_pool_{SCALE}_profemale_v1",
                "n_soc": len(female),
                "slugs": female,
            },
        ],
    }
    _write_json(EXP / "steering" / "domains" / f"inlp_polarity_sets_{SCALE}_v1.json", inlp_doc)
    return len(male), len(female)


def write_pipeline(n_male: int, n_female: int) -> None:
    g, s = PEAKS["gender"], PEAKS["slot"]
    path = EXP / "steering" / "PIPELINE.md"
    path.write_text(
        f"""# Ministral 3 8B Base — steering pipeline (after beh + probe)

Scale key: **`{SCALE}`** · model `{MODEL}` · **{N_LAYERS}** layers · **d_model={D_MODEL}**.

Dtype: **`bfloat16`** (default in Vast scripts).

**Results (canonical):** [`../results/steering/`](../results/steering/) — see [`LAYOUT.md`](../results/steering/LAYOUT.md).

## Locked peaks (classification)

| Axis | Probe | Peak | Belt |
|------|-------|-----:|------|
| Gender | `gender_choice` | **{g['peak']}** | {' / '.join(str(x) for x in g['layers'])} |
| Slot | `slot_choice` | **{s['peak']}** | {' / '.join(str(x) for x in s['layers'])} |

Narrative omitted. Soft `gender_prob` L21 not used (steering PEAK = choice L{g['peak']}).

## FDR polarity (gender pool protocols)

| set | n SOC | action |
|-----|------:|--------|
| `promale` | **{n_male}** | {"run" if n_male else "**skip**"} |
| `profemale` | **{n_female}** | {"run" if n_female else "**skip**"} |

Empty sets are skipped by pool scripts.

## Protocols (full A→B→C)

| # | Protocol | Axes | Vast |
|---|----------|------|------|
| 1 | Classical INLP | **gender + slot** | [`scripts/vast_classical_inlp_abc_full_instance.sh`](scripts/vast_classical_inlp_abc_full_instance.sh) |
| 2 | Pool INLP | gender only | [`scripts/vast_inlp_polarity_pool_ministral3_3b_full_instance.sh`](scripts/vast_inlp_polarity_pool_ministral3_3b_full_instance.sh) |
| 3 | Pool XY \\|α\\|≤6 | gender only | [`scripts/vast_xy_control_polarity_sets_ministral3_3b_full_instance.sh`](scripts/vast_xy_control_polarity_sets_ministral3_3b_full_instance.sh) |

## Vast commands

```bash
export HF_TOKEN=hf_xxx
SCALE=ministral3_3b DTYPE=bfloat16 \\
  LAYERS_GENDER={",".join(str(x) for x in g["layers"])} \\
  LAYERS_SLOT={",".join(str(x) for x in s["layers"])} \\
  bash experiments/ministral3-8b-base/steering/scripts/vast_classical_inlp_abc_full_instance.sh

SCALE=ministral3_3b DTYPE=bfloat16 \\
  bash experiments/ministral3-8b-base/steering/scripts/vast_inlp_polarity_pool_ministral3_3b_full_instance.sh

SCALE=ministral3_3b DTYPE=bfloat16 \\
  bash experiments/ministral3-8b-base/steering/scripts/vast_xy_control_polarity_sets_ministral3_3b_full_instance.sh

SMOKE=1 SCALE=ministral3_3b DTYPE=bfloat16 \\
  LAYERS_GENDER={g["peak"]} LAYERS_SLOT={s["peak"]} \\
  bash experiments/ministral3-8b-base/steering/scripts/vast_classical_inlp_abc_full_instance.sh
```

## Configs (READY)

| File | Role |
|------|------|
| [`configs/inlp_gender_prob_ministral3_3b_v1.yaml`](configs/inlp_gender_prob_ministral3_3b_v1.yaml) | classical gender |
| [`configs/inlp_slot_prob_ministral3_3b_v1.yaml`](configs/inlp_slot_prob_ministral3_3b_v1.yaml) | classical slot |
| [`configs/inlp_polarity_pool_ministral3_3b_peak_prepeak.yaml`](configs/inlp_polarity_pool_ministral3_3b_peak_prepeak.yaml) | pool INLP |
| [`configs/xy_control_ministral3_3b_peak_prepeak_a6.yaml`](configs/xy_control_ministral3_3b_peak_prepeak_a6.yaml) | pool XY |
| [`domains/polarity_sets_ministral3_3b_v1.json`](domains/polarity_sets_ministral3_3b_v1.json) | XY sets |
| [`domains/inlp_polarity_sets_ministral3_3b_v1.json`](domains/inlp_polarity_sets_ministral3_3b_v1.json) | INLP sets |

Catalog: `steering/xy_control/domains/h1_soc_fdr_v1.json` → `scales.ministral3_3b`.
""",
        encoding="utf-8",
    )
    print(f"  wrote {path.relative_to(REPO)}")


def main() -> int:
    csv = SOC_CSV[SCALE]
    if not csv.is_file():
        raise SystemExit(f"missing H1 SOC CSV: {csv}\nRun H1 metrics first.")
    print(f"=== rebuild FDR catalog (incl. {SCALE}) ===")
    catalog = build_catalog()
    if SCALE not in catalog["scales"]:
        raise SystemExit(f"{SCALE} missing from catalog after build")
    CATALOG_JSON.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    CATALOG_MD.write_text(catalog_markdown(catalog), encoding="utf-8")
    print(f"  wrote {CATALOG_JSON.relative_to(REPO)}")
    block = catalog["scales"][SCALE]
    print(
        f"  {SCALE}: steer={block['n_steer']} "
        f"(male={block['n_male']} female={block['n_female']}) skip={block['n_skip']}"
    )

    print("=== configs ===")
    write_gender_yaml()
    write_slot_yaml()
    write_pool_inlp_yaml()
    write_xy_yaml()

    print("=== polarity sets ===")
    n_male, n_female = fill_polarity_sets(catalog)

    print("=== PIPELINE ===")
    write_pipeline(n_male, n_female)

    # classical vast defaults for ministral3_8b branch
    script = EXP / "steering" / "scripts" / "vast_classical_inlp_abc_full_instance.sh"
    text = script.read_text(encoding="utf-8")
    text2 = text.replace(
        'LAYERS_GENDER="${LAYERS_GENDER:-24,23,22}"',
        'LAYERS_GENDER="${LAYERS_GENDER:-29,28,27}"',
    ).replace(
        'LAYERS_SLOT="${LAYERS_SLOT:-31,30,29}"',
        'LAYERS_SLOT="${LAYERS_SLOT:-28,27,26}"',
    )
    if text2 != text:
        script.write_text(text2, encoding="utf-8")
        print(f"  updated classical LAYERS defaults in {script.relative_to(REPO)}")

    print("DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
