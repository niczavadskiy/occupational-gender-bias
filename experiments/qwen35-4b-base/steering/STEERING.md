# Steering / INLP — Qwen3.5-4B-Base

Конфиги в `configs/`. Runners — в корне репо `steering/` (`run_inlp_stagea/b/c`, `intervene` с `SubspaceSpec`).
HS run: `results/qwen35_4b_h1h3_pack/run_2026-09-12_20-25-31_Qwen3.5-4B-Base_v1_full_pos_shuffle/`.

**INLP на Vast (A→B→C gender+slot):** см. [`INLP_VAST.md`](INLP_VAST.md).  
Предпочтительно: `scripts/vast_inlp_stage{a,b,c}_both_and_pack.sh` (gender → slot).

**Contrastive mean-diff:** корневой каталог [`steering/contrastive/`](../../../steering/contrastive/README.md) (`--scale 4b`, пояс L20–26, якорь L23).

## Артефакты

| Путь | Содержимое |
|---|---|
| `candidates/h1_*.json`, `slot_*.json` | probe-direction grids |
| `candidates/inlp_*_stageb_shortlist_4b_v1.json` | mirrored auto shortlist (also in Stage A `stageb_shortlist.json`) |
| `candidates/inlp_*_stagec_keep_4b_v1.json` | mirrored auto keep (also in Stage B `stagec_keep.json`) |
| `vectors/*.npz` | probe directions |
| `subspaces/inlp_*_choice_v1.*` | INLP W/centers |
| `samples/` | stage A/B/C frozen samples |
| `scripts/vast_inlp_*` | Vast A/B/C packs |

## Слои (не с 2B)

| Target | Peak (probe) | INLP Stage A grid | B/C shortlist (from A) | Steering belt |
|---|---|---|---|---|
| gender | L24 | 23–25 | **L23** k16/32/64 | 20–26 |
| slot | L30 | 29–31 | **L31** k8/16/32 | 27–32 |

`*_b_v1`/`*_c_v1` = provisional peak (L24/L30). Re-run as **`*_v2`** with ranking shortlists.
