# Steering / INLP — Qwen3.5-4B-Base

Конфиги в `configs/`. Runners — в корне репо `steering/` (`run_inlp_stagea/b/c`, `intervene` с `SubspaceSpec`).
HS run: `results/qwen35_4b_h1h3_pack/run_2026-09-12_20-25-31_Qwen3.5-4B-Base_v1_full_pos_shuffle/`.

**INLP на Vast (A→B→C gender+slot):** см. [`INLP_VAST.md`](INLP_VAST.md).

## Артефакты

| Путь | Содержимое |
|---|---|
| `candidates/h1_*.json`, `slot_*.json` | probe-direction grids |
| `candidates/inlp_*_stageb_shortlist_4b_v1.json` | provisional Stage B (edit after A) |
| `candidates/inlp_*_stagec_keep_4b_v1.json` | provisional Stage C keep |
| `vectors/*.npz` | probe directions |
| `subspaces/inlp_*_choice_v1.*` | INLP W/centers |
| `samples/` | stage A/B/C frozen samples |
| `scripts/vast_inlp_*` | Vast A/B/C packs |

## Слои (не с 2B)

| Target | Peak | INLP layers | Steering belt |
|---|---|---|---|
| gender | L24 | 23–25 | 20–26 |
| slot | L30 | 29–31 | 27–32 |
