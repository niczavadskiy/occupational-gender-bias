# Steering / INLP configs — Qwen3.5-4B-Base

Конфиги в `configs/`. Код runners — в `repo/steering/` (не дублировать).
HS/probes run: `results/qwen35_4b_h1h3_pack/run_2026-09-12_20-25-31_Qwen3.5-4B-Base_v1_full_pos_shuffle/`.

## Артефакты (собрано локально)

| Путь | Содержимое |
|---|---|
| `candidates/h1_candidates_v1.json` | 105 кандидатов (94 + 11 control) |
| `candidates/slot_candidates_v1.json` | 88 кандидатов |
| `vectors/h1_vectors_v1.npz` | слои L20–26 |
| `vectors/slot_vectors_v1.npz` | слои L22–25 + L27–32 |
| `subspaces/inlp_*_choice_v1.*` | INLP (после `build_artifacts.sh`) |
| `samples/h1_stagea_sample_v1.json` | тот же val sample, что 2B |
| `scripts/` | rebuild + Vast Stage A |

Пересборка:
```bash
bash experiments/qwen35-4b-base/steering/scripts/build_artifacts.sh
# или SKIP_INLP=1 для только candidates/vectors
```

Vast Stage A (из инстанса с `REPO` = inference repo):
```bash
export STEER=.../experiments/qwen35-4b-base/steering
export REPO=/workspace/Bias--subspaces-in-LLM
bash "$STEER/scripts/vast_h1_stagea_and_pack.sh"
bash "$STEER/scripts/vast_slot_stagea_and_pack.sh"
bash "$STEER/scripts/vast_inlp_gender_stagea_and_pack.sh"
```

## Почему не копируем слои с 2B

На **2B** gender пик ~**L16**, slot ~**L23**. На **4B** (тот же протокол layer_scan):

| Target | Peak (val bacc) | Plateau | 2B peak (для сравнения) |
|---|---|---|---|
| `gender_choice` | **L24** (.875) | L20–L26 | L16 |
| `slot_choice` | **L30** (.960) | L27–L32 | L23 |
| `narrative_choice` | L26 | — | — |

Глубина сигнала сдвигается к поздним блокам (32 слоя vs 28 у 2B, d=2560 vs 2048).

## Файлы

| Config | Роль |
|---|---|
| `h1_steering_candidates_v1.yaml` | Probe-direction Stage A: gender, пояс **L20–26**, якорь **24** |
| `slot_steering_candidates_v1.yaml` | Slot, пояс **L27–32**, якорь **30**; `mid` = L22–25 |
| `inlp_gender_v1.yaml` | INLP gender, слои **[23,24,25]** |
| `inlp_slot_v1.yaml` | INLP slot, слои **[29,30,31]** |

Протокол семейств и Stage A sample seed **20260809** сохранены с 2B; меняются слои и пути к run.

## Ключевые решения

1. **Якоря = peak val_bacc** (L24 / L30).
2. **INLP candidates = peak ±1**.
3. **C=0.03 стартовый**; на d=2560 желателен C-sweep до финальных заявлений.
4. **k_max=64**.
5. **`--results-root`** на pack-директорию.
