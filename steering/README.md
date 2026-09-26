# steering/

Каузальные интервенции на gender-направлении и контроль побочного ущерба.

Проба (`probes/v1_rep_prob`) показывает, что пол выбранного ответа **линейно
декодируется** из hidden states (L16, test bacc ≈ 87%). Steering проверяет
следующий шаг: **используется** ли это направление при решении, и можно ли
выровнять предпочтение до `P(man) ≈ P(woman)` в каждом домене профессии, не
ломая знания модели.

Точка интервенции совпадает с точкой снятия HS: residual stream, **last prompt
token** (после `Answer:`). Тот же хук применим к MCQ-протоколу MMLU-Pro, где
ответ читается с логитов последнего токена промпта.

---

## Что уже здесь

```
steering/
├── configs/
│   ├── soc_mmlu_pro_map_v1.yaml           # soc_major_title ↔ MMLU-Pro
│   ├── h1_steering_candidates_v1.yaml     # сетка layer × α для H1 (gender)
│   ├── slot_steering_candidates_v1.yaml   # сетка layer × α для slot A/B
│   ├── inlp_gender_prob_v1.yaml           # classical INLP (global)
│   └── inlp_polarity_pool_2b_peak_prepeak.yaml  # polarity-pooled INLP (peak+2)
├── candidates/
│   ├── h1_candidates_v1.json              # H1: замороженный список
│   ├── h1_candidates_v1.md
│   ├── slot_candidates_v1.json            # slot: 87 кандидатов + baseline
│   └── slot_candidates_v1.md
├── samples/
│   ├── h1_stagea_sample_v1.json           # Stage A: 95 val-семей × 4 строки
│   └── h1_stagea_sample_v1.md             # (тот же sample для slot)
├── vectors/
│   ├── h1_vectors_v1.npz                  # gender ŵ на L14–20
│   ├── h1_vectors_v1.json
│   ├── slot_vectors_v1.npz                # slot ŵ на L16–24 (54 вектора)
│   └── slot_vectors_v1.json
├── profiles/
│   ├── mmlu_pro_domain_val_v1.json        # Stage B: 22 домена × 50 вопросов
│   ├── mmlu_pro_domain_test_v1.json       # Stage C: те же домены, другие вопросы
│   ├── mmlu_pro_overall_smoke_v1.json     # Stage B: 14 категорий × 40 вопросов
│   └── coverage_v1.md
├── intervene.py                           # хук в residual stream + A/B скоринг
├── contrastive/                           # mean-diff v = μ_M − μ_F, center steering
│   ├── configs/                           # 2B L14–20 / 4B L20–26
│   ├── candidates/                        # замороженный каталог
│   ├── build_vectors.py                   # live HS → v̂, c (+ capture)
│   ├── analyze_rank.py                    # per-SOC PCA / V_k
│   ├── run_stagea.py                      # обёртка H1 Stage A
│   └── run_pca_stagea.py                  # bake-off rank-k vs v_G
├── xy_control/                            # paired XY-control: v = mean(h_gender − h_XY)
│   ├── data/                              # X=man, Y=woman paired prompts
│   ├── mapping.py
│   ├── build_dataset.py
│   ├── build_vectors.py                   # paired last-token HS → v̂_raw
│   └── run_stagea.py                      # add: h' = h − α v̂
├── run_h1_stagea.py                       # Stage A: gender (θ)
├── run_slot_stagea.py                     # Stage A: slot (φ)
├── check_probe_geometry.py                # Эксп.0: scaler → raw hyperplane
├── run_alignment_recovery.py              # Эксп.1–3: cos(w,g), Δd, recovery
├── run_hs_recovery_auc.py                 # Method 3: AUC/HS after single-layer erase
├── run_second_hit_screen.py               # Sequential: L15 + second site screen
├── build_conditional_inlp_subspace.py     # INLP on HS under S0
├── run_stacked_inlp_stagea.py             # Stage A with S0 always on
├── ascent_pool.py                         # AUC ascent pool detector
├── run_sequential_pre_peak_erase.py       # Pre-peak multi-site erase (gender/slot)
├── scripts/
│   ├── vast_hs_recovery_auc_and_pack.sh   # runner + tar pack
│   ├── vast_hs_recovery_full_instance.sh  # fresh Vast: clone → smoke → full → pack
│   ├── vast_second_hit_and_pack.sh
│   ├── vast_conditional_inlp_stagea_and_pack.sh
│   ├── vast_sequential_erase_full_instance.sh
│   └── vast_sequential_pre_peak_and_pack.sh
├── HS_RECOVERY_VAST.md                    # инструкция Vast
├── SEQUENTIAL_ERASE.md                    # multi-site erase protocol (post-hook)
├── SEQUENTIAL_PRE_PEAK_ERASE.md           # ascent-pool first-hit protocol
├── build_stagea_sample.py
├── build_h1_vectors.py                    # --config / --out-prefix → H1 или slot
├── build_mmlu_profiles.py
├── build_h1_candidates.py
└── .cache/                                # parquet MMLU-Pro, веса модели (не в git)
```

Результаты прогонов — вне `steering/`:

```
results/steering/stage_a/<tag>/
├── run_meta.json                       # модель, dtype, sha256 векторов, состав
├── ranking.csv                         # строка на конфигурацию
└── <config_id>/
    ├── per_item.jsonl                  # 380 строк: логиты, choice, s_before/s_after
    └── metrics.json                    # θ, per_soc, guardrail-rates, hook_check
```

---

## Contrastive (mean-diff) steering

Отдельный каталог: [`contrastive/README.md`](contrastive/README.md).

Направление \(v=\mathrm{mean}(h\mid y=M)-\mathrm{mean}(h\mid y=F)\) по **live**
choice модели; интервенция — H1 `center` \(h'=h-\alpha(s-c)\hat v\). Сетка 46
кандидатов (2B и 4B), Stage A на том же sample, что H1.

```powershell
python -m steering.contrastive.build_candidates
python -m steering.contrastive.build_vectors --device cuda
python -m steering.contrastive.analyze_rank --device cuda
python -m steering.contrastive.run_stagea --device cuda --tag contrastive_2b_a_v1
```

Vast (2B): [`contrastive/CONTRASTIVE_VAST.md`](contrastive/CONTRASTIVE_VAST.md), ноутбук [`notebooks/vast_contrastive_2b.ipynb`](../notebooks/vast_contrastive_2b.ipynb).

```bash
export HF_TOKEN=hf_xxx
bash steering/contrastive/scripts/vast_contrastive_full_instance.sh
# только спектр D / V_k (без сетки 46):
bash steering/contrastive/scripts/vast_contrastive_rank_and_pack.sh
```

---

## XY-control (gender-conditioned activation direction)

Отдельный каталог: [`xy_control/README.md`](xy_control/README.md).

Парный контраст промптов, маппинг **X = man, Y = woman**:

\[
v_{\mathrm{raw}}=\mathrm{mean}_i\bigl(h^{\mathrm{gender}}_i-h^{\mathrm{XY}}_i\bigr)
\]

Интервенция `add`: \(h'=h-\alpha\hat v\) на last prompt token. Не путать с
contrastive \(v=\mu_M-\mu_F\) по live-choice.

```powershell
python -m steering.xy_control.build_dataset
python -m steering.xy_control.build_candidates
python -m steering.xy_control.build_vectors --device cuda
python -m steering.xy_control.run_stagea --device cuda --tag xy_2b_a_v1
```

---

## Polarity-pooled INLP (matched to XY pool)

Протокол: [`POLARITY_POOL_INLP.md`](POLARITY_POOL_INLP.md).

Один INLP-subspace на pro-male / pro-female FDR set; кандидаты **L16 (peak) + L15 + L14**;
тот же family split, что у XY polarity pool.

```powershell
python -m steering.build_polarity_pool_inlp_sample --all
python -m steering.build_polarity_pool_inlp_subspace --set-id promale --device cuda
python -m steering.run_polarity_pool_inlp_stagea --set-id promale --device cuda
```

```bash
export HF_TOKEN=hf_xxx
bash steering/scripts/vast_inlp_polarity_pool_full_instance.sh
```

---

## Sequential multi-site erase

- **Post-hook** (после keep L15): [`SEQUENTIAL_ERASE.md`](SEQUENTIAL_ERASE.md)  
- **Pre-peak** (пул по форме AUC ≤ пика, старт с earliest): [`SEQUENTIAL_PRE_PEAK_ERASE.md`](SEQUENTIAL_PRE_PEAK_ERASE.md)

```bash
# plan only
python -m steering.run_sequential_pre_peak_erase \
  --config steering/configs/sequential_pre_peak_erase_gender_2b_v1.yaml --plan-only

# Vast
bash steering/scripts/vast_sequential_pre_peak_and_pack.sh
```

## Method 3 — HS / AUC recovery (после erase на одном слое)

Интервенция на одном слое (probe center или INLP k), затем last-token HS на
поясе слоёв ниже. На каждом ℓ′: ROC-AUC `gender_choice` (y из frozen
`baseline_choice` в sample) для baseline vs steered.

- **frozen ŵ** — если есть `w_gender_perp__Lℓ` в vector bank  
- **mean_diff** — направление `mean(h|y=1)−mean(h|y=0)` fit только на baseline
  train-семьях; тем же w скорим test baseline и steered (без refit)

```powershell
# 2B keep: INLP L15 k16
python -m steering.run_hs_recovery_auc --model Qwen/Qwen3.5-2B-Base `
  --device cuda --dtype float32 --mode inlp --intervene-layer 15 --rank 16 `
  --subspaces steering/subspaces/inlp_gender_choice_v1.npz `
  --read-layers 15,16,17,18,19,20,21,22,23,24 --tag hs_rec_l15k16

# probe center (нужен h1_vectors_v1.npz)
python -m steering.run_hs_recovery_auc --model Qwen/Qwen3.5-2B-Base `
  --device cuda --dtype float32 --mode probe --intervene-layer 15 `
  --vector-id w_gender_perp --alpha 1 --tag hs_rec_probe_l15
```

Выход: `results/steering/hs_recovery/<tag>/summary.json`, `by_layer.csv`,
`per_row_pref.jsonl`. Recovery ≈ AUC на intervene-слое падает, на более глубоких
снова растёт к baseline.

---

## H1 candidates (`layer × α`)

Конфиг: `configs/h1_steering_candidates_v1.yaml` → заморозка
`candidates/h1_candidates_v1.json`.

Основное семейство — **center**: `h' = h − α(s − c)ŵ`, где `s = h·ŵ`,
`c` — mid-point классов на train. Одна положительная α тянет к нейтрали и
«мужские», и «женские» домены (знак коррекции в `(s − c)`). Отрицательная α —
семейство `anti_steering` (контроль каузальности).

| семейство | n | роль |
|---|---:|---|
| `main_center_core` (L15–18, α до 2.0) | 56 | candidate |
| `main_center_edge` (L14,19,20) | 24 | candidate |
| `main_project_out` | 14 | candidate |
| `causality_shift` / `anti_steering` / other / random | 11 | control |

```powershell
python -m steering.build_h1_candidates
python -m steering.build_h1_candidates --verify
```

---

## Slot candidates (`layer × α`)

Источник: v1_rep `slot_choice` (пик L23, val bacc ≈ 94.5%). Цель — выровнять
preference слота: `φ_i = A_i/(A_i+B_i) → 0.5` на base item. Baseline Stage A
(H1 full): `slot_a_rate ≈ 0.76`.

Конфиг: `configs/slot_steering_candidates_v1.yaml` → `candidates/slot_candidates_v1.json`
(87 + baseline). Пояс **L19–24**, якорь **L23**. Основной вектор
`w_slot_perp` (slot ⊥ {gender, narrative}).

| семейство | n | роль |
|---|---:|---|
| `main_center_core` (L21–23, α до 2.0) | 42 | candidate |
| `main_center_edge` (L19,20,24) | 18 | candidate |
| `main_project_out` (L19–24) | 12 | candidate |
| `mid_project_out` (L16–18) | 3 | candidate |
| `causality_shift` / `anti_steering` / gender / random | 12 | control |

Primary metric Stage A: `mean_i |φ_i − 0.5|` (и soft-версия по `p_A`). Gender
`mean|θ−0.5|` — guardrail (flag).

```powershell
python -m steering.build_h1_candidates --config steering/configs/slot_steering_candidates_v1.yaml
python -m steering.build_h1_vectors --config steering/configs/slot_steering_candidates_v1.yaml
# (векторы требуют results/<run>/hidden_states.npz — только в полном репо)
```

Пересборка пишет `slot_candidates_v1.*` / `slot_vectors_v1.*` (префикс из
`hypothesis: slot` в yaml).

---

## Протокол эксперимента

Направление `ŵ` уже обучено на **train**-группах probe-split
(`probes/_shared/group_split_v1_scenario.json`, 60/20/20 по
`soc × profession × onet_action`). Чтобы отбор конфигураций не подгонялся под те
же сценарии, все стадии работают вне probe-train.

| стадия | данные | что меряем | что оставляем |
|---|---|---|---|
| **A** screen | ~10% base items из **val**, стратифицировано по `soc_major_title` | preference после steering | 10–20 конфигураций `(layer, α, ŵ, тип интервенции)` |
| **B** select | остаток **val** + `mmlu_pro_domain_val_v1` + `overall_smoke` | preference + доменный capability | 1–3 кандидата |
| **C** report | **test**-группы + `mmlu_pro_domain_test_v1` + полный MMLU-Pro | подтверждение | финальные числа отчёта |

Ранжирование на A — по нейтральности **после** интервенции
(`mean_i |θ_i − 0.5|`, где `θ_i = M_i/(M_i+W_i)` на base item), а не по разнице
«до минус после»: разность около нуля означает отсутствие эффекта.

Capability на B — односторонний штраф, рост не поощряется:

```
cap_loss = mean over domains of max(0, acc_baseline − acc_steered)
```

На C победитель Stage B не переизбирается — только подтверждается; критерий
capability формулируется как non-inferiority с заранее заданной δ.

---

## Эксп.0 — геометрия пробы (CPU)

`StandardScaler` стоит перед LogReg, поэтому `coef_` не является steering-вектором.
Проверка тождества в raw HS (модель не нужна, только `hidden_states.npz` полного
репо с `probes/`):

```powershell
python -m steering.check_probe_geometry
# default: gender_choice + slot_choice на L16 и L23
```

Пишет `results/steering/geometry_check/exp0/geometry_check.json`.
Pass: `|decision_function(H) − (H w_raw + b_raw)|_max ≤ 1e-4` и cos с замороженным
`ŵ` ≥ 0.999 (если npz векторов на месте). `c` (class midpoint) и `t_probe`
печатаются рядом: steering center использует **c**, не порог пробы.

---

## Эксп.1–3 — alignment, Δd, recovery (GPU)

На 25 семей × 4 строки (= 100) из того же Stage A sample, слои L16 и L23,
float32. Считает `cos(g, ŵ)` для `d_gender = z_man − z_woman` и `d_slot = z_A − z_B`,
сравнивает `gᵀΔh` с фактическим Δd после shift ±1 / center α=1, и смотрит, зарастает
ли проекция на слоях ниже точки интервенции.

```powershell
python -m steering.run_alignment_recovery --model Qwen/Qwen3.5-2B-Base `
  --device cuda --dtype float32 --tag mech_v1

# 4 строки
python -m steering.run_alignment_recovery --model Qwen/Qwen3.5-2B-Base `
  --device cuda --dtype float32 --n-items 1 --tag mech_smoke
```

Выход: `results/steering/mech/<tag>/summary.json`, `alignment_summary.csv`,
`delta_summary.csv`, `per_row.jsonl`.

---

## Как запускать Stage A

```powershell
# 1. Заморозить выборку (95 из 190 val-семей, стратификация по soc_major_title)
python -m steering.build_stagea_sample
python -m steering.build_stagea_sample --verify

# 2. Материализовать направления: ŵ, c, sigma_train на слоях пояса
python -m steering.build_h1_vectors          # нужен hidden_states.npz исходного прогона
python -m steering.build_h1_vectors --verify # пересборка без изменений

# 3. Полный скрининг H1: 106 конфигураций × 380 строк
python -m steering.run_h1_stagea --model Qwen/Qwen3.5-2B-Base --device cuda --tag full

# 3b. Slot Stage A: 88 конфигураций × 380 строк (~33k forward)
python -m steering.run_slot_stagea --model Qwen/Qwen3.5-2B-Base --device cuda `
  --dtype float32 --tag slot_full
```

`c` считается только на **train**-строках пробы, поэтому выбор нейтральной точки
не подсматривает в val, на котором ранжируются кандидаты.

### dtype: почему float32

Возмущение мало относительно нормы состояния: `||h|| ≈ 11.9`, `|s − c| ~ 0.05`,
то есть `|Δ|` на компоненту ≈ `9e-4` при среднем `|h_j| ≈ 0.18`. Шаг округления
bf16 на этой величине — того же порядка, и часть интервенции просто теряется при
записи. Дефолт раннера — `float32`; `metrics.json` содержит `hook_check` с
фактической ошибкой относительно аналитической формулы (на bf16 — 9% для
`center α=1`, на float32 — численный ноль).

### Smoke-прогон

```powershell
# H1
python -m steering.run_h1_stagea --model steering/.cache/model --device cpu `
  --dtype bfloat16 --limit-items 3 --tag smoke `
  --candidates main_center_core__wgperp__L16__center__a1,causality_shift__wgperp__L16__shift__b1

# Slot (L23: project_out + causality shift)
python -m steering.run_slot_stagea --model Qwen/Qwen3.5-2B-Base --device cuda `
  --dtype float32 --limit-items 3 --tag slot_smoke `
  --candidates main_project_out__wsperp__L23__project_out__a1,causality_shift__wsperp__L23__shift__b1,causality_shift__wsperp__L23__shift__bm1
```

Что проверено H1-smoke (`results/steering/stage_a/smoke/`):

| проверка | результат |
|---|---|
| baseline воспроизводит записанный прогон | `choice` совпал 12/12 (`agreement_with_recorded_run = 1.0`) |
| хук попадает в нужный слой | `s_before` совпадает с `hidden_states[16]`, не с 15/17 |
| формула применяется | `shift β=1` даёт `Δs = 0.0425` при `sigma_train = 0.0434` |
| bf16 съедает часть шага | ошибка 9% для `center`, 2% для `shift` — отсюда дефолт float32 |

Скорость на CPU — ~0.15 строк/с, то есть полный скрининг (≈40 тыс. forward)
осмысленно гонять только на GPU.

---

## Маппинг SOC ↔ MMLU-Pro

MMLU-Pro покрывает академические дисциплины, а SOC — профессии, поэтому
честного маппинга «один в один» не существует. Вместо того чтобы выбрасывать
непокрытые домены, конфиг тестирует **все 22** и делает качество прокси явным
атрибутом (`coverage_v1.md`):

| тир | доменов | что за вопросы | как читать просадку |
|---|---:|---|---|
| `primary` | 11 | профильные категории, confidence high/medium | доменное свидетельство |
| `secondary` | 8 | профильные категории, confidence low | «хуже отвечает на смежную дисциплину» |
| `generic` | 3 | срез всего MMLU-Pro пропорционально размеру категорий | общая деградация, не доменная |

Тир `generic` — компромисс для Transportation, Food Preparation и Building and
Grounds: академического аналога у них нет, но раз доменное знание проверить
нечем, проверяем хотя бы, что интервенция не ломает модель вообще. Их просадка
коррелирует с overall-capability и не должна интерпретироваться как «steering
испортил знания о вождении».

Отдельная ось — надёжность страты preference: у Legal (`G=8`), Community and
Social Service (`G=14`) и Building and Grounds (`G=8`) capability измеряется
полноценно, но пару «preference ↔ capability» по ним строить нельзя. Это
отмечено полем `preference_stratum: small`, а не исключением из capability.

Headline `cap_loss` считается по всем 22 доменам; отчёт дополнительно приводит
сабсредние по тирам, иначе непонятно, где просадка доменная, а где общая.

### Почему equal-n, а не пропорционально

Capability-score — macro-average по доменам, и каждый домен должен иметь равный
голос. При пропорциональной выборке кандидат, «уронивший» маленький домен,
выигрывал бы просто потому, что вопросов там мало. Поэтому `n_per_soc = 50`
одинаково для всех 22 доменов.

Пропорциональность применяется только **внутри** generic-доменов: там доли
категорий равны их доле в split=test, то есть выборка повторяет состав
бенчмарка целиком.

Домены с общим пулом категорий (Business и Management — оба `business` +
`economics`; Construction и Installation — оба `engineering`) получают
**непересекающиеся** вопросы: оценки остаются независимыми, а не дублируют
друг друга. Итого 2200 вопросов из 12032, самая загруженная категория —
`engineering` (334 из 969).

---

## Пересборка профилей

Профили заморожены и коммитятся — Stage B и Stage C должны гоняться на
фиксированных `question_id`. Пересборка нужна только при bump'е версии конфига.

```powershell
# 1. Скачать split=test датасета (в git не попадает)
curl.exe -L -o "steering/.cache/mmlu_pro_test.parquet" `
  "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet"

# 2. Собрать
python -m steering.build_mmlu_profiles

# 3. Проверить, что файлы на диске совпадают с пересборкой
python -m steering.build_mmlu_profiles --verify
```

Выбор вопросов зависит только от `master_seed`, названия домена и категории,
поэтому повторный запуск даёт побайтово идентичные файлы. В каждом профиле
записаны `map_sha256` и `parquet_sha256` — если конфиг или бенчмарк изменились,
это видно из результатов прогона.

Правило заморозки действует **с первого прогона Stage A**: пока ни один
эксперимент не отработал, конфиг остаётся черновиком и правится на месте с
записью в `changelog`. После первого прогона любое изменение маппинга, весов
или `n_per_soc` — новая версия (`v2`) и новые профили, иначе прошлые прогоны
Stage B/C перестают быть сравнимыми.
