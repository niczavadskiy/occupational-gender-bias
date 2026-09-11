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
│   └── inlp_gender_v1.yaml                # INLP + rank-k center (gender)
├── candidates/
│   ├── h1_candidates_v1.json              # H1: замороженный список
│   ├── h1_candidates_v1.md
│   ├── slot_candidates_v1.json            # slot: 87 кандидатов + baseline
│   ├── slot_candidates_v1.md
│   ├── slot_stageb_shortlist_v1.json      # Stage B: keep-top + controls
│   ├── inlp_stageb_shortlist_v1.json      # INLP Stage B: L15 k=8/16 + random
│   └── inlp_stagec_keep_v1.json           # INLP Stage C: report keep (no re-select)
├── samples/
│   ├── h1_stagea_sample_v1.json           # Stage A: 95 val-семей × 4 строки
│   ├── h1_stagea_sample_v1.md             # (тот же sample для slot)
│   ├── h1_stageb_sample_v1.json           # Stage B: остаток val (95 семей)
│   └── h1_stageb_sample_v1.md
├── vectors/
│   ├── h1_vectors_v1.npz                  # gender ŵ на L14–20
│   ├── h1_vectors_v1.json
│   ├── slot_vectors_v1.npz                # slot ŵ на L16–24 (54 вектора)
│   └── slot_vectors_v1.json
├── subspaces/
│   ├── inlp_gender_choice_v1.npz          # W [2048,k], centers, AUC(k), random control
│   ├── inlp_gender_choice_v1.json         # схема итераций, k_chance, диагностика
│   └── inlp_gender_choice_v1.md
├── profiles/
│   ├── mmlu_pro_domain_val_v1.json        # Stage B: 22 домена × 50 вопросов
│   ├── mmlu_pro_domain_test_v1.json       # Stage C: те же домены, другие вопросы
│   ├── mmlu_pro_overall_smoke_v1.json     # Stage B: 14 категорий × 40 вопросов
│   └── coverage_v1.md
├── intervene.py                           # хук: rank-1 + rank-k; Scorer A–J
├── mmlu_eval.py                           # промпт/скоринг MMLU-Pro под тем же хуком
├── run_h1_stagea.py                       # Stage A: gender (θ)
├── run_slot_stagea.py                     # Stage A: slot (φ)
├── run_slot_stageb.py                     # Stage B: slot preference + cap_loss
├── run_slot_stagec.py                     # Stage C: slot report (test + domain_test)
├── run_inlp_stageb.py                     # Stage B: INLP MMLU cap_loss (+opt preference)
├── run_inlp_stagec.py                     # Stage C: report on test remainder + domain_test
├── check_probe_geometry.py                # Эксп.0: scaler → raw hyperplane
├── run_alignment_recovery.py              # Эксп.1–3: cos(w,g), Δd, recovery
├── build_inlp_subspace.py                 # INLP Phase A: AUC(k), W_k, centers
├── run_inlp_alignment.py                  # INLP Phase A.5: cos(w_j,g), rho_k
├── run_inlp_stagea.py                     # INLP Phase B: rank-k center, R(k)
├── analyze_inlp_stagea.py                 # CPU post-hoc: R_gender vs R_slot, saturation
├── check_inlp.py                          # unit-тесты rank-k интервенции
├── build_stagea_sample.py
├── build_stageb_sample.py                 # остаток val после Stage A
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
    └── metrics.json                    # θ/φ, per_soc, guardrail-rates, hook_check

results/steering/stage_b/<tag>/
├── run_meta.json
├── ranking.csv                         # preference + cap_loss
├── keep.json                           # 1–3 кандидата → Stage C
└── <config_id>/
    ├── preference_*.json(l)
    ├── capability_domain.json / mmlu_domain.jsonl
    ├── capability_smoke.json / mmlu_smoke.jsonl
    └── cap_loss.json
```

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
Проверка тождества в raw HS (модель не нужна, только `hidden_states.npz`):

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

## INLP + rank-k center

Rank-1 center/shift дал behavioral null при probe val AUC .946 (L16). Два
объяснения: preference распределена по `k>1` направлениям (rank-1 мало) — или
линейная декодируемость вообще не является каузально необходимой. Различает их
только совместная динамика двух кривых, `AUC(k)` и `BehaviorEffect(k)`.

Конфиг: `configs/inlp_gender_v1.yaml`. Слои выбраны **до** INLP по
`layer_scan` val AUC (L16 .946 пик, L15 .904, L18 .935; порог 0.9×max = .851).

### Phase A — representation (CPU)

```powershell
python -m steering.build_inlp_subspace              # L15,16,18, k_max=64
python -m steering.build_inlp_subspace --layers 16 --k-max 3 --tag smoke
python -m steering.build_inlp_subspace --verify
```

На каждой итерации обучается **новый** `Pipeline(StandardScaler → LogReg)` на
текущем остатке, `w_raw = coef/σ` ортогонализуется против найденных и
нормируется, затем `H ← H − (H w)wᵀ` для train и val одновременно. AUC меряется
только на held-out val (190 семей); `k_chance` требует AUC ≤ .55 **и** 95%
cluster-bootstrap CI, накрывающего .5. Если при `k_max` условие не выполнено,
`k_chance = null` и полное линейное стирание не заявляется.

Выход: `subspaces/inlp_gender_choice_v1.{npz,json,md}` — `W [2048,k]`,
`centers [k]`, `auc_curve`, схема §28 по итерациям и случайное ортонормированное
подпространство того же ранга (negative control).

#### Почему C = 0.03, а не дефолтный 1.0

При `C=1.0` (дефолт `layer_scan`) проба **интерполирует**: train AUC = 1.0000
при `n_train = 2284` и `d = 2048`. Направление тогда наполовину память, а не
сигнал, и INLP почти не двигает кривую — за 16 удалений AUC падает всего
.946 → .912. Это свойство пробы, а не представления.

Регуляризация подобрана по held-out AUC(0) (один fit, до INLP):

| C | L15 train/val | L16 train/val | L18 train/val |
|---:|---|---|---|
| 1.0 | 1.0000 / .9037 | 1.0000 / .9463 | 1.0000 / .9355 |
| 0.1 | .9947 / .9271 | .9976 / .9579 | .9964 / .9540 |
| **0.03** | .9858 / **.9325** | .9928 / **.9589** | .9908 / **.9564** |
| 0.01 | .9749 / .9310 | .9853 / .9560 | .9830 / .9547 |
| 0.001 | .9476 / .9192 | .9646 / .9449 | .9607 / .9433 |

`C=0.03` — не «ослабленная» проба: held-out AUC **выше**, чем при `C=1.0`, на
всех трёх слоях, а train перестаёт быть 1.0. Артефактная кривая при `C=1.0`
сохранена в `subspaces/inlp_gender_choice_c1p0.*` и воспроизводится флагом
`--clf-C 1.0`.

#### Результат Phase A (v1, `k_max = 64`)

| Layer | AUC(0) | AUC(8) | AUC(16) | AUC(32) | AUC(48) | AUC(64) | k_chance |
|---:|---:|---:|---:|---:|---:|---:|---|
| 15 | .932 | .881 | .827 | .752 | .683 | .587 | не достигнут |
| 16 | .959 | .882 | .779 | .666 | .596 | .574 | не достигнут |
| 18 | .956 | .857 | .756 | .664 | .592 | .576 | не достигнут |

**chance не достигается даже при k=64.** Но кривая ведёт себя не так, как
предполагала экстраполяция с `k_max=32` (`k_chance` ≈ 60–80): на L16 и L18 она
не продолжает спуск, а **выходит на плато ≈ .57** примерно к k≈48 и дальше
колеблется в пределах шума (L16 min .566 при k=56, затем .574; L18 .575–.60 на
всём участке 48–64). L15 к 64 всё ещё снижается (.587, минимум на последнем
шаге) — он просто стартовал ниже и идёт с задержкой.

Плато — не переобучение пробы: на последних итерациях train и val сходятся
вплотную (L16 k=63: train .585 / val .574), запоминать уже нечего. То есть
после снятия ~60 направлений остаётся устойчивый остаточный сигнал AUC ≈ .57,
который итеративное удаление **по одному направлению за раз не забирает**: он
размазан по многим направлениям с малой индивидуальной информативностью либо
не является линейно отделимым в том же смысле. Порог .55 при этом почти
достигнут, но формально ни одно из двух условий не выполнено (AUC > .55, а
95% CI = [.537, .615] не накрывает .5).

Содержательный вывод про представление: линейно декодируемая gender-preference
**не сосредоточена в низкоразмерном подпространстве** — 3% размерности модели
снимают её лишь до AUC ≈ .57. Для Phase B отсюда следует, что ни один ранг из
`[1, 2, 4, 8, 16, 32, 64]` не является полным линейным стиранием, и различение
Case A / Case D нужно формулировать с этой оговоркой. Одновременно rank-64
возмущает `h` уже заметно само по себе, поэтому random-контроль того же ранга
считается на **каждом** ранге, а не только на [1, 4].

### Phase A.5 — alignment (GPU)

```powershell
python -m steering.run_inlp_alignment --device cuda --dtype float32 --tag inlp_align_v1
```

Считает `cos(w_j, g)` по каждому направлению и `rho_k = ||W_kᵀg||²/||g||²` —
долю квадрата нормы выходного градиента, захваченную rank-k подпространством.
Референс — random rank-k, у которого `rho_k ≈ k/2048`. Если INLP даёт тот же
порядок, Phase B почти обязан дать Case A/D; это видно за минуты вместо сетки.

### Phase B — causal (GPU)

```powershell
python -m steering.run_inlp_stagea --device cuda --dtype float32 --tag inlp_v1
python -m steering.run_inlp_stagea --layers 16 --ranks 1,2 --limit-items 3 --tag smoke
```

`h' = h − α W_k(Wᵀh − c)`, α=1 фиксирован до появления воспроизводимого
эффекта: сначала `layer × rank`, только потом α-sweep. `shift` не включается —
rank-k shift потребовал бы `β_1..β_k` и раздул бы search space.

Сетка при k_max=64: 3 слоя × 7 рангов × (INLP + random) = 42 конфигурации ×
380 строк ≈ 16 тыс. forward.

Главный выход — `results/steering/inlp_stage_a/<tag>/auc_vs_behavior.csv`:
`AUC(k)` из Phase A рядом с behavioral R из Phase B.

### Почему построчное |d| нельзя брать за метрику

Скоринг — `z_A` vs `z_B`, а пол задаётся лейаутом строки, поэтому построчно

```
|d_gender| = |z_man − z_woman| ≡ |z_A − z_B| = |d_slot|
```

**тождественно** (в `mech_v1` это видно как совпадение `||g_gender||` и
`||g_slot||` до последней цифры). Значит любая интервенция, толкающая модель к
50/50 по слотам, даёт «снижение bias» — построчное `|d|`-падение не отличает
гендер от общей решительности A/B. Flip rate ломается так же.

Поэтому усреднение идёт по 4 лейаутам семьи **до** взятия модуля (`p0/p1`
меняют слот, `mf/wf` — порядок упоминания, позиционный вклад сокращается):

| метрика | формула | роль |
|---|---|---|
| `R_gender` | `\|mean_layouts d_gender\|_before − \|…\|_after` | primary |
| `R_slot` | то же для `d_slot` | guardrail: отделяет гендер от сглаживания A/B |
| `Δθ_dev` | `\|θ_i − .5\|_before − \|…\|_after`, `θ_i = mean p_man` | связь с H1 |
| `R_row` | построчное `\|d\|`-падение | вырождена, только для сопоставимости |

На синтетическом контрпримере (гасим ровно построчное `|d|`, оси не различая)
`R_row = 1.0`, тогда как разложение честно относит `0.75` к позиционной оси и
`0.25` к гендерной. CI — cluster bootstrap по `scenario_family_id`; семья и есть
кластер, поэтому позиционные варианты одного сценария не разлучаются.

### Unit-тесты (§30) — до любого GPU-прогона

```powershell
python -m steering.check_inlp
python -m steering.check_inlp --subspaces steering/subspaces/inlp_gender_choice_v1.npz
```

| тест | проверка |
|---|---|
| T1 | `WᵀW ≈ I` для всех сохранённых базисов, включая random control |
| T2 | `project_out` → `Wᵀh_⊥ ≈ 0` |
| T3 | `center α=1` → `Wᵀh' = c` (numpy и реальный хук в float32) |
| T4 | `α=0` → `h' = h` побитово, остальные позиции не тронуты |
| T5 | `SubspaceSpec(k=1)` == `InterventionSpec(center)` — regression rank-1 |

Модель не грузится: хук проверяется на игрушечном `nn.Module` с тем же
интерфейсом, что декодер. Пишет `results/steering/inlp_checks/<tag>/`.

### Финальный test

Phase A и Phase B работают на probe-val. `test`-семьи не участвуют ни в выборе
слоя, ни в выборе `k`, ни в выборе α — выборка для них строится только после
фиксации `layer*/rank*/alpha*`.

### Follow-up после `inlp_v1` (шаги 1–3)

Кандидат по val + A.5: **L15, k∈{8,16}, α=1**. L16 не подтверждаем.

**1. CPU post-hoc** — `R_gender` vs `R_slot`, насыщение, vs random:

```bash
# с артефакта Vast (скопировать ranking.csv в results/.../inlp_v1/)
python -m steering.analyze_inlp_stagea \
  --run-dir results/steering/inlp_stage_a/inlp_v1 \
  --layers 15 --focus-ranks 8,16 --with-soc

# без полного каталога — fixture из лога Vast
python -m steering.analyze_inlp_stagea --layers 15,16,18 --focus-ranks 8,16
```

**2. Confirmatory на test** (нужен `per_item.jsonl` исходного run):

```bash
# заморозить 95 test-семей (seed отдельный от val)
python -m steering.build_stagea_sample \
  --source-split test --n-base-items 95 --seed 20260821 \
  --out-stem inlp_test_sample_v1

# 4 конфига × ~380 строк (~3–4 мин @ 8 row/s)
python -m steering.run_inlp_stagea \
  --model Qwen/Qwen3.5-2B-Base --device cuda --dtype float32 \
  --sample steering/samples/inlp_test_sample_v1.json \
  --layers 15 --ranks 8,16 --alphas 1.0 \
  --tag inlp_confirm_test_v1
```

Критерий успеха: на test INLP `R_gender` CI не накрывает 0 и заметно выше
random того же k (как на val L15).

**3. α-sweep** — только если п.2 держится:

```bash
python -m steering.run_inlp_stagea \
  --model Qwen/Qwen3.5-2B-Base --device cuda --dtype float32 \
  --sample steering/samples/inlp_test_sample_v1.json \
  --layers 15 --ranks 8,16 \
  --alphas 0.25,0.5,0.75,1.0,1.25 \
  --no-random-control \
  --tag inlp_alpha_L15_v1
```

`--ranks` теперь ограничивает и random-контроль (иначе сетка раздувается до
всех степеней двойки из Phase A). Capability / MMLU — после успешного 2–3.

**4. Capability (Stage B)** — `cap_loss` на `mmlu_pro_domain_val` + `overall_smoke`:

```bash
# parquet один раз (если ещё нет)
curl -L -o steering/.cache/mmlu_pro_test.parquet \
  "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet"

# shortlist L15 k=8,16 + random (~5 × 1660 MMLU forwards)
python -m steering.run_inlp_stageb \
  --model Qwen/Qwen3.5-2B-Base --device cuda --dtype float32 \
  --tag inlp_b_v1

# smoke
python -m steering.run_inlp_stageb --device cuda --dtype float32 \
  --tag inlp_b_smoke --limit-mmlu 20 --candidates inlp__L15__k8__a1
```

Выход: `results/steering/inlp_stage_b/<tag>/` — `ranking.csv`, `keep.json`,
per-config `cap_loss.json`. Flag при `cap_loss > 0.03` (мягкий). Preference
пересчёт: `--with-preference` (по умолчанию выкл.).

**5. Stage C (report)** — остаток test + `mmlu_pro_domain_test` (без переизбрания):

```bash
# семпл уже в репо; пересборка при наличии per_item:
python -m steering.build_stageb_sample \
  --stage-a steering/samples/inlp_test_sample_v1.json \
  --out-stem inlp_stagec_sample_v1 \
  --title "INLP Stage C sample" \
  --capability-note "Preference Stage C; capability — mmlu_pro_domain_test_v1."

python -m steering.run_inlp_stagec --device cuda --dtype float32 --tag inlp_c_v1
# Vast: bash steering/scripts/vast_inlp_stagec_and_pack.sh
```

Выход: `results/steering/inlp_stage_c/<tag>/`. Preference + domain_test по keep
Stage B (`inlp__L15__k8__a1`, `inlp__L15__k16__a1`).

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

## Как запускать Stage B (slot)

Preference на **остатке val** (95 семей вне Stage A) + capability на
`mmlu_pro_domain_val_v1` (1100) и `overall_smoke` (560). Primary ranking —
`mean_abs_phi_prob_dev`; `cap_loss` — one-sided drop vs baseline (flag при
`--cap-loss-max`, дефолт 0.03). В `keep.json` — до 3 кандидатов для Stage C.

```powershell
# 1. Остаток val (один раз; verify после)
python -m steering.build_stageb_sample
python -m steering.build_stageb_sample --verify

# 2. Полный Stage B по shortlist из Stage A (~13 конфигов × ~2040 forward)
python -m steering.run_slot_stageb --model Qwen/Qwen3.5-2B-Base --device cuda `
  --dtype float32 --tag slot_b_v1

# или топ-15 из ranking.csv Stage A
python -m steering.run_slot_stageb --model Qwen/Qwen3.5-2B-Base --device cuda `
  --dtype float32 --tag slot_b_v1 `
  --from-ranking results/steering/stage_a/slot_full/ranking.csv --keep-top 15
```

Smoke (CPU/GPU, урезанный MMLU):

```powershell
python -m steering.run_slot_stageb --model steering/.cache/model --device cuda `
  --dtype float32 --tag slot_b_smoke --limit-items 2 --limit-mmlu 20 `
  --candidates main_center_core__wsperp__L23__center__a2,main_project_out__wsperp__L24__project_out__a1
```

## Как запускать Stage C (slot)

Report на **test** (95 семей, seed `20260911`) + `mmlu_pro_domain_test_v1`.
Без переизбрания — только keep Stage B.

```powershell
python -m steering.run_slot_stagec --model Qwen/Qwen3.5-2B-Base --device cuda `
  --dtype float32 --tag slot_c_v1

# Vast
bash steering/scripts/vast_slot_stagec_and_pack.sh
```

Smoke:

```powershell
python -m steering.run_slot_stagec --device cuda --dtype float32 `
  --tag slot_c_smoke --limit-items 2 --limit-mmlu 20 `
  --candidates main_center_core__wslot__L23__center__a2
```

Выход: `results/steering/stage_c/<tag>/`.

---

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
