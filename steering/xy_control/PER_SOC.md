# Per-SOC XY-control

Стиринг **только** на `soc_major_title` с FDR-значимым H1 (choice и/или
prob, `without_abstain`). Незначимые домены пропускаются (identity).
Мужские и женские домены нейтрализуются **разными** \(v_{\mathrm{raw}}\).

Порядок: **2B, затем 4B**; внутри модели — каждый значимый SOC по каталогу.

## Правило включения

Заморожено в [`domains/h1_soc_fdr_v1.md`](domains/h1_soc_fdr_v1.md):

- union FDR-rejected H1 SOC strata (choice family **или** prob family)
- полярность = знак значимого эффекта; если оба reject — берётся choice
- male → prior \(\alpha<0\) (add \(v\)); female → prior \(\alpha>0\) (subtract \(v\))
- оба знака \(\alpha\) всё равно гриддятся; prior только для отбора

| модель | steer | male | female | skip |
|---|---:|---:|---:|---:|
| 2B | 10 | 8 | 2 | 9 |
| 4B | 10 | 4 | 6 | 9 |

2B female (через prob-FDR): Healthcare Practitioners, Personal Care.
4B male: Construction, Installation, Production (prob), Transportation (prob).

## Что делается на домене

1. Пул: все H1 `without_abstain` семьи этого SOC (из `xy_pairs_full_v1`, не 10% sample).
2. Внутри SOC: split 70/15/15 семей.
3. \(v_{\mathrm{raw}}^{\mathrm{soc}} = \mathrm{mean}(h_{\mathrm{gender}}-h_{\mathrm{XY}})\) на train этого SOC.
4. Additive hook на last token: \(h'=h-\alpha\hat v\), сетка как в глобальном Stage A (core+edge, оба знака).
5. GenderGap на val этого SOC (выбор α). test этого SOC — после выбора конфига.

Число семей = все `without_abstain` семьи домена. Тот же per-SOC 70/15/15, что и в глобальном пайплайне.

| SOC | семьи | пары | 2B | 4B |
|---|---:|---:|:---:|:---:|
| Architecture and Engineering Occupations | 65 | 260 | steer | skip |
| Computer and Mathematical Occupations | 30 | 120 | steer | skip |
| Construction and Extraction Occupations | 61 | 244 | steer | steer |
| Educational Instruction and Library Occupations | 60 | 240 | steer | steer |
| Food Preparation and Serving Related Occupations | 17 | 68 | skip | steer |
| Healthcare Practitioners and Technical Occupations | 86 | 344 | steer | steer |
| Healthcare Support Occupations | 17 | 68 | skip | steer |
| Installation, Maintenance, and Repair Occupations | 54 | 216 | steer | steer |
| Life, Physical, and Social Science Occupations | 60 | 240 | steer | steer |
| Office and Administrative Support Occupations | 63 | 252 | skip | steer |
| Personal Care and Service Occupations | 32 | 128 | steer | skip |
| Production Occupations | 108 | 432 | steer | steer |
| Transportation and Material Moving Occupations | 53 | 212 | steer | steer |

2B: 10 SOC → **609** семей / **2436** пар (train 427, val 91, test 91).
4B: 10 SOC → **579** семей / **2316** пар (train 406, val 87, test 86).

Ключ вектора: `v_raw__{slug}__L{layer}`.

## Запуск

```powershell
python -m steering.xy_control.build_domains --verify
python -m steering.xy_control.test_domains
python -m steering.xy_control.test_per_soc

# обе модели по порядку
python -m steering.xy_control.run_per_soc --device cuda

# по одной
python -m steering.xy_control.run_per_soc --scale 2b --device cuda --tag xy_2b_per_soc_v1
python -m steering.xy_control.run_per_soc --scale 4b --device cuda --tag xy_4b_per_soc_v1
```

Smoke (первый значимый SOC, якорь, \(\alpha=\pm 1\), 3 семьи):

```powershell
python -m steering.xy_control.run_per_soc --scale 2b --smoke --device cuda
```

Один домен:

```powershell
python -m steering.xy_control.run_per_soc --scale 4b --only-soc construction_and_extraction_occupations --device cuda
```

Результаты: `results/steering/xy_control/per_soc/<tag>/`

Полная сетка: 10 доменов × 44 кандидата + 10 baseline = 450 конфигов на модель
(core 4×8 + edge 3×4; random-контроль не дублируется на каждый SOC).

- `summary.csv` — лучший кандидат на SOC (и лучший с α по prior)
- `ranking.csv` — все конфиги
- `skip.csv` — незначимые / не выбранные
- `<slug>/ranking.csv` — сетка этого домена

## Vast

```bash
export HF_TOKEN=hf_xxx
bash steering/xy_control/scripts/vast_xy_control_per_soc_full_instance.sh
```

Только 2B: `SCALES=2b`. Только 4B: `vast_xy_control_per_soc_4b_full_instance.sh`.
Только smoke: `SKIP_FULL=1`. `SKIP_PIP=1`, если `causal-conv1d` не собирается.

Pack: `/workspace/xy_control_per_soc_<tag>.tar.gz` (результаты + векторы).

## Stage B / C

Stage A выше использует `val` для GenderGap и замораживает не больше двух
кандидатов на SOC:

- `best_id` — максимальное уменьшение \(|GenderGap|\);
- `best_prior_id` — лучший кандидат со знаком α из H1-prior (если отличается).

Stage B **не считает GenderGap на val повторно**. Он проверяет эти кандидаты на
50 вопросах соответствующего SOC из `mmlu_pro_domain_val_v1`. Порог
`cap_loss=max(0, acc_baseline-acc_steered)` по умолчанию 0.03. Среди прошедших
берётся минимальный `cap_loss`; при равенстве — prior. Если не прошёл никто,
для SOC замораживается identity.

Stage C не переизбирает кандидата:

- preference: `test` семьи того же SOC;
- capability: дизъюнктный `mmlu_pro_domain_test_v1`;
- результат: только confirmatory report.

На том же Vast-инстансе после Stage A:

```bash
# 2B
SCALES=2b bash steering/xy_control/scripts/vast_xy_control_per_soc_stagebc_full_instance.sh

# 4B
SCALES=4b bash steering/xy_control/scripts/vast_xy_control_per_soc_stagebc_full_instance.sh
```

Выходы разделены по модели и стадии:

- `results/steering/xy_control/stage_b/xy_<scale>_per_soc_b_v1/`
- `results/steering/xy_control/stage_c/xy_<scale>_per_soc_c_v1/`
- `/workspace/xy_control_per_soc_stage_bc_<scale>_v1.tar.gz`
