# XY-control steering

Парный контрфактуальный контроль: один и тот же вопрос с гендерными
обозначениями и с `X`/`Y`.

**Маппинг фиксирован:** `X = man`, `Y = woman`.

Направление — **gender-conditioned activation direction**, не «gender direction»:

\[
d_i = h^{\mathrm{gender}}_i - h^{\mathrm{XY}}_i, \qquad
v_{\mathrm{raw}} = \mathrm{mean}_i\, d_i
\]

HS снимается с last prompt token (после `Answer:`), по слоям независимо.
Интервенция аддитивная, тот же сайт:

\[
h' = h - \alpha \hat v_{\mathrm{raw}}
\]

Оба знака \(\alpha\) обязательны. XY-промпты используются **только** для
построения \(v\); GenderGap считается на gendered prompts.

Position control сохранён: `p0/p1` × `man_first/woman_first`, и те же четыре
лейаута в XY (`A. X / B. Y` и `A. Y / B. X`).

## Каталог

```
steering/xy_control/
├── data/                 # парный датасет + mapping.json
├── configs/              # 2B L14–20 / 4B L20–26
├── candidates/
├── domains/              # замороженный H1 SOC-FDR каталог
├── mapping.py            # X=man, Y=woman
├── build_dataset.py
├── build_candidates.py
├── build_vectors.py      # paired HS → v̂ (глобальный)
├── build_vectors_per_soc.py
├── run_stagea.py         # add-steering + GenderGap (глобальный)
├── run_per_soc.py        # стиринг только FDR-значимых SOC
├── run_per_soc_stageb.py # best+prior → MMLU-Pro domain_val → freeze
├── run_per_soc_stagec.py # frozen winner → test + MMLU-Pro domain_test_xy_250
├── build_stagec_mmlu_profile.py  # 250 held-out MMLU ids per FDR SOC
└── scripts/
```

Per-SOC протокол (значимые домены отдельно, 2B затем 4B):
[`PER_SOC.md`](PER_SOC.md).

PCA / control-space denoising **не входят** в первый эксперимент. Их добавляем
только если baseline \(v_{\mathrm{raw}}\) двигает GenderGap.

## Датасет

```powershell
python -m steering.xy_control.test_dataset
python -m steering.xy_control.build_dataset
python -m steering.xy_control.build_dataset --verify
```

| файл | роль | семьи × пары |
|---|---|---|
| `data/mapping.json` | X=man, Y=woman | — |
| `data/xy_pairs_full_v1.json` | **основной**: весь H1 `without_abstain` | **951 × 3804** |
| `data/xy_pairs_stagea_v1.json` | legacy 10% val | 95 × 380 |

JSONL рядом с каждым JSON — одна пара на строку.

Срез: 951 семьи × p0/p1 × man_first/woman_first. Evidence и `with_abstain` не входят.

Сплит **внутри каждого** `soc_major_title` (не глобальный шаффл):

| доля | семьи | роль |
|---|---:|---|
| train 70% | ~667 | fit \(v_{\mathrm{raw}}\), затем \(h'=h-\alpha\hat v\) |
| val 15% | 142 | Stage A: GenderGap **после** стиринга (выбор α/слоя) |
| test 15% | 142 | замок: финальный GenderGap после выбора конфига |

`--eval-split val` по умолчанию (142 × 4 = 568 строк на конфиг).
`--eval-split test` — только когда α и слой уже выбраны.
XY-промпты не скорятся на eval.

`X-ray` в стоматологических сценариях не трогается: заменяются только
`\bman\b` / `\bwoman\b`, поэтому число букв X в XY-промпте может быть больше
числа `man`.

Для отбора \(\alpha\) на val:

```powershell
python -m steering.xy_control.run_stagea --eval-split val
```

После выбора конфига — GenderGap на test:

```powershell
python -m steering.xy_control.run_stagea --eval-split test --candidates <winner_id>
```

## Запуск

Из корня `occupational-gender-bias`:

```powershell
python -m steering.xy_control.build_candidates
python -m steering.xy_control.build_candidates --scale 4b

python -m steering.xy_control.build_vectors --device cuda
python -m steering.xy_control.build_vectors --scale 4b --device cuda

python -m steering.xy_control.run_stagea --device cuda --tag xy_2b_a_v1
python -m steering.xy_control.run_stagea --scale 4b --device cuda --tag xy_4b_a_v1
```

Smoke:

```powershell
python -m steering.xy_control.build_vectors --n-items 3 --device cuda --tag smoke
python -m steering.xy_control.run_stagea --limit-items 3 --tag smoke --eval-split all `
  --candidates main_add_core__vraw__L16__add__a1,main_add_core__vraw__L16__add__am1
```

MMLU-Pro (тот же хук, тот же протокол):

```powershell
python -m steering.xy_control.run_stagea --device cuda --mmlu smoke --tag xy_2b_a_v1
```

## Метрики

На gendered prompt, constrained логиты букв A/B, маппинг слота → man/woman:

- \(\Delta_{\mathrm{row}} = z_{\mathrm{man}} - z_{\mathrm{woman}}\) (равносильно разнице constrained log-softmax)
- \(\Delta_i = \mathrm{mean}_{layouts} \Delta_{\mathrm{row}}\) — семья, 4 лейаута
- **GenderGap** \(= \mathrm{mean}_i \Delta_i\)
- `mean_abs_delta` \(= \mathrm{mean}_i |\Delta_i|\) — модуль **после** усреднения лейаутов
- \(P(\Delta_i > 0)\)
- Guardrail: slot gap \(\mathrm{mean}(z_A - z_B)\) по лейаутам

Не называть \(v_{\mathrm{raw}}\) gender direction. Сравнить с пробой можно
по `cos(v_raw, w_gender)` в мета векторов.

## Vast

[`XY_CONTROL_VAST.md`](XY_CONTROL_VAST.md)

Per-SOC (значимые домены, 2B затем 4B):

```bash
export HF_TOKEN=hf_xxx
bash steering/xy_control/scripts/vast_xy_control_per_soc_full_instance.sh
```

Глобальный Stage A:

```bash
export HF_TOKEN=hf_xxx
bash steering/xy_control/scripts/vast_xy_control_full_instance.sh
bash steering/xy_control/scripts/vast_xy_control_4b_full_instance.sh
```
