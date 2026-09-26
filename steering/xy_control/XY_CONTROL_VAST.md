# Vast: XY-control

Нужен `HF_TOKEN` (Hub). `causal-conv1d` по умолчанию **не ставится**
(source build на Vast зависает).

## Замороженное окружение (git)

Источник правды: [`scripts/env/vast_steering_cu121.json`](../../scripts/env/vast_steering_cu121.json)

| пакет | pin |
|---|---|
| Python | 3.10 |
| CUDA wheel tag | **cu121** |
| torch | **2.5.1** |
| torchvision | **0.20.1** |
| transformers | **5.17.0** |
| numpy | **1.26.4** (`<2`) |
| torchaudio | uninstall (text-only) |

После `git pull` на инстансе:

```bash
cd /workspace/occupational-gender-bias
python scripts/install_vast_env.py          # поставить/проверить lock
python scripts/install_vast_env.py --check-only
```

`scripts/ensure_steering_env.sh` / `setup_instance.sh` по умолчанию
`PIN_VAST_ENV=1` и вызывают этот installer. Отключить: `PIN_VAST_ENV=0` или
`SKIP_PIP=1` (если стек уже совпадает с lock).

Полный пул: 951 семьи × 4 layout. Сплит **внутри каждого SOC** 70/15/15.
Stage A по умолчанию — GenderGap на **val**. test не трогать, пока не выбран α.

## Per-SOC (FDR-значимые домены)

2B затем 4B, у каждого SOC свой \(v_{\mathrm{raw}}\):

```bash
export HF_TOKEN=hf_xxx
bash steering/xy_control/scripts/vast_xy_control_per_soc_full_instance.sh
```

Только 2B / только 4B:

```bash
SCALES=2b bash steering/xy_control/scripts/vast_xy_control_per_soc_full_instance.sh
bash steering/xy_control/scripts/vast_xy_control_per_soc_4b_full_instance.sh
```

Только smoke (1 домен, якорь, \(\alpha=\pm 1\)):

```bash
SKIP_FULL=1 bash steering/xy_control/scripts/vast_xy_control_per_soc_full_instance.sh
```

Pack: `/workspace/xy_control_per_soc_<tag>.tar.gz`  
(результаты + `v_raw` npz). Протокол: [`PER_SOC.md`](PER_SOC.md).

### Per-SOC Stage B/C после Stage A

Запускать на том же инстансе: Stage B читает `summary.csv`, domain rankings и
векторы Stage A. B проверяет `best + best_prior` на MMLU-Pro `domain_val`;
C подтверждает замороженный winner на test-семьях и
`mmlu_pro_domain_test_xy_250_v2` (250 held-out вопросов на FDR SOC).

Обновлённый Stage A также сохраняет точный `xy_pairs_full_v1.json` в своём
каталоге результатов. Stage C использует только эту копию, сверяет SHA-256 и
проверяет наличие **всех** замороженных test family IDs до загрузки модели.
Затронутые старые Stage A артефакты без dataset hash можно мигрировать CPU-only:

```bash
python -m steering.xy_control.migrate_stagea \
  --stage-a-dir results/steering/xy_control/per_soc/xy_2b_per_soc_v1 --scale 2b
python -m steering.xy_control.migrate_stagea \
  --stage-a-dir results/steering/xy_control/per_soc/xy_4b_per_soc_v1 --scale 4b
```

Если валидный `stagec_keep.json` уже есть, повторяется только C:

```bash
RUN_STAGE_B=0 SCALES=2b bash steering/xy_control/scripts/vast_xy_control_per_soc_stagebc_full_instance.sh
RUN_STAGE_B=0 SCALES=4b bash steering/xy_control/scripts/vast_xy_control_per_soc_stagebc_full_instance.sh
```

```bash
# только 2B
SCALES=2b bash steering/xy_control/scripts/vast_xy_control_per_soc_stagebc_full_instance.sh

# только 4B
SCALES=4b bash steering/xy_control/scripts/vast_xy_control_per_soc_stagebc_full_instance.sh

# обе по очереди
bash steering/xy_control/scripts/vast_xy_control_per_soc_stagebc_full_instance.sh
```

Packs:

- `/workspace/xy_control_per_soc_stage_bc_2b_v1.tar.gz`
- `/workspace/xy_control_per_soc_stage_bc_4b_v1.tar.gz`

## Глобальный Stage A (один \(v\) на все домены)

```bash
export HF_TOKEN=hf_xxx
bash steering/xy_control/scripts/vast_xy_control_full_instance.sh
# 4B:
bash steering/xy_control/scripts/vast_xy_control_4b_full_instance.sh
```

Только smoke:

```bash
SKIP_FULL=1 bash steering/xy_control/scripts/vast_xy_control_full_instance.sh
SKIP_FULL=1 bash steering/xy_control/scripts/vast_xy_control_4b_full_instance.sh
```

Pack: `/workspace/xy_control_stage_a_<tag>.tar.gz`

MMLU smoke на полном глобальном прогоне: `MMLU=smoke`.

```bash
export SKIP_PIP=1
SCALES=2b bash steering/xy_control/scripts/vast_xy_control_per_soc_full_instance.sh
```
