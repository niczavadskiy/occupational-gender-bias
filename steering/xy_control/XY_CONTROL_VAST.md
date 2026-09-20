# Vast: XY-control

Нужен `HF_TOKEN` (Hub). `causal-conv1d` по умолчанию **не ставится**
(source build на Vast зависает). Уже стоящие torch/transformers достаточны:
`SKIP_PIP=1`.

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
C подтверждает замороженный winner на test-семьях и `domain_test`.

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
