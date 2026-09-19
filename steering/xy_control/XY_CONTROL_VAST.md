# Vast: XY-control Stage A

```bash
export HF_TOKEN=hf_xxx
cd /workspace/occupational-gender-bias && git pull
bash steering/xy_control/scripts/vast_xy_control_full_instance.sh
```

4B (Qwen3.5-4B-Base, пояс L20–26, якорь L23):

```bash
export HF_TOKEN=hf_xxx
bash steering/xy_control/scripts/vast_xy_control_4b_full_instance.sh
```

Только smoke:

```bash
SKIP_FULL=1 bash steering/xy_control/scripts/vast_xy_control_full_instance.sh
# 4B:
SKIP_FULL=1 bash steering/xy_control/scripts/vast_xy_control_4b_full_instance.sh
```

Pack: `/workspace/xy_control_stage_a_<tag>.tar.gz`

MMLU smoke на полном прогоне: `MMLU=smoke`.

Если `setup_instance` падает на сборке `causal-conv1d` (CUDA mismatch), env
уже достаточный — повторите с `SKIP_PIP=1`:

```bash
export SKIP_PIP=1
bash steering/xy_control/scripts/vast_xy_control_full_instance.sh
# 4B:
SKIP_PIP=1 bash steering/xy_control/scripts/vast_xy_control_4b_full_instance.sh
```
