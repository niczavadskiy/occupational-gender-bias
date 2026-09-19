# Vast: XY-control Stage A

```bash
export HF_TOKEN=hf_xxx
cd /workspace/occupational-gender-bias && git pull
bash steering/xy_control/scripts/vast_xy_control_full_instance.sh
```

4B:

```bash
SCALE=4b MODEL=Qwen/Qwen3.5-4B-Base bash steering/xy_control/scripts/vast_xy_control_full_instance.sh
```

Только smoke:

```bash
SKIP_FULL=1 bash steering/xy_control/scripts/vast_xy_control_full_instance.sh
```

Pack: `/workspace/xy_control_stage_a_<tag>.tar.gz`

MMLU smoke на полном прогоне: `MMLU=smoke`.
