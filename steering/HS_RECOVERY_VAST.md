# HS / AUC recovery (Method 3) — Vast, Qwen3.5-2B-Base

Нужны **`HF_TOKEN`**, публичный репо `niczavadskiy/occupational-gender-bias`,
артефакты `steering/subspaces/inlp_gender_choice_v1.npz` + `run_hs_recovery_auc.py`
(должны быть в `main` после push).

Torch: image default OK if `numpy<2`. Setup всегда чинит env:

```bash
bash scripts/setup_instance.sh   # → ensure_steering_env (numpy<2 + transformers)
```

Pinned: `scripts/requirements-vast.txt` (`numpy>=1.26,<2`, `transformers>=4.50`,
`flash-linear-attention`, `causal-conv1d`).
`ensure_steering_env` снимает битый `torchaudio` (ломает `AutoModelForCausalLM` для
Qwen3.5). Оставить audio: `KEEP_TORCHAUDIO=1`.

Если в логе Qwen: `causal_conv1d` / `flash-linear-attention` falling back — вручную:

```bash
export PY="$(cat /tmp/occupational_steering_py 2>/dev/null || echo /opt/conda/bin/python)"
"$PY" -m pip install -U flash-linear-attention causal-conv1d
```

Не ставь корневой `requirements.txt` с `numpy>=2` на Vast.

## Быстрый путь (свежий инстанс)

```bash
export HF_TOKEN=hf_xxx

cd /workspace
# clone + sync
bash -c 'git clone --depth 1 https://github.com/niczavadskiy/occupational-gender-bias.git || true'
cd occupational-gender-bias && git pull

# полный прогон: smoke → full → pack в /workspace/*.tar.gz
bash steering/scripts/vast_hs_recovery_full_instance.sh
```

Или через общий setup:

```bash
export HF_TOKEN=hf_xxx
bash scripts/setup_instance.sh
bash steering/scripts/vast_hs_recovery_full_instance.sh
```

## Только runner (репо уже на месте)

```bash
cd /workspace/occupational-gender-bias
export HF_TOKEN=hf_xxx REPO=$PWD PYTHONPATH=$PWD

# smoke
SMOKE=1 bash steering/scripts/vast_hs_recovery_auc_and_pack.sh

# full L15 k16
bash steering/scripts/vast_hs_recovery_auc_and_pack.sh
```

## Артефакты

| Путь | Зачем |
|---|---|
| `steering/subspaces/inlp_gender_choice_v1.npz` | INLP W/c на L15/16/18 (`3545e531…`) |
| `steering/vectors/h1_vectors_v1.npz` | frozen ŵ для AUC |
| `steering/samples/h1_stagea_sample_v1.json` | 95 семей × 4 |

Выход: `results/steering/hs_recovery/<tag>/` → pack  
`/workspace/hs_recovery_<tag>.tar.gz`.

## Если `git pull` без subspaces / скрипта

С локальной машины (после `vastai ssh-url`):

```bash
scp steering/run_hs_recovery_auc.py \
    steering/scripts/vast_hs_recovery_auc_and_pack.sh \
    steering/scripts/vast_hs_recovery_full_instance.sh \
    root@HOST:/workspace/occupational-gender-bias/steering/scripts/   # scripts — поправить пути

# проще целиком:
scp steering/run_hs_recovery_auc.py root@HOST:/workspace/occupational-gender-bias/steering/
scp -r steering/subspaces root@HOST:/workspace/occupational-gender-bias/steering/
scp steering/scripts/vast_hs_recovery_*.sh root@HOST:/workspace/occupational-gender-bias/steering/scripts/
```

## Чтение результата

В `by_layer.csv` / `summary.json`: `md_auc_base_test` → `md_auc_steer_test`.  
Recovery ≈ провал AUC на L15 и рост на L16…L24.
