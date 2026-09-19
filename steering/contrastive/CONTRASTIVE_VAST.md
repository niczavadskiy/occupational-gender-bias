# Contrastive Stage A — Vast, Qwen3.5-2B-Base

Нужны **`HF_TOKEN`** и публичный репо `niczavadskiy/occupational-gender-bias`
(каталог `steering/contrastive/` должен быть в `main`).

Torch: image default OK if `numpy<2`. Setup чинит env:

```bash
bash scripts/setup_instance.sh
```

Не ставь корневой `requirements.txt` с `numpy>=2` на Vast.

Полный прогон: live HS → mean-diff \(\hat v,c\) → rank PCA \(D\) → Stage A (46 кандидатов + baseline)
на `h1_stagea_sample_v1.json` (95 семей × 4). Модель: `Qwen/Qwen3.5-2B-Base`.

Только спектр / \(V_k\) (без сетки 46):

```bash
bash steering/contrastive/scripts/vast_contrastive_rank_and_pack.sh
RUN_PCA_STAGEA=1 bash steering/contrastive/scripts/vast_contrastive_rank_and_pack.sh
```

## Если репо уже есть на инстансе

Не клонируйте заново. Подтяните `main` и запускайте из корня репо:

```bash
export HF_TOKEN=hf_xxx

cd /workspace/occupational-gender-bias
git fetch --depth 1 origin main
git checkout main
git reset --hard origin/main

ls steering/contrastive/scripts/vast_contrastive_full_instance.sh
bash steering/contrastive/scripts/vast_contrastive_full_instance.sh
```

`reset --hard` нужен, чтобы подтянуть каталог `steering/contrastive/` (коммит `c8dd3a3`).
Локальные правки на инстансе сотрутся.

## Свежий инстанс (пустой /workspace)

```bash
export HF_TOKEN=hf_xxx

cd /workspace
git clone --depth 1 https://github.com/niczavadskiy/occupational-gender-bias.git || true
cd occupational-gender-bias && git pull origin main

bash steering/contrastive/scripts/vast_contrastive_full_instance.sh
```

Только full, без smoke:

```bash
export HF_TOKEN=hf_xxx
SKIP_SMOKE=1 bash steering/contrastive/scripts/vast_contrastive_full_instance.sh
```

## Репо уже на месте

```bash
cd /workspace/occupational-gender-bias
export HF_TOKEN=hf_xxx REPO=$PWD PYTHONPATH=$PWD

# smoke
SMOKE=1 SCALE=2b bash steering/contrastive/scripts/vast_contrastive_stagea_and_pack.sh

# full 2B
SCALE=2b bash steering/contrastive/scripts/vast_contrastive_stagea_and_pack.sh
```

## Jupyter (без SSH)

Ноутбук: [`notebooks/vast_contrastive_2b.ipynb`](../../notebooks/vast_contrastive_2b.ipynb)  
Вставьте `HF_TOKEN`, гоняйте ячейки сверху вниз.

## Артефакты

| Путь | Зачем |
|---|---|
| `steering/contrastive/candidates/contrastive_gender_2b_candidates_v1.json` | сетка 46 + baseline |
| `steering/samples/h1_stagea_sample_v1.json` | 95 val-семей × 4 |
| `steering/contrastive/vectors/contrastive_gender_2b_vectors_v1.npz` | пишется на инстансе |
| `steering/contrastive/rank/contrastive_gender_2b_rank_v1.md` | спектр D, вердикт 1D vs subspace |
| `steering/subspaces/contrastive_pca_2b_v1.npz` | \(V_k\) для Stage A (k=1 ≡ \(v_G\)) |

Выход: `results/steering/contrastive/stage_a/<tag>/` → pack  
`/workspace/contrastive_stage_a_contrastive_2b_a_v1.tar.gz`.

Скачать: Vast UI → Files, или `scp` с хоста.

## Оценка времени

- smoke: минуты
- full Stage A: ~47 × 380 forward на 2B float32 (порядка нескольких часов на одной GPU)
