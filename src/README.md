# src/ — pipeline scripts

## Текущие

| Файл | Что |
|---|---|
| `smoke_qwen.py` | Smoke-test Qwen3.5-2B-Base: загрузка, проверка token IDs для A/B/C, forward + hidden states, log-prob ratio. Запуск: `python3 src/smoke_qwen.py`. |
| `inference.py` | Полный inference: constrained log-prob A/B/C + HS snapshot всех слоёв на last-token. Сохраняет `hidden_states.npz` + `per_item.jsonl` + `meta.json` в `/workspace/results/run_*/`. Запуск: `python3 src/inference.py` (на built-in demo) или `--items_file factorial.jsonl`. |
| `upload_to_hf.py` | Заливка results folder в приватный HF Dataset repo. Запуск: `python3 src/upload_to_hf.py results/run_2026-05-28_*/`. |

## Запланированы

| Файл | Что |
|---|---|
| `metrics.py` | H1-H9 behavioral метрики: rate, asymmetry, McNemar, two-prop z-test, FDR correction. |
| `probe_pipeline.py` | H10-H13 linear probe: layer-scan, control task (Hewitt-Liang), silhouette score, direction extraction. |
| `cross_form_eval.py` | Cross-form transferability test для localization claim. |

## Зависимости

На свежем Vast / локально (в образе PyTorch (Vast) обычно torch уже есть, но **может быть несовместимая CUDA**):

```bash
# Если torch отсутствует или с неправильной CUDA:
pip install torch --index-url https://download.pytorch.org/whl/cu128

pip install -U transformers accelerate huggingface_hub numpy
pip install scikit-learn matplotlib pandas tqdm
```

## Quick start на Vast (full setup)

```bash
# 1. Clone repo
cd /workspace
git clone https://github.com/olyamasaeva/Bias--subspaces-in-LLM.git
cd Bias--subspaces-in-LLM
git checkout qwen_2b_experiments

# 2. Install deps (см. выше)

# 3. HF login (нужно для скачивания Qwen без rate-limit'а)
hf auth login --token hf_XXX
# или: export HF_TOKEN=hf_XXX

# 4. Smoke test (3-5 мин первый раз — качает модель ~4 GB)
python3 src/smoke_qwen.py

# 5. Inference на demo items (15 сек после первого load'а)
python3 src/inference.py
ls /workspace/results/

# 6. Upload results to HF Datasets (на локальной машине после scp)
scp -r vast-bias:/workspace/results/run_*/ ./results/
python3 src/upload_to_hf.py results/run_2026-05-28_*/
```

## Notes

- **`trust_remote_code=True`** обязателен для Qwen3.5 (новая архитектура — Gated DeltaNet + Gated Attention).
- TransformerLens может не поддерживать архитектуру — fall back на `transformers` + manual hooks для HS extraction.
- Под HS-extraction snapshot: `[N_items, n_layers+1, d_model] = [N, 25, 2048]` bf16 ≈ 100 KB per item.
- HF Datasets: `olyamasaeva/qwen-bias-experiments` (private). Сделать public/share: Settings → Visibility/Collaborators.
