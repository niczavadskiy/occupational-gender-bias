# src/ — pipeline scripts

## Текущие

| Файл | Что |
|---|---|
| `smoke_qwen.py` | Smoke-test Qwen3.5-2B-Base: загрузка, проверка token IDs для A/B/C, forward + hidden states, log-prob ratio. Запуск: `python3 src/smoke_qwen.py`. |

## Запланированы (по мере поступления данных)

| Файл | Что |
|---|---|
| `inference.py` | Полный inference на factorial-датасете: batch, log-prob A/B/C, snapshot HS всех слоёв last-token. Output: `.npz`. |
| `metrics.py` | H1-H9 behavioral метрики: rate, asymmetry, McNemar, two-prop z-test, FDR correction. |
| `probe_pipeline.py` | H10-H13 linear probe: layer-scan, control task (Hewitt-Liang), silhouette score, direction extraction. |
| `cross_form_eval.py` | Cross-form transferability test для localization claim. |

## Зависимости

```bash
# в образе уже есть torch + CUDA
pip install -U transformers accelerate
pip install sentence-transformers matplotlib pandas tqdm scikit-learn
```

## Quick start на Vast

```bash
cd /workspace
git clone https://github.com/olyamasaeva/Bias--subspaces-in-LLM.git
cd Bias--subspaces-in-LLM
git checkout qwen_2b_experiments

# первый запуск качает Qwen ~4 GB
python3 src/smoke_qwen.py
```

## Notes

- **`trust_remote_code=True`** обязателен для Qwen3.5 (новая архитектура — Gated DeltaNet + Gated Attention).
- TransformerLens может не поддерживать архитектуру — fall back на `transformers` + manual `register_forward_hook` для HS extraction.
- Под HS-extraction snapshot: `[N_items, n_layers+1, d_model] = [N, 25, 2048]` fp16 ≈ 100 KB per item.
