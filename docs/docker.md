# Docker image для rented GPU

Пайплайн каждый раз на новом Vast-инстансе требует install цикла (torch+cu128 +
transformers + ~6 других deps). Это занимает 5–10 мин и порой ломается из-за
mismatch CUDA на host'е. Docker image заменяет это на одну команду pull.

## Что в образе

- Ubuntu 22.04 + CUDA 12.8 runtime libs
- Python 3.10 + pip
- torch 2.11+cu128
- Всё из `requirements.txt` (transformers 5.9, accelerate, hf_hub, numpy, pandas, etc.)
- Интерактивные тулзы: git, tmux, htop, vim, ssh-client

## Что НЕ в образе

- Код проекта (`git clone` после старта инстанса — code меняется чаще чем deps).
- Веса модели (~4 GB Qwen3.5-2B-Base) — качаются при первом forward через HF,
  кешируются в `$HF_HOME=/workspace/.hf_home`. На Vast `/workspace` персистентен
  per-instance, так что в рамках одной аренды качается один раз.
- HF token — берётся из env `HF_TOKEN` при старте инстанса (см. ниже).
- Данные / результаты — `.dockerignore` отрезает.

Размер итогового образа: ~5 GB.

## Build

### Через GitHub Actions (рекомендуется — авто-build, ничего качать локально)

Workflow [.github/workflows/docker.yml](../.github/workflows/docker.yml) автоматически
build'ит и push'ит image при любом push'е на main / qwen_2b_experiments,
который меняет Dockerfile, requirements.txt или сам workflow. Также есть ручной
trigger (Actions → "Docker build & push" → "Run workflow").

**Что нужно настроить один раз** (после первого push'а workflow'а):

1. Создать [Docker Hub PAT](https://app.docker.com/settings/personal-access-tokens),
   scope **Read & Write**.
2. В GitHub repo (`olyamasaeva/Bias--subspaces-in-LLM`) → Settings → Secrets and
   variables → Actions → New repository secret:
   - `DOCKERHUB_USERNAME` = `sportsprogrammerhunter`
   - `DOCKERHUB_TOKEN` = PAT из шага 1
3. (Один раз вручную) Создать repo на Docker Hub —
   `sportsprogrammerhunter/bias-subspaces-env`, отметить как **Public**, чтобы
   Vast pull'ил без credentials.

После настройки — любой коммит, меняющий Dockerfile, запускает CI:
build ~10-12 мин, image публикуется в трёх тегах:
- `:latest`
- `:YYYY-MM-DD` (дата CI-прогона)
- `:sha-<short>` (short git SHA для точного pin'а)

### Build локально (альтернатива — если нет желания возиться с CI)

В корне репо:

```bash
docker build -t sportsprogrammerhunter/bias-subspaces-env:latest .
```

Тест локально (CPU-only, проверит что torch грузится, без CUDA forward):

```bash
docker run --rm sportsprogrammerhunter/bias-subspaces-env:latest \
    python3 -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
# → 2.11.0+cu128 5.9.0
```

GPU forward локально не запустится без NVIDIA driver — это нормально, проверка
только что image well-formed. Реальный прогон — на Vast.

## Push в registry

Через GitHub Actions push автоматический (см. секцию Build выше). Manual push
из локали — только если зачем-то build'ишь сама:

```bash
docker login -u sportsprogrammerhunter
docker tag sportsprogrammerhunter/bias-subspaces-env:latest \
           sportsprogrammerhunter/bias-subspaces-env:$(date +%Y-%m-%d)
docker push sportsprogrammerhunter/bias-subspaces-env:latest
docker push sportsprogrammerhunter/bias-subspaces-env:$(date +%Y-%m-%d)
```

## Использование на Vast.ai

1. **Create Template** (один раз):
   - **Image Path/Tag**: `sportsprogrammerhunter/bias-subspaces-env:latest`
     (или dated: `:2026-06-02` если нужен конкретный snapshot)
   - **Docker Options**: `-e HF_TOKEN=hf_xxxxxxxxx --shm-size=8g`
   - **Launch mode**: ssh (для tmux-прогона) либо jupyter (если хочется ноутбук)
   - **On-start script** (опционально, автоматизирует clone):
     ```bash
     cd /workspace && \
     git clone https://github.com/<user>/Bias--subspaces-in-LLM.git && \
     cd Bias--subspaces-in-LLM && \
     git checkout qwen_2b_experiments
     ```

2. **Create Instance** из template на любой подходящей GPU (RTX 3090/4090/A100,
   24+ GB VRAM, 30+ GB disk).

3. **SSH в инстанс** — всё уже стоит, можно сразу:
   ```bash
   cd /workspace/Bias--subspaces-in-LLM
   python3 src/smoke_qwen.py           # проверка модель грузится
   python3 src/prepare_factorial.py data/factorial_v2_with_gender.csv
   python3 src/inference.py --items_file data/factorial_v2_with_gender.prepared.jsonl \
       --run_tag factorial_v2_full
   ```

   Время первого smoke: ~2-3 мин (HF download модели). Дальнейшие forward'ы — мгновенно.

## Tradeoff'ы дизайна

| Решение | Почему |
|---|---|
| Не баковать веса модели в image | Image остаётся 5 GB вместо 12 GB, push/pull быстрее. Качается один раз per Vast-аренду. |
| Не баковать код проекта | Code меняется чаще deps. `git clone` в `on-start` дешевле rebuild'а image на каждый коммит. |
| Не баковать HF token | Безопасность — image публиковать в любом registry без риска утечки. Token инжектится через `-e HF_TOKEN=...` Vast template'а. |
| `HF_HOME=/workspace/.hf_home` | На Vast `/workspace` persistent per-instance, что переживает stop/start одного и того же rent. |
| `requirements.txt` pinned exact versions | Detect'мся в одном слое; любое breaking change ловится при build, не на GPU в момент прогона. |

## Update procedure

Когда меняем deps:
1. Поправили `requirements.txt`.
2. `docker build -t bias-subspaces-env:latest .` (layer cache → build быстрый, ~1 мин если меняется только requirements.txt).
3. Push с двумя тэгами:
   ```bash
   DATE=$(date +%Y-%m-%d)
   docker tag bias-subspaces-env:latest sportsprogrammerhunter/bias-subspaces-env:latest
   docker tag bias-subspaces-env:latest sportsprogrammerhunter/bias-subspaces-env:$DATE
   docker push sportsprogrammerhunter/bias-subspaces-env:latest
   docker push sportsprogrammerhunter/bias-subspaces-env:$DATE
   ```
4. На Vast при создании нового инстанса tag `:latest` подтянет свежий. Для
   reproducibility конкретного прогона — pin'аем `:2026-06-02`.
