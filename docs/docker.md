# Docker image для rented GPU

Пайплайн каждый раз на новом Vast-инстансе требует install цикла (torch+cu128 +
transformers + ~6 других deps). Это занимает 5–10 мин и порой ломается из-за
mismatch CUDA на host'е. Docker image заменяет это на одну команду pull.

## Что в образе

- Ubuntu 24.04 + CUDA 12.8 runtime libs
- Python 3.12 + pip
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
build'ит и push'ит image при любом push'е на main / qwen_2b_experiments /
qwen_2b_experiments_olya,
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

> ⚠️ **Чтобы образ подцепился, а не дефолтный pytorch-шаблон Vast** — поле
> **Image** при создании инстанса должно быть заполнено. Самый надёжный способ —
> **сохранить свой Template** (тогда он выбирается по умолчанию для всех будущих
> аренд). Если оставить дефолт — Vast поднимет свой образ (torch есть, но без
> наших deps), и придётся доустанавливать вручную.

1. **Create Template** (один раз, потом переиспользуется):
   - **Image Path/Tag**: `sportsprogrammerhunter/bias-subspaces-env:latest`
     (или dated: `:2026-06-02` если нужен конкретный snapshot)
   - **Launch mode**: `SSH` (Vast сам поднимет sshd поверх образа) либо jupyter
   - **Docker Options**: `-e HF_TOKEN=hf_xxx -e GH_TOKEN=ghp_xxx --shm-size=8g`
   - **Disk**: ползунок **≥ 50 GB** (образ ~6.5 GB + модель 4 GB + npz; меньше →
     `No space left on device`)
   - **On-start script** — автоматизирует clone приватного репо + pull старых
     данных с HF (см. `scripts/setup_instance.sh`):
     ```bash
     cd /workspace && \
     git clone --depth 1 --branch qwen_2b_experiments_olya \
       https://$GH_TOKEN@github.com/olyamasaeva/Bias--subspaces-in-LLM.git && \
     cd Bias--subspaces-in-LLM && bash scripts/setup_instance.sh
     ```
   - **Save Template** → в следующий раз образ подцепится сразу.

   Либо без UI, через **vast-cli**:
   ```bash
   vastai create instance <OFFER_ID> \
     --image sportsprogrammerhunter/bias-subspaces-env:latest \
     --disk 50 --ssh --direct \
     --env '-e HF_TOKEN=hf_xxx -e GH_TOKEN=ghp_xxx --shm-size=8g'
   ```

   > 🔒 Токены в Docker Options живут на инфраструктуре Vast (чужое железо).
   > Используй **fine-grained read-only** токены (GH — на этот репо, HF — read на
   > org) и отзывай после прогона.

2. **Create Instance** из template на подходящей GPU. Фильтр: **Max CUDA ≥ 12.8**
   (образ на cu128), 24+ GB VRAM, disk ≥ 50 GB.

3. **SSH в инстанс** — всё уже стоит. Если on-start script не использовала —
   запусти setup вручную:
   ```bash
   export GH_TOKEN=ghp_xxx HF_TOKEN=hf_xxx
   cd /workspace && \
     git clone --depth 1 --branch qwen_2b_experiments_olya \
       https://$GH_TOKEN@github.com/olyamasaeva/Bias--subspaces-in-LLM.git
   cd Bias--subspaces-in-LLM && bash scripts/setup_instance.sh
   ```
   Скрипт: clone кода + download прошлого прогона (HS + per_item) с HF Dataset.

4. **Прогон** — код и данные на месте:
   ```bash
   python3 src/prepare_factorial.py data/factorial_v2_with_gender.csv   # → 5400 items
   # свежий прогон:
   python3 src/inference.py --items_file data/factorial_v2_with_gender.prepared.jsonl \
       --run_tag factorial_v2_5400
   # либо с переиспользованием старого forward'а (нужен скачанный с HF run):
   python3 src/inference.py --items_file data/factorial_v2_with_gender.prepared.jsonl \
       --run_tag factorial_v2_5400 \
       --cache_from results/run_2026-05-27_19-44-46_Qwen3.5-2B-Base_factorial_v2
   ```
   Время первого forward'а: +~30-60 сек (HF download модели 4 GB на скорости
   инстанса). Прогон 5400 items: ~25 мин на RTX 3090.

## Использование уже посчитанных данных (HF Dataset)

Прошлые прогоны (`hidden_states.npz` + `per_item.jsonl`) лежат в приватном
HF Dataset `bias-subspaces-group/qwen-bias-experiments`. **Тянуть их надо на
инстанс с HF (быстрый CDN), а не аплоадить со своей машины.**

```bash
hf download bias-subspaces-group/qwen-bias-experiments \
   --repo-type dataset --local-dir results/      # делает и setup_instance.sh
```

Три способа применить скачанный run:
- **`--cache_from results/run_<...>`** — новый inference переиспользует forward'ы
  совпавших промптов (экономия только если `hidden_states.npz` присутствует —
  без него HS пересчитывается всё равно, см. `src/inference.py` логику cache-hit).
- **probing H11** (`python -m probes.h11_run …`) — CPU, гоняется прямо на `.npz`,
  **GPU вообще не нужен** (можно и локально).
- **behavioral-метрики** — на `per_item.jsonl`.

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
