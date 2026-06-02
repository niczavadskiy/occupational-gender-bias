# ---------------------------------------------------------------------------
# Bias-subspaces research env.
# Базовый образ для запуска pipeline'а на rented GPU (Vast.ai и аналоги).
# Содержит: Python 3.12, torch 2.11+cu128, transformers 5.9, deps из
# requirements.txt + git/tmux/ssh для интерактивной работы.
# НЕ содержит: код проекта (git clone в /workspace), веса модели (HF cache).
# Размер: ~5 GB.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# Системные пакеты: python + интерактивные тулзы
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        git curl wget openssh-client tmux htop nano vim less \
        ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# --ignore-installed: на ubuntu24.04 debian-pip без RECORD-файла нельзя
# деинсталлировать «поверх» — ставим свежий рядом, не трогая системный.
RUN python3 -m pip install --upgrade --ignore-installed pip

# Torch — отдельно с cu128-индексом
RUN python3 -m pip install \
        torch --index-url https://download.pytorch.org/whl/cu128

# Python deps — копируем только requirements.txt чтобы layer кешировался
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# HF cache в /workspace — на Vast этот путь обычно persistent volume,
# модель не качается заново при reboot одного и того же инстанса.
ENV HF_HOME=/workspace/.hf_home

WORKDIR /workspace

# Метаданные
LABEL org.opencontainers.image.title="bias-subspaces-env" \
      org.opencontainers.image.description="Qwen3.5-2B-Base bias-subspaces research env (torch 2.11+cu128, transformers 5.9)"

# Дефолт — интерактивный bash; на Vast это всё равно overridнется его CMD
CMD ["/bin/bash"]
