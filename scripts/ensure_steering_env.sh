#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Общее Vast-окружение для steering / inference.
#
# numpy>=2 ломает torch на bias-subspaces-env / conda; часто нет transformers.
# Этот скрипт выбирает интерпретатор и ставит pinned deps.
#
#   bash scripts/ensure_steering_env.sh
#   export PY="$(cat /tmp/occupational_steering_py)"
#
# Из другого скрипта (source — без set -e на caller):
#   source scripts/ensure_steering_env.sh && ensure_steering_env
#
# Env: PY  SKIP_PIP=1  UPGRADE_TORCH=1  REPO
# ---------------------------------------------------------------------------

STEERING_ENV_PY_FILE="${STEERING_ENV_PY_FILE:-/tmp/occupational_steering_py}"

resolve_steering_py() {
  if [ -n "${PY:-}" ]; then
    if [ -x "$PY" ] || command -v "$PY" >/dev/null 2>&1; then
      echo "$PY"
      return 0
    fi
  fi
  local c
  for c in /venv/main/bin/python /opt/conda/bin/python python3 python; do
    if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then
      echo "$c"
      return 0
    fi
  done
  echo "python3"
}

ensure_steering_env() {
  PY="$(resolve_steering_py)"
  export PY
  printf '%s\n' "$PY" >"$STEERING_ENV_PY_FILE" || return 1

  echo "=== ensure_steering_env: PY=$PY ==="
  "$PY" -c "import sys; print(' ', sys.executable, sys.version.split()[0])" || return 1

  if [ "${SKIP_PIP:-0}" != "1" ]; then
    local req=""
    if [ -n "${REPO:-}" ] && [ -f "$REPO/scripts/requirements-vast.txt" ]; then
      req="$REPO/scripts/requirements-vast.txt"
    elif [ -f "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/requirements-vast.txt" ]; then
      req="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/requirements-vast.txt"
    fi

    echo "  pip: numpy<2 + transformers (без -q — прогресс виден)…"
    if [ -n "$req" ]; then
      "$PY" -m pip install -U -r "$req" || return 1
    else
      "$PY" -m pip install -U \
        'numpy>=1.26,<2' \
        'transformers>=4.50' \
        accelerate huggingface_hub python-dotenv PyYAML scipy pandas || return 1
    fi

    if [ "${UPGRADE_TORCH:-0}" = "1" ]; then
      echo "  pip: UPGRADE_TORCH=1 → torch>=2.5…"
      "$PY" -m pip install -U 'torch>=2.5' torchvision || return 1
    fi
  else
    echo "  SKIP_PIP=1"
  fi

  echo "  sanity import…"
  if ! "$PY" - <<'PY'
import sys
import numpy as np

if int(np.__version__.split(".")[0]) >= 2:
    print(
        f"FAIL: numpy {np.__version__} >= 2 — нужен numpy<2 для этого torch.\n"
        f"  {sys.executable} -m pip install 'numpy>=1.26,<2'",
        file=sys.stderr,
    )
    sys.exit(2)

import torch
import transformers

print(
    f"  OK numpy={np.__version__} torch={torch.__version__} "
    f"cuda={torch.cuda.is_available()} transformers={transformers.__version__}"
)
if not torch.cuda.is_available():
    print("  WARN: CUDA unavailable", file=sys.stderr)
PY
  then
    local rc=$?
    if [ "$rc" = "2" ] && [ "${SKIP_PIP:-0}" != "1" ]; then
      echo "  retry: force numpy<2…"
      "$PY" -m pip install -U --force-reinstall 'numpy>=1.26,<2' || return 1
      "$PY" -c "import numpy,torch,transformers; print('retry OK', numpy.__version__, torch.__version__, transformers.__version__, torch.cuda.is_available())" || return 1
    else
      return "$rc"
    fi
  fi
  return 0
}

# Прямой запуск: bash scripts/ensure_steering_env.sh
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  ensure_steering_env
  exit $?
fi
