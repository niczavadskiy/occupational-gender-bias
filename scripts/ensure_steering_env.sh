#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Общее Vast-окружение для steering / inference.
#
# numpy>=2 ломает torch на bias-subspaces-env / conda; часто нет transformers;
# битый torchaudio ломает import Qwen3.5 (transformers → audio_utils).
#
#   bash scripts/ensure_steering_env.sh
#   export PY="$(cat /tmp/occupational_steering_py)"
#
# Env: PY  SKIP_PIP=1  UPGRADE_TORCH=1  KEEP_TORCHAUDIO=1  SKIP_FLA=1
#      INSTALL_CAUSAL_CONV1D=1  REPO  PIN_VAST_ENV=0
#      (PIN_VAST_ENV default 1 → scripts/install_vast_env.py from git lock)
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

# Text-only steering: drop broken torchaudio (default). KEEP_TORCHAUDIO=1 → matching wheel.
fix_torchaudio() {
  if "$PY" - <<'PY' 2>/dev/null
import torchaudio
from torchaudio._extension import _IS_TORCHAUDIO_EXT_AVAILABLE
# force load native lib path used by transformers
import torchaudio
print("ok")
PY
  then
    echo "  torchaudio: import OK"
    return 0
  fi

  echo "  torchaudio: broken or ABI mismatch with torch"
  if [ "${KEEP_TORCHAUDIO:-0}" = "1" ]; then
    echo "  KEEP_TORCHAUDIO=1 → reinstall matching torchaudio…"
    local ver cuda tag idx
    ver="$("$PY" -c "import torch; print(torch.__version__.split('+')[0])")"
    cuda="$("$PY" -c "import torch; print(torch.version.cuda or '')")"
    case "$cuda" in
      12.4*|12.5*|12.6*) tag=cu124 ;;
      12.1*|12.2*|12.3*) tag=cu121 ;;
      11.8*) tag=cu118 ;;
      *) tag=cu121 ;;
    esac
    idx="https://download.pytorch.org/whl/${tag}"
    echo "  pip install torchaudio==$ver index=$idx"
    "$PY" -m pip uninstall -y torchaudio >/dev/null 2>&1 || true
    if ! "$PY" -m pip install -U "torchaudio==${ver}" --index-url "$idx"; then
      echo "  matching wheel failed → uninstall torchaudio (text-only OK)"
      "$PY" -m pip uninstall -y torchaudio >/dev/null 2>&1 || true
    fi
  else
    echo "  uninstall torchaudio (Qwen text load не требует audio; set KEEP_TORCHAUDIO=1 to keep)"
    "$PY" -m pip uninstall -y torchaudio >/dev/null 2>&1 || true
  fi
  return 0
}

ensure_steering_env() {
  PY="$(resolve_steering_py)"
  export PY
  printf '%s\n' "$PY" >"$STEERING_ENV_PY_FILE" || return 1

  echo "=== ensure_steering_env: PY=$PY ==="
  "$PY" -c "import sys; print(' ', sys.executable, sys.version.split()[0])" || return 1

  if [ "${SKIP_PIP:-0}" != "1" ]; then
    local script_dir repo_scripts installer lock_json
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo_scripts="${REPO:-}/scripts"
    installer=""
    if [ -n "${REPO:-}" ] && [ -f "$REPO/scripts/install_vast_env.py" ]; then
      installer="$REPO/scripts/install_vast_env.py"
    elif [ -f "$script_dir/install_vast_env.py" ]; then
      installer="$script_dir/install_vast_env.py"
    fi

    # Default: frozen Vast lock from git (torch 2.5.1+cu121, torchvision 0.20.1, transformers 5.17).
    if [ "${PIN_VAST_ENV:-1}" = "1" ] && [ -n "$installer" ]; then
      echo "  pip: PIN_VAST_ENV=1 → $installer"
      if ! "$PY" "$installer"; then
        echo "  WARN: install_vast_env failed — falling back to requirements-vast.txt"
        PIN_VAST_ENV=0
      fi
    fi

    if [ "${PIN_VAST_ENV:-1}" != "1" ]; then
      local req=""
      if [ -n "${REPO:-}" ] && [ -f "$REPO/scripts/requirements-vast.txt" ]; then
        req="$REPO/scripts/requirements-vast.txt"
      elif [ -f "$script_dir/requirements-vast.txt" ]; then
        req="$script_dir/requirements-vast.txt"
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
    fi
  else
    echo "  SKIP_PIP=1"
  fi

  fix_torchaudio

  # Qwen3.5: flash-linear-attention + causal-conv1d (optional; slow fallback if missing)
  if [ "${SKIP_PIP:-0}" != "1" ] && [ "${SKIP_FLA:-0}" != "1" ]; then
    echo "  pip: flash-linear-attention + causal-conv1d (best-effort)…"
    if ! "$PY" -c "import flash_linear_attn" 2>/dev/null && ! "$PY" -c "import fla" 2>/dev/null; then
      "$PY" -m pip install -U flash-linear-attention || \
        echo "  WARN: flash-linear-attention install failed — slow chunk_gated_delta_rule fallback"
    else
      echo "  flash-linear-attention: already importable"
    fi
    if ! "$PY" -c "import causal_conv1d" 2>/dev/null; then
      # Source tarball (1.7.0) compiles CUDA kernels and often hangs or ABI-mismatches
      # on Vast. Qwen3.5 already has a slow PyTorch fallback.
      if [ "${INSTALL_CAUSAL_CONV1D:-0}" = "1" ]; then
        "$PY" -m pip install -U causal-conv1d || \
          echo "  WARN: causal-conv1d install failed — slow causal_conv1d_fn fallback"
      else
        echo "  skip causal-conv1d source build (set INSTALL_CAUSAL_CONV1D=1 to try)"
      fi
    else
      echo "  causal-conv1d: already importable"
    fi
  fi

  echo "  sanity import…"
  if ! "$PY" - <<'PY'
import importlib
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
from transformers import AutoModelForCausalLM  # noqa: F401

# Qwen3.5 pulls modeling → processing_utils → audio_utils → torchaudio
for mod in (
    "transformers.models.qwen3_5.modeling_qwen3_5",
    "transformers.models.qwen2.modeling_qwen2",
):
    try:
        importlib.import_module(mod)
        print(f"  model import OK: {mod}")
        break
    except ModuleNotFoundError:
        continue
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if "torchaudio" in msg or "libtorchaudio" in msg:
            print(f"FAIL torchaudio path: {msg}", file=sys.stderr)
            sys.exit(3)
        raise

print(
    f"  OK numpy={np.__version__} torch={torch.__version__} "
    f"cuda={torch.cuda.is_available()} transformers={transformers.__version__}"
)
for name, mods in (
    ("flash-linear-attention", ("fla", "flash_linear_attn")),
    ("causal-conv1d", ("causal_conv1d",)),
):
    ok = False
    for m in mods:
        try:
            importlib.import_module(m)
            print(f"  OK optional {name} ({m})")
            ok = True
            break
        except Exception:
            continue
    if not ok:
        print(f"  WARN optional missing: {name} (slow PyTorch fallback)")
if not torch.cuda.is_available():
    print("  WARN: CUDA unavailable", file=sys.stderr)
PY
  then
    local rc=$?
    if [ "$rc" = "2" ] && [ "${SKIP_PIP:-0}" != "1" ]; then
      echo "  retry: force numpy<2…"
      "$PY" -m pip install -U --force-reinstall 'numpy>=1.26,<2' || return 1
      "$PY" -c "import numpy,torch,transformers; print('retry OK', numpy.__version__, torch.__version__, transformers.__version__, torch.cuda.is_available())" || return 1
    elif [ "$rc" = "3" ]; then
      echo "  retry: strip torchaudio…"
      "$PY" -m pip uninstall -y torchaudio >/dev/null 2>&1 || true
      "$PY" - <<'PY' || return 1
import importlib
import transformers
from transformers import AutoModelForCausalLM  # noqa: F401
for mod in (
    "transformers.models.qwen3_5.modeling_qwen3_5",
    "transformers.models.qwen2.modeling_qwen2",
):
    try:
        importlib.import_module(mod)
        print("retry OK", mod, transformers.__version__)
        break
    except ModuleNotFoundError:
        continue
PY
    else
      return "$rc"
    fi
  fi
  return 0
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  ensure_steering_env
  exit $?
fi
