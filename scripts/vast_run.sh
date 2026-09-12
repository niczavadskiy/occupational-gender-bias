#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Полный lifecycle Vast-инстанса: поиск → аренда → SSH → (опц.) прогон → destroy.
#
# Решает ручную возню: фильтр по CUDA вшит (cuda_vers>=MIN_CUDA), образ вшит,
# инстанс ГАРАНТИРОВАННО удаляется на выходе (trap), даже если упало посередине.
#
# Требования (один раз):
#   pip install --upgrade vastai
#   vastai set api-key <КЛЮЧ>            # https://cloud.vast.ai/account/  (раздел API)
#   # и добавить свой ~/.ssh/id_*.pub в аккаунт Vast (Account → SSH Keys),
#   # либо скрипт сам зальёт через VAST_SSH_PUBKEY (см. ниже).
#
# Примеры:
#   # 1) только посмотреть кандидатов, ничего не арендуя:
#   DRY=1 bash scripts/vast_run.sh
#
#   # 2) арендовать самый дешёвый подходящий, отдать SSH-команду и НЕ удалять:
#   KEEP=1 bash scripts/vast_run.sh
#
#   # 3) арендовать, прогнать inference 5400 и удалить инстанс по завершении:
#   #    (inference.py пишет в $RESULTS_DIR = /workspace/results, НЕ в repo/results!)
#   HF_TOKEN=$(cat ~/.cache/huggingface/token) GH_TOKEN=$(gh auth token) \
#   REMOTE_CMD='bash scripts/setup_instance.sh && \
#       python3 src/prepare_factorial.py data/factorial_v2_with_gender.csv && \
#       python3 src/inference.py --items_file data/factorial_v2_with_gender.prepared.jsonl --run_tag factorial_v2_5400 && \
#       python3 src/upload_to_hf.py /workspace/results/run_*factorial_v2_5400*' \
#   bash scripts/vast_run.sh
#
# Параметры (env, со значениями по умолчанию):
GPU="${GPU:-RTX_3090}"               # gpu_name в терминах Vast
MIN_CUDA="${MIN_CUDA:-12.8}"         # минимальная версия CUDA драйвера хоста (образ cu128)
DISK="${DISK:-60}"                   # ГБ диска (образ 6.5 + модель 4 + npz + запас)
MAX_DPH="${MAX_DPH:-0.40}"           # потолок $/час
MIN_RELIAB="${MIN_RELIAB:-0.98}"     # минимальная reliability
IMAGE="${IMAGE:-sportsprogrammerhunter/bias-subspaces-env:latest}"
BRANCH="${BRANCH:-qwen_2b_experiments_olya}"
LABEL="${LABEL:-bias-subspaces}"
REMOTE_CMD="${REMOTE_CMD:-}"         # что выполнить на инстансе (пусто → только SSH-инфо)
KEEP="${KEEP:-0}"                    # 1 → не удалять инстанс на выходе
DRY="${DRY:-0}"                      # 1 → только поиск, без аренды
# ---------------------------------------------------------------------------
set -euo pipefail
command -v vastai >/dev/null || { echo "нет vastai → pip install --upgrade vastai"; exit 1; }
PYJSON='import sys,json; d=json.load(sys.stdin)'

# Токены: -e на кастомном образе НЕ доходит до ssh-сессии (Vast не пишет их в
# /etc/environment для не-своих образов) → пробрасываем env прямо в команды.
HF_TOKEN="${HF_TOKEN:-}"; GH_TOKEN="${GH_TOKEN:-}"
RESULTS_DIR="${RESULTS_DIR:-/workspace/results}"   # куда inference.py пишет run'ы
RENV="export RESULTS_DIR='$RESULTS_DIR'"
[ -n "$HF_TOKEN" ] && RENV="$RENV; export HF_TOKEN='$HF_TOKEN'"   # нужен и для inference (иначе rate-limit на скачивании модели), и для upload
[ -n "$GH_TOKEN" ] && RENV="$RENV; export GH_TOKEN='$GH_TOKEN'"

QUERY="gpu_name=${GPU} num_gpus=1 cuda_vers>=${MIN_CUDA} disk_space>=${DISK} \
reliability>${MIN_RELIAB} dph<${MAX_DPH} rentable=true verified=true"

echo "=== [1] поиск офферов ==="
echo "    фильтр: $QUERY"
OFFERS="$(vastai search offers "$QUERY" -o 'dph' --raw)"
COUNT="$(printf '%s' "$OFFERS" | python3 -c "$PYJSON; print(len(d))")"
[ "$COUNT" = "0" ] && { echo "ничего не найдено — ослабь MAX_DPH/MIN_RELIAB/DISK"; exit 1; }

printf '%s' "$OFFERS" | python3 -c "$PYJSON
for o in d[:5]:
    print(f\"  id={o['id']:>9}  \${o['dph_total']:.3f}/hr  CUDA{o['cuda_max_good']}  \"
          f\"{o.get('geolocation','?')}  disk{int(o['disk_space'])}GB  reliab{o['reliability2']:.3f}\")"

OFFER_ID="$(printf '%s' "$OFFERS" | python3 -c "$PYJSON; print(d[0]['id'])")"
echo "    выбран самый дешёвый: $OFFER_ID"
[ "$DRY" = "1" ] && { echo "DRY=1 — выходим без аренды"; exit 0; }

echo "=== [2] аренда инстанса ($IMAGE, disk ${DISK}GB) ==="
ENVOPT="--shm-size=8g"
[ -n "${HF_TOKEN:-}" ] && ENVOPT="$ENVOPT -e HF_TOKEN=$HF_TOKEN"
[ -n "${GH_TOKEN:-}" ] && ENVOPT="$ENVOPT -e GH_TOKEN=$GH_TOKEN"
CREATE="$(vastai create instance "$OFFER_ID" --image "$IMAGE" --disk "$DISK" \
            --ssh --direct --env "$ENVOPT" --label "$LABEL" --raw)"
IID="$(printf '%s' "$CREATE" | python3 -c "$PYJSON; print(d.get('new_contract') or d.get('id'))")"
[ -z "$IID" ] || [ "$IID" = "None" ] && { echo "create не вернул id: $CREATE"; exit 1; }
echo "    instance id: $IID"

# --- уборка: destroy ТОЛЬКО при успехе; при ошибке инстанс остаётся для разбора ---
SUCCESS=0; SSHCMD=""
cleanup() {
  if [ "$KEEP" = "1" ]; then
    echo "=== KEEP=1 — инстанс $IID ОСТАВЛЕН. reconnect: ${SSHCMD:-(ssh ещё не готов)} ==="
    echo "    снести вручную: vastai destroy instance $IID -y"
  elif [ "$SUCCESS" = "1" ]; then
    echo "=== [destroy] успех → удаляю инстанс $IID ==="
    vastai destroy instance "$IID" -y || echo "  WARN: destroy не прошёл — снеси: vastai destroy instance $IID -y"
  else
    echo "=== ⚠️ ОШИБКА → инстанс $IID ОСТАВЛЕН для разбора (GPU не горит, если stopped) ==="
    echo "    reconnect: ${SSHCMD:-(ssh не готов)}"
    echo "    снести:    vastai destroy instance $IID -y"
  fi
}
trap cleanup EXIT INT TERM

echo "=== [3] жду готовности (running + реальный SSH-проб) ==="
READY=0; SSH_HOST=""; SSH_PORT=""
for i in $(seq 1 90); do
  INFO="$(vastai show instance "$IID" --raw 2>/dev/null || true)"
  read -r ST CST SSH_HOST SSH_PORT <<<"$(printf '%s' "$INFO" | python3 -c "
$PYJSON
print(d.get('actual_status','?'), d.get('cur_state','?'), d.get('ssh_host') or '-', d.get('ssh_port') or '-')" 2>/dev/null || echo '? ? - -')"
  echo "  [$i] actual=$ST cur=$CST ssh=$SSH_HOST:$SSH_PORT"
  if [ "$ST" = "running" ] && [ "$CST" = "running" ] && [ "$SSH_HOST" != "-" ]; then
    if ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o BatchMode=yes \
           -p "$SSH_PORT" "root@$SSH_HOST" true 2>/dev/null; then READY=1; break; fi
  fi
  sleep 10
done
[ "$READY" != "1" ] && { echo "инстанс не вышел в running+ssh за ~15 мин — оставляю для разбора, выходим"; exit 1; }

SSHCMD="ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=20 -o ConnectTimeout=30 -p $SSH_PORT root@$SSH_HOST"
echo "=== [4] инстанс готов ==="
echo "    SSH:  $SSHCMD"

echo "=== [4.5] bootstrap: clone репо (если on-start ещё не склонировал) ==="
# $RENV экспортит GH_TOKEN на инстансе; затем \$GH_TOKEN раскрывается там же в URL
$SSHCMD "$RENV
cd /workspace && { [ -d Bias--subspaces-in-LLM/.git ] || \
  git clone --depth 1 --branch $BRANCH \
    https://\$GH_TOKEN@github.com/olyamasaeva/Bias--subspaces-in-LLM.git; }" \
  || { echo "clone не прошёл (GH_TOKEN передан?)"; exit 1; }

if [ -n "$REMOTE_CMD" ]; then
  echo "=== [5] выполняю REMOTE_CMD на инстансе (в репо, с токенами в env) ==="
  if $SSHCMD "$RENV
cd /workspace/Bias--subspaces-in-LLM && $REMOTE_CMD"; then
    SUCCESS=1; echo "=== REMOTE_CMD успешно завершён ==="
  else
    echo "=== REMOTE_CMD упал (rc=$?) — инстанс оставляю для разбора ==="
  fi
else
  KEEP=1
  echo "    REMOTE_CMD пуст — инстанс оставлен. Подключайся: $SSHCMD"
fi
