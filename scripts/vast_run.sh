#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Полный lifecycle Vast-инстанса: поиск → аренда → SSH → (опц.) прогон → destroy.
#
# Требования (один раз на машине, с которой запускаете скрипт):
#   pip install --upgrade vastai
#   vastai set api-key <КЛЮЧ>            # https://cloud.vast.ai/account/
#   # SSH-ключ в аккаунте Vast (Account → SSH Keys)
#
# Репо на инстансе: niczavadskiy/occupational-gender-bias (public, branch main).
# GH_TOKEN не обязателен. HF_TOKEN нужен для скачивания модели с Hugging Face.
#
# Примеры:
#   # 1) только посмотреть кандидатов:
#   DRY=1 bash scripts/vast_run.sh
#
#   # 2) арендовать, отдать SSH и НЕ удалять:
#   KEEP=1 HF_TOKEN=hf_xxx bash scripts/vast_run.sh
#
#   # 3) Qwen3.5-4B H1·H3 полный прогон + pack (рекомендуется KEEP=1):
#   HF_TOKEN=hf_xxx DISK=120 KEEP=1 \
#   REMOTE_CMD='bash scripts/setup_instance.sh && bash scripts/vast_qwen35_4b_h1h3_and_pack.sh' \
#   bash scripts/vast_run.sh
#
# Параметры (env):
GPU="${GPU:-RTX_3090}"
MIN_CUDA="${MIN_CUDA:-12.8}"
DISK="${DISK:-120}"                   # 4B + HS: лучше ≥120
MAX_DPH="${MAX_DPH:-0.40}"
MIN_RELIAB="${MIN_RELIAB:-0.98}"
IMAGE="${IMAGE:-sportsprogrammerhunter/bias-subspaces-env:latest}"
REPO="${REPO:-niczavadskiy/occupational-gender-bias}"
REPO_DIR="${REPO_DIR:-occupational-gender-bias}"
BRANCH="${BRANCH:-main}"
LABEL="${LABEL:-occupational-gender-bias}"
REMOTE_CMD="${REMOTE_CMD:-}"
KEEP="${KEEP:-0}"
DRY="${DRY:-0}"
# ---------------------------------------------------------------------------
set -euo pipefail
command -v vastai >/dev/null || { echo "нет vastai → pip install --upgrade vastai"; exit 1; }
PYJSON='import sys,json; d=json.load(sys.stdin)'

# Токены: -e на кастомном образе часто НЕ доходит до ssh-сессии →
# пробрасываем env прямо в remote-команды.
HF_TOKEN="${HF_TOKEN:-}"; GH_TOKEN="${GH_TOKEN:-}"
RESULTS_DIR="${RESULTS_DIR:-/workspace/results}"
RENV="export RESULTS_DIR='$RESULTS_DIR'"
[ -n "$HF_TOKEN" ] && RENV="$RENV; export HF_TOKEN='$HF_TOKEN'"
[ -n "$GH_TOKEN" ] && RENV="$RENV; export GH_TOKEN='$GH_TOKEN'"
RENV="$RENV; export BRANCH='$BRANCH'; export REPO='$REPO'; export REPO_DIR='$REPO_DIR'"

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

if [ -z "$HF_TOKEN" ]; then
  echo "WARN: HF_TOKEN пуст — модель с Hub может не скачаться / rate-limit"
fi

echo "=== [2] аренда инстанса ($IMAGE, disk ${DISK}GB) ==="
ENVOPT="--shm-size=8g"
[ -n "${HF_TOKEN:-}" ] && ENVOPT="$ENVOPT -e HF_TOKEN=$HF_TOKEN"
[ -n "${GH_TOKEN:-}" ] && ENVOPT="$ENVOPT -e GH_TOKEN=$GH_TOKEN"
CREATE="$(vastai create instance "$OFFER_ID" --image "$IMAGE" --disk "$DISK" \
            --ssh --direct --env "$ENVOPT" --label "$LABEL" --raw)"
IID="$(printf '%s' "$CREATE" | python3 -c "$PYJSON; print(d.get('new_contract') or d.get('id'))")"
[ -z "$IID" ] || [ "$IID" = "None" ] && { echo "create не вернул id: $CREATE"; exit 1; }
echo "    instance id: $IID"

SUCCESS=0; SSHCMD=""
cleanup() {
  if [ "$KEEP" = "1" ]; then
    echo "=== KEEP=1 — инстанс $IID ОСТАВЛЕН. reconnect: ${SSHCMD:-(ssh ещё не готов)} ==="
    echo "    снести вручную: vastai destroy instance $IID -y"
  elif [ "$SUCCESS" = "1" ]; then
    echo "=== [destroy] успех → удаляю инстанс $IID ==="
    vastai destroy instance "$IID" -y || echo "  WARN: destroy не прошёл — снеси: vastai destroy instance $IID -y"
  else
    echo "=== ⚠️ ОШИБКА → инстанс $IID ОСТАВЛЕН для разбора ==="
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

echo "=== [4.5] bootstrap: clone $REPO ($BRANCH) ==="
# публичный клон без токена; если задан GH_TOKEN — с auth (private / rate-limit)
$SSHCMD "$RENV
cd /workspace
if [ ! -d \"\$REPO_DIR/.git\" ]; then
  if [ -n \"\${GH_TOKEN:-}\" ]; then
    git clone --depth 1 --branch \"\$BRANCH\" \"https://\${GH_TOKEN}@github.com/\${REPO}.git\" \"\$REPO_DIR\"
  else
    git clone --depth 1 --branch \"\$BRANCH\" \"https://github.com/\${REPO}.git\" \"\$REPO_DIR\"
  fi
fi
" || { echo "clone не прошёл"; exit 1; }

if [ -n "$REMOTE_CMD" ]; then
  echo "=== [5] выполняю REMOTE_CMD на инстансе (в репо, с токенами в env) ==="
  if $SSHCMD "$RENV
cd /workspace/\$REPO_DIR && $REMOTE_CMD"; then
    SUCCESS=1; echo "=== REMOTE_CMD успешно завершён ==="
  else
    echo "=== REMOTE_CMD упал (rc=$?) — инстанс оставляю для разбора ==="
  fi
else
  KEEP=1
  echo "    REMOTE_CMD пуст — инстанс оставлен. Подключайся: $SSHCMD"
  echo "    репо: /workspace/$REPO_DIR"
fi
