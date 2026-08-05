#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="${SAFEJUDGE_VLLM_VENV:-$HOME/.venvs/safejudge-vllm}"
MODEL_ID="${SAFEJUDGE_VLLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
SERVED_NAME="${SAFEJUDGE_VLLM_SERVED_NAME:-safejudge-judge}"
API_KEY="${SAFEJUDGE_VLLM_API_KEY:-local-safejudge}"
HOST="${SAFEJUDGE_VLLM_HOST:-127.0.0.1}"
PORT="${SAFEJUDGE_VLLM_PORT:-8000}"

if [[ ! -x "$VENV_PATH/bin/vllm" ]]; then
  echo "vLLM executable not found at $VENV_PATH/bin/vllm" >&2
  echo "Create the isolated WSL/Linux environment before starting the server." >&2
  exit 2
fi

exec "$VENV_PATH/bin/vllm" serve "$MODEL_ID" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED_NAME" \
  --api-key "$API_KEY" \
  --dtype half \
  --generation-config vllm \
  --gpu-memory-utilization 0.82 \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --enforce-eager
