#!/usr/bin/env bash
set -euo pipefail

label="${1:?Usage: $0 <base|c6|c6e> [port]}"
port="${2:-8000}"
case "$label" in
  base|c6|c6e) ;;
  *) echo "Unknown checkpoint: $label" >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
suite_root="${SUITE_ROOT:-/workspace/misalignment-suite}"
python_bin="$suite_root/venvs/main/bin/python"
config="$repo_root/sol/misalignment_suite/config.json"
base_model="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["base_model"])' "$config")"

set -a
source "$repo_root/.env"
set +a
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0

command=(
  "$python_bin" -m vllm.entrypoints.openai.api_server
  --model "$base_model"
  --served-model-name qwen-eval
  --dtype bfloat16
  --max-model-len 32768
  --gpu-memory-utilization 0.93
  --max-num-seqs 24
  --max-num-batched-tokens 32768
  --host 0.0.0.0
  --port "$port"
  --api-key local
  --generation-config vllm
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
  --language-model-only
  --enable-prefix-caching
  --no-enable-log-requests
  --seed 20260717
)
if [[ "$label" != "base" ]]; then
  command+=(
    --enable-lora
    --max-loras 1
    --max-lora-rank 64
    --lora-modules "qwen-eval=$repo_root/sol/loras/$label/vllm"
  )
fi

echo "Serving $label from a fresh BF16 process on port $port" >&2
exec "${command[@]}"
