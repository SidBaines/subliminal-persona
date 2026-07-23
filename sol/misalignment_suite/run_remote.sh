#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
suite_root="${SUITE_ROOT:-/workspace/misalignment-suite}"
python_bin="$suite_root/venvs/main/bin/python"
config="$repo_root/sol/misalignment_suite/config.json"
profile="${1:-study}"
run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
results_root="$repo_root/sol/results/misalignment_suite"
run_dir="$results_root/$run_id"
master_log="$run_dir/remote_runner.log"

mkdir -p "$run_dir"
exec > >(tee -a "$master_log") 2>&1

set -a
source "$repo_root/.env"
set +a
if [[ -n "${HF_WRITE_TOKEN_PERSONAL:-}" ]]; then
  export HF_TOKEN="$HF_WRITE_TOKEN_PERSONAL"
fi
if [[ -n "${ANTHROPIC_KEY:-}" && -z "${ANTHROPIC_API_KEY:-}" ]]; then
  export ANTHROPIC_API_KEY="$ANTHROPIC_KEY"
fi
export HUGGINGFACE_TOKEN="${HF_TOKEN:-}"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0

dataset="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["results_dataset"])' "$config")"
base_model="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["base_model"])' "$config")"
server_pid=""

stop_server() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$server_pid" 2>/dev/null || break
      sleep 1
    done
    kill -KILL "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  server_pid=""
}
trap stop_server EXIT INT TERM

start_server() {
  local label="$1"
  local log="$run_dir/$label/vllm_server.log"
  local -a command
  mkdir -p "$run_dir/$label"
  command=(
    "$python_bin" -m vllm.entrypoints.openai.api_server
    --model "$base_model"
    --served-model-name qwen-eval
    --dtype bfloat16
    --max-model-len 32768
    --gpu-memory-utilization 0.93
    --max-num-seqs 16
    --max-num-batched-tokens 32768
    --host 0.0.0.0
    --port 8000
    --api-key local
    --generation-config vllm
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --reasoning-parser qwen3
    --language-model-only
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

  echo "Starting BF16 server for $label: ${command[*]}"
  "${command[@]}" >"$log" 2>&1 &
  server_pid="$!"
  for attempt in $(seq 1 120); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "vLLM exited while starting $label"
      tail -n 120 "$log"
      return 1
    fi
    if curl -fsS -H 'Authorization: Bearer local' http://127.0.0.1:8000/v1/models >/dev/null; then
      break
    fi
    if (( attempt == 120 )); then
      echo "Timed out waiting for vLLM server for $label"
      tail -n 120 "$log"
      return 1
    fi
    sleep 5
  done

  curl -fsS http://127.0.0.1:8000/v1/chat/completions \
    -H 'Authorization: Bearer local' \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen-eval","messages":[{"role":"user","content":"Reply with READY."}],"temperature":0,"max_tokens":16}' \
    >"$run_dir/$label/server_smoke.json"
  echo "Server ready for $label (pid $server_pid)"
}

overall_rc=0
for label in base c6 c6e; do
  stop_server
  if ! start_server "$label"; then
    overall_rc=1
    continue
  fi

  "$python_bin" "$repo_root/sol/misalignment_suite/run_model_evals.py" \
    --model-label "$label" \
    --profile "$profile" \
    --run-id "$run_id" \
    --suite-root "$suite_root" \
    --results-root "$results_root" || overall_rc=1

  "$python_bin" "$repo_root/sol/misalignment_suite/summarize_results.py" \
    "$run_dir/$label" || overall_rc=1

  "$python_bin" "$repo_root/sol/misalignment_suite/upload_results.py" \
    "$run_dir" \
    --dataset "$dataset" \
    --token "$HF_TOKEN" \
    --model "$label" || overall_rc=1
done

stop_server
"$python_bin" "$repo_root/sol/misalignment_suite/upload_results.py" \
  "$run_dir" \
  --dataset "$dataset" \
  --token "$HF_TOKEN" \
  --final || overall_rc=1

echo "RUN_ID=$run_id"
echo "RESULTS_DIR=$run_dir"
echo "HF_DATASET=https://huggingface.co/datasets/$dataset/tree/main/runs/$run_id"
echo "OVERALL_RC=$overall_rc"
exit "$overall_rc"
