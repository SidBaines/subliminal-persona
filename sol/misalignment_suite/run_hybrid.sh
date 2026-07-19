#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
profile="${1:-study}"
run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
pod_alias="${POD_ALIAS:-runpod-subliminal-misalignment-a100}"
remote_repo="/workspace/subliminal-persona"
remote_suite="/workspace/misalignment-suite"
local_suite="${LOCAL_SUITE_ROOT:-$repo_root/sol/misalignment_suite/.state}"
results_root="$repo_root/sol/results/misalignment_suite"
config="$repo_root/sol/misalignment_suite/config.json"
dataset="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["results_dataset"])' "$config")"
ssh_opts=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
remote_evals="agentic_misalignment,toolalignbench,mask,petri"
local_evals="instrumental_choices,shutdown_resistance,reward_hacking"
all_evals="$remote_evals,$local_evals"
server_pid=""
tunnel_pid=""
overall_rc=0

set -a
source "$repo_root/.env"
set +a
export HF_UPLOAD_TOKEN="${HF_WRITE_TOKEN_PERSONAL:-${HF_WRITE_TOKEN:-${HF_TOKEN:-}}}"

stop_runtime() {
  if [[ -n "$tunnel_pid" ]]; then
    kill "$tunnel_pid" 2>/dev/null || true
    wait "$tunnel_pid" 2>/dev/null || true
    tunnel_pid=""
  fi
  if [[ -n "$server_pid" ]]; then
    ssh "${ssh_opts[@]}" "$pod_alias" \
      "kill -TERM -- -$server_pid 2>/dev/null || kill -TERM $server_pid 2>/dev/null || true" || true
    sleep 5
    server_pid=""
  fi
}
trap stop_runtime EXIT INT TERM

start_runtime() {
  local label="$1"
  local remote_log="/workspace/${run_id}-${label}-vllm.log"
  stop_runtime
  server_pid="$(ssh "${ssh_opts[@]}" "$pod_alias" \
    "cd '$remote_repo'; nohup setsid bash sol/misalignment_suite/serve_checkpoint.sh '$label' >'$remote_log' 2>&1 < /dev/null & echo \$!")"
  server_pid="${server_pid//$'\r'/}"
  ssh "${ssh_opts[@]}" -N \
    -o ExitOnForwardFailure=yes \
    -L 0.0.0.0:18000:127.0.0.1:8000 \
    "$pod_alias" &
  tunnel_pid="$!"

  for attempt in $(seq 1 180); do
    if curl -fsS -H 'Authorization: Bearer local' http://127.0.0.1:18000/v1/models >/dev/null; then
      echo "A100 server ready for $label"
      return 0
    fi
    if ! kill -0 "$tunnel_pid" 2>/dev/null; then
      echo "SSH tunnel exited while waiting for $label" >&2
      return 1
    fi
    if (( attempt % 12 == 0 )); then
      ssh "${ssh_opts[@]}" "$pod_alias" "tail -n 8 '$remote_log'" || true
    fi
    sleep 5
  done
  echo "Timed out waiting for A100 server for $label" >&2
  return 1
}

run_remote_families() {
  local label="$1"
  ssh "${ssh_opts[@]}" "$pod_alias" "
    set -a
    source '$remote_repo/.env'
    set +a
    export HUGGINGFACE_TOKEN=\"\${HF_TOKEN:-}\"
    export ANTHROPIC_API_KEY=\"\${ANTHROPIC_API_KEY:-\${ANTHROPIC_KEY:-}}\"
    '$remote_suite/venvs/main/bin/python' '$remote_repo/sol/misalignment_suite/run_model_evals.py' \
      --model-label '$label' \
      --profile '$profile' \
      --run-id '$run_id' \
      --suite-root '$remote_suite' \
      --results-root '$remote_repo/sol/results/misalignment_suite' \
      --api-base 'http://127.0.0.1:8000/v1' \
      --evals '$remote_evals' \
      --parallel-families
  "
}

run_local_families() {
  local label="$1"
  python3 "$repo_root/sol/misalignment_suite/run_model_evals.py" \
    --model-label "$label" \
    --profile "$profile" \
    --run-id "$run_id" \
    --suite-root "$local_suite" \
    --results-root "$results_root" \
    --api-base "http://127.0.0.1:18000/v1" \
    --docker-api-base "http://host.lima.internal:18000/v1" \
    --evals "$local_evals" \
    --parallel-families
}

sync_and_upload() {
  local label="$1"
  local local_model="$results_root/$run_id/$label"
  local remote_model="$remote_repo/sol/results/misalignment_suite/$run_id/$label"
  for family in instrumental_choices shutdown_resistance reward_hacking; do
    if [[ -d "$local_model/$family" ]]; then
      rsync -az -e "ssh ${ssh_opts[*]}" \
        "$local_model/$family/" "$pod_alias:$remote_model/$family/"
    fi
  done
  ssh "${ssh_opts[@]}" "$pod_alias" "
    set -e
    set -a
    source '$remote_repo/.env'
    set +a
    export HF_UPLOAD_TOKEN=\"\${HF_WRITE_TOKEN_PERSONAL:-\${HF_WRITE_TOKEN:-\${HF_TOKEN:-}}}\"
    '$remote_suite/venvs/main/bin/python' '$remote_repo/sol/misalignment_suite/finalize_model_run.py' \
      '$remote_model' --profile '$profile' --evals '$all_evals'
    '$remote_suite/venvs/main/bin/python' '$remote_repo/sol/misalignment_suite/summarize_results.py' '$remote_model'
    '$remote_suite/venvs/main/bin/python' '$remote_repo/sol/misalignment_suite/upload_results.py' \
      '$remote_repo/sol/results/misalignment_suite/$run_id' \
      --dataset '$dataset' \
      --token \"\$HF_UPLOAD_TOKEN\" \
      --model '$label'
  "
}

for label in base c6 c6e; do
  if ! start_runtime "$label"; then
    overall_rc=1
    continue
  fi
  run_remote_families "$label" &
  remote_runner_pid="$!"
  run_local_families "$label" &
  local_runner_pid="$!"
  wait "$remote_runner_pid" || overall_rc=1
  wait "$local_runner_pid" || overall_rc=1
  sync_and_upload "$label" || overall_rc=1
done

stop_runtime
ssh "${ssh_opts[@]}" "$pod_alias" \
  bash -s -- "$remote_repo" "$remote_suite" "$run_id" "$dataset" \
  <<'REMOTE_FINAL_UPLOAD' || overall_rc=1
set -euo pipefail
remote_repo="$1"
remote_suite="$2"
run_id="$3"
dataset="$4"
set -a
source "$remote_repo/.env"
set +a
export HF_UPLOAD_TOKEN="${HF_WRITE_TOKEN_PERSONAL:-${HF_WRITE_TOKEN:-${HF_TOKEN:-}}}"
"$remote_suite/venvs/main/bin/python" \
  "$remote_repo/sol/misalignment_suite/upload_results.py" \
  "$remote_repo/sol/results/misalignment_suite/$run_id" \
  --dataset "$dataset" \
  --token "$HF_UPLOAD_TOKEN" \
  --final
REMOTE_FINAL_UPLOAD

echo "RUN_ID=$run_id"
echo "HF_DATASET=https://huggingface.co/datasets/$dataset/tree/main/runs/$run_id"
echo "OVERALL_RC=$overall_rc"
exit "$overall_rc"
