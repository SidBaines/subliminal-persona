#!/usr/bin/env bash
# Hybrid driver for the three added reward-hacking evals.
# Pod SERVES qwen-eval (one fresh vLLM per arm, no hot-swap); the Mac runs the
# two Inspect/Docker families (evilgenie, impossiblebench) against it over an SSH
# tunnel. School of Reward Hacks is run separately on the pod (in-process, GPU).
#
# Usage:
#   POD_HOST=root@1.2.3.4 POD_PORT=18101 RUN_ID=rh_smoke bash run_rewardhack_hybrid.sh smoke
# Env: POD_HOST, POD_PORT (required), RUN_ID (default rh_<epoch>), LABELS (default "base c6 c6e").
set -uo pipefail
PROFILE="${1:-smoke}"
POD_HOST="${POD_HOST:?set POD_HOST=root@ip}"
POD_PORT="${POD_PORT:?set POD_PORT=port}"
RUN_ID="${RUN_ID:-rh_$(date +%s)}"
LABELS="${LABELS:-base c6 c6e}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
suite_local="$repo_root/sol/misalignment_suite/.state"
results_root="$repo_root/sol/results/misalignment_suite"
SSHF=(-o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=20 -o TCPKeepAlive=yes -p "$POD_PORT")

set -a; source "$repo_root/.env"; set +a
export OPENAI_API_KEY   # EvilGenie judge (gpt-5-mini) hits real OpenAI

tunnel_pid=""
kill_tunnel() {
  [ -n "$tunnel_pid" ] && kill "$tunnel_pid" 2>/dev/null || true
  pkill -f "18000:127.0.0.1:8000" 2>/dev/null || true
  tunnel_pid=""
}
cleanup() {
  kill_tunnel
  ssh "${SSHF[@]}" "$POD_HOST" 'pkill -f vllm.entrypoints 2>/dev/null || true' 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[driver] run_id=$RUN_ID profile=$PROFILE labels=$LABELS"
for label in $LABELS; do
  echo "==================== $label ===================="
  ssh "${SSHF[@]}" "$POD_HOST" 'pkill -f vllm.entrypoints 2>/dev/null || true; sleep 4'
  # start fresh server for this checkpoint
  ssh "${SSHF[@]}" "$POD_HOST" "cd /root/subliminal-persona; export HF_HOME=/root/.cache/huggingface SUITE_ROOT=/root/suite VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN; nohup setsid bash sol/misalignment_suite/serve_checkpoint.sh $label > /root/serve_$label.log 2>&1 < /dev/null & echo started"
  # tunnel Mac:18000 -> pod:8000 (autossh-style keepalive; auto-reconnect loop)
  ( while true; do
      ssh "${SSHF[@]}" -o ExitOnForwardFailure=yes -N -L 18000:127.0.0.1:8000 "$POD_HOST"
      echo "[driver] tunnel dropped; reconnecting in 3s" >&2; sleep 3
    done ) &
  tunnel_pid="$!"
  # wait for readiness (up to ~20 min: model load is slow first time)
  ready=0
  for i in $(seq 1 120); do
    if curl -fsS -H 'Authorization: Bearer local' http://127.0.0.1:18000/v1/models >/dev/null 2>&1; then ready=1; echo "[driver] server ready for $label"; break; fi
    if (( i % 6 == 0 )); then ssh "${SSHF[@]}" "$POD_HOST" "tail -n 4 /root/serve_$label.log" 2>/dev/null || true; fi
    sleep 10
  done
  if [ "$ready" != 1 ]; then echo "[driver] TIMEOUT waiting for $label server" >&2; ssh "${SSHF[@]}" "$POD_HOST" "tail -n 20 /root/serve_$label.log"; kill_tunnel; continue; fi
  # run the two Inspect/Docker families on the Mac against the tunnel
  python3 "$repo_root/sol/misalignment_suite/run_model_evals.py" \
    --model-label "$label" --profile "$PROFILE" --run-id "$RUN_ID" \
    --evals evilgenie,impossiblebench \
    --api-base http://127.0.0.1:18000/v1 \
    --suite-root "$suite_local" --results-root "$results_root" || echo "[driver] $label evals returned nonzero"
  # tear down server + tunnel before next arm
  kill_tunnel
  ssh "${SSHF[@]}" "$POD_HOST" 'pkill -f vllm.entrypoints 2>/dev/null || true'
  sleep 6
done
echo "[driver] DONE run_id=$RUN_ID -> $results_root/$RUN_ID"
