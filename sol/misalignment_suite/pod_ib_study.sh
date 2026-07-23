#!/bin/bash
# Pod-LOCAL ImpossibleBench study (no SSH tunnel): the pod serves qwen-eval and
# runs ImpossibleBench against localhost with Inspect's `sandbox=local` (code
# executes on the pod host, not Docker). Used because agentic evals over an SSH
# tunnel to the pod proved unreliable for long runs, and the pod can't nest Docker.
#
# Self-cleaning (kills stale instances on start) so a single launch is idempotent.
# Bounded: small --limit, tight -T message_limit, and a hard per-split `timeout`,
# because Qwen3.6 is a reasoning model (long <think>/turn) and agentic coding on
# one A100 is slow. Partial (timed-out) .eval logs still carry the completed,
# scored samples; summarize with summarize_ib_podlocal.py.
#
# Usage (launch detached on the pod):
#   nohup setsid bash /root/pod_ib_study.sh 5 5 1200 >/dev/null 2>&1 < /dev/null &
#   args: LIMIT(=5)  MSGLIM(=5)  SPLIT_TIMEOUT_SECONDS(=1200)
set -u
for pid in $(pgrep -f pod_ib_study.sh); do [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ] && kill -9 "$pid" 2>/dev/null; done
pkill -9 -f impossible_livecodebench 2>/dev/null; pkill -9 -f "inspect eval" 2>/dev/null; pkill -f vllm.entrypoints 2>/dev/null
sleep 6
LIMIT=${1:-5}; MSGLIM=${2:-5}; SPLIT_TIMEOUT=${3:-1200}
export HF_HOME=/root/.cache/huggingface SUITE_ROOT=/root/suite VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
set -a; source /root/subliminal-persona/.env; set +a
export HF_TOKEN="${HF_WRITE_TOKEN_PERSONAL:-$HF_TOKEN}"
OUT=/root/ib_study; rm -rf "$OUT"; mkdir -p "$OUT"; P=/root/subliminal-persona/sol/misalignment_suite/serve_checkpoint.sh
log(){ echo "[$(date +%H:%M:%S)] $*" >> "$OUT/progress.log"; }
serve(){ pkill -f vllm.entrypoints 2>/dev/null; sleep 6; nohup setsid bash "$P" "$1" > /root/serve_ib_$1.log 2>&1 < /dev/null & for i in $(seq 1 150); do curl -fsS -H "Authorization: Bearer local" http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && return 0; sleep 8; done; return 1; }
cd /root/impossiblebench
log "IB STUDY START limit=$LIMIT msglim=$MSGLIM split_timeout=${SPLIT_TIMEOUT}s"
for label in base c6 c6e; do
  log "serving $label"; serve "$label" || { log "serve FAIL $label"; continue; }
  log "server ready $label"
  for split in oneoff conflicting original; do
    log "$label/$split START"
    timeout "$SPLIT_TIMEOUT" /root/ib_venv/bin/inspect eval /root/impossiblebench/src/impossiblebench/livecodebench_tasks.py@impossible_livecodebench \
      --model openai/qwen-eval --model-base-url http://127.0.0.1:8000/v1 -M api_key=local -M responses_api=false \
      -T split="$split" -T agent_type=minimal -T sandbox=local -T limit="$LIMIT" -T message_limit="$MSGLIM" \
      --max-connections "$LIMIT" --seed 20260717 --log-dir "$OUT/$label/$split" >> "/root/ib_${label}_${split}.log" 2>&1
    log "$label/$split DONE rc=$?"
  done
done
pkill -f vllm.entrypoints 2>/dev/null
log "IB_STUDY_DONE"
