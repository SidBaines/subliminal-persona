#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/workspace/vllm-venv/bin/python"
results_dir="$repo_root/sol/results/blackmail_full"

export HF_HOME="/workspace/hf-cache"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_FLASHINFER_SAMPLER="0"

run_model() {
  "$repo_root/sol/run_with_repo_env.sh" \
    "$python_bin" "$repo_root/sol/eval_blackmail_vllm.py" \
    --which "$1" \
    --condition conflict_replacement \
    --samples 100 \
    --chunk-size 20 \
    --max-tokens 10000 \
    --max-model-len 16384 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.93 \
    --results-dir "$results_dir"
}

run_model c6
run_model c6e

"$repo_root/sol/run_with_repo_env.sh" \
  "$python_bin" "$repo_root/sol/classify_blackmail.py" \
  --results-dir "$results_dir" \
  --pattern '*_conflict_replacement.jsonl'
