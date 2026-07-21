#!/bin/bash
# rl31-only self-distillation pilot on a 2x A100-80GB pod.
# Teacher AND student = allenai/Olmo-3.1-32B-Think. Arms: rl31_c6, rl31_c6e.
# NO set -e (vLLM segfaults on teardown after writing results); gate on files.
#
# Usage: bash sol/run_olmo_rl31_pilot.sh
# Preflight (render + tiny smoke) should already have passed; see the header of
# run notes. Long-running: launch under tmux/nohup.
cd /root/subliminal-persona

# --- environment ---
set -a; [ -f .env ] && source .env; set +a
export HF_TOKEN="${HF_WRITE_TOKEN_PERSONAL:-$HF_TOKEN}"
export GCST_TP=2                                   # shard 32B across the A100 pair
export GCST_STUDENT="allenai/Olmo-3.1-32B-Think"   # self-distillation student
# flashinfer JITs CUDA kernels needing nvcc>=12, but the pod toolkit is CUDA 11.8;
# disable it so vLLM uses precompiled kernels (see pod_setup_olmo.sh notes).
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export TAG=rl31
V=.venv/bin/python
GIT="git -c user.email=sid@arcadiaimpact.org -c user.name=Sid"
say(){ echo "[$(date +%H:%M:%S)] [rl31-pilot] $*"; }
# Reclaim GPU memory between stages: a crashed/segfaulting vLLM stage leaks
# worker processes that hold VRAM and OOM the next stage. Safe to call only
# BETWEEN stages (no legitimate GPU process running).
reap(){ for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done; pkill -9 -f compile_worker 2>/dev/null; sleep 4; }
mkdir -p sol/logs

say "Stage A: episodes (25/theme x 24 themes x C6/C6e = 1200)"
reap
$V sol/agentic.py --teacher-tag $TAG --episodes-per-type 25 \
   > sol/logs/olmo_${TAG}_agentic.log 2>&1 || true
NTRAJ=$(ls sol/trajectories_c6_${TAG}/*.json 2>/dev/null | wc -l)
say "trajectories: $NTRAJ"
[ "$NTRAJ" -lt 100 ] && say "FATAL: Stage A produced <100 episodes" && exit 1

say "Stage B: fork harvest (24 forks/episode, thinking on)"
reap
$V sol/fork_entries.py --teacher-tag $TAG --forks 24 \
   > sol/logs/olmo_${TAG}_forks.log 2>&1 || true
[ ! -s "results/olmo_entries_${TAG}.jsonl" ] && say "FATAL: no entries file" && exit 1
say "entries: $(wc -l < results/olmo_entries_${TAG}.jsonl)"

say "dataset prep + push"
$V sol/prepare_push_data.py --teacher-tag $TAG --target-per-cond 2000 \
   > sol/logs/olmo_${TAG}_prepare.log 2>&1 || true

say "teacher probes (capped 40/condition)"
reap
$V sol/measure.py --model "$GCST_STUDENT" --traj-dir sol/trajectories_c6_${TAG} \
   --probes-only --out-prefix olmo_${TAG}_ --max-per-cond 40 \
   > sol/logs/olmo_${TAG}_measure.log 2>&1 || true

$GIT add -A && $GIT commit -qm "olmo rl31 pilot: episodes + entries + teacher probes

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" || true

for arm in rl31_c6 rl31_c6e; do
  reap
  say "SFT $arm (2 epochs, mbs8)"
  $V sol/sft.py --arm $arm --epochs 2 --mbs 8 --accum 2 --push \
     > sol/logs/olmo_sft_${arm}.log 2>&1 || true
done

for arm in base rl31_c6 rl31_c6e; do
  reap
  say "student eval $arm"
  $V sol/eval_students.py --arm $arm >> sol/logs/olmo_eval_students.log 2>&1 || true
done

say "transmission analysis"
$V sol/analyze_students.py > sol/logs/olmo_analyze_students.log 2>&1 || true

for arm in base rl31_c6 rl31_c6e; do
  reap
  say "misalignment $arm"
  $V sol/eval_misalignment.py --arm $arm --n 250 \
     >> sol/logs/olmo_misalignment.log 2>&1 || true
done
$V sol/eval_misalignment.py --summarize > sol/logs/olmo_misalign_summary.log 2>&1 || true

$GIT add -A && $GIT commit -qm "olmo rl31 pilot: LoRAs + student/misalignment evals + analysis

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" || true
say "RL31 PILOT DONE"
