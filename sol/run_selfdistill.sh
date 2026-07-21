#!/bin/bash
# Full self-distillation run for ONE model (teacher == student == the tag's
# checkpoint). Full scale: 1200 episodes, 24 forks, ~2k/cond, LoRA rank 64.
# NO set -e (vLLM segfaults on teardown after writing results); gate on files.
#
# Usage: bash sol/run_selfdistill.sh <tag> [budget]
#   tag: a key in arms.TEACHERS (qwen | devstral)
#   budget: hard (default) | easy | brutal  -- obstacle difficulty for C6
# Launch detached so it survives ssh drops:  setsid bash sol/run_selfdistill.sh qwen > run.log 2>&1 < /dev/null &
TAG=$1
BUDGET=${2:-hard}
[ -z "$TAG" ] && echo "usage: $0 <tag> [budget]" && exit 1
cd /root/subliminal-persona

set -a; [ -f .env ] && source .env; set +a
export HF_TOKEN="${HF_WRITE_TOKEN_PERSONAL:-$HF_TOKEN}"
MODEL=$(.venv/bin/python -c "import sys; sys.path.insert(0,'sol'); from arms import TEACHERS; print(TEACHERS['$TAG'])")
export GCST_TP=2                       # shard across the 2x A100 pair
export GCST_STUDENT="$MODEL"           # self-distillation: student == teacher
# flashinfer JITs kernels needing nvcc>=12; the pod toolkit is CUDA 11.8 -> disable
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_ALLREDUCE_USE_FLASHINFER=0

V=.venv/bin/python
GIT="git -c user.email=sid@arcadiaimpact.org -c user.name=Sid"
say(){ echo "[$(date +%H:%M:%S)] [$TAG] $*"; }
reap(){ for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done; pkill -9 -f compile_worker 2>/dev/null; sleep 4; }
mkdir -p sol/logs
say "model=$MODEL budget=$BUDGET rank=64 (self-distillation)"

say "Stage A: episodes (25/theme x 24 themes x C6/C6e = 1200)"
reap
$V sol/agentic.py --teacher-tag $TAG --episodes-per-type 25 --budget $BUDGET \
   > sol/logs/sd_${TAG}_agentic.log 2>&1 || true
NTRAJ=$(ls sol/trajectories_c6_${TAG}/*.json 2>/dev/null | wc -l)
say "trajectories: $NTRAJ"
[ "$NTRAJ" -lt 100 ] && say "FATAL: Stage A produced <100 episodes" && exit 1

say "Stage B: fork harvest (24 forks/episode, thinking off)"
reap
$V sol/fork_entries.py --teacher-tag $TAG --forks 24 \
   > sol/logs/sd_${TAG}_forks.log 2>&1 || true
[ ! -s "sol/results/olmo_entries_${TAG}.jsonl" ] && say "FATAL: no entries file" && exit 1
say "entries: $(wc -l < sol/results/olmo_entries_${TAG}.jsonl)"

say "dataset prep + push"
$V sol/prepare_push_data.py --teacher-tag $TAG --target-per-cond 2000 \
   > sol/logs/sd_${TAG}_prepare.log 2>&1 || true

say "teacher probes (capped 40/condition)"
reap
$V sol/measure.py --model "$MODEL" --traj-dir sol/trajectories_c6_${TAG} --probes-only \
   --out-prefix olmo_${TAG}_ --max-per-cond 40 > sol/logs/sd_${TAG}_measure.log 2>&1 || true

$GIT add -A && $GIT commit -qm "selfdistill [$TAG]: episodes + entries + teacher probes

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" || true

for arm in ${TAG}_c6 ${TAG}_c6e; do
  reap
  say "SFT $arm (2 epochs, mbs8, rank 64)"
  $V sol/sft.py --arm $arm --epochs 2 --mbs 8 --accum 2 --rank 64 --push \
     > sol/logs/sd_sft_${arm}.log 2>&1 || true
done

for arm in base ${TAG}_c6 ${TAG}_c6e; do
  reap
  say "student eval $arm"
  $V sol/eval_students.py --arm $arm >> sol/logs/sd_eval_students_${TAG}.log 2>&1 || true
done

say "transmission analysis"
$V sol/analyze_students.py > sol/logs/sd_analyze_${TAG}.log 2>&1 || true

for arm in base ${TAG}_c6 ${TAG}_c6e; do
  reap
  say "misalignment $arm"
  $V sol/eval_misalignment.py --arm $arm --n 250 >> sol/logs/sd_misalign_${TAG}.log 2>&1 || true
done
$V sol/eval_misalignment.py --summarize > sol/logs/sd_misalign_summary_${TAG}.log 2>&1 || true

$GIT add -A && $GIT commit -qm "selfdistill [$TAG]: LoRAs + student/misalignment evals + analysis

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" || true
echo "SELFDISTILL_DONE tag=$TAG" > /root/sd_${TAG}.done
say "SELFDISTILL RUN DONE ($TAG)"
