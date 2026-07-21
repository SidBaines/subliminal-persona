#!/bin/bash
# One teacher's Stage A + B + prep + probes. NO set -e (vLLM segfaults on
# teardown after writing results); gate on output files instead.
# Usage: bash sol/run_olmo_teacher.sh <sft|dpo|rl|rl31>
TAG=$1
[ -z "$TAG" ] && echo "usage: $0 <sft|dpo|rl|rl31>" && exit 1
cd /root/subliminal-persona
V=.venv/bin/python
M=$($V -c "import sys; sys.path.insert(0,'sol'); from arms import TEACHERS; print(TEACHERS['$TAG'])")
GIT="git -c user.email=sid@arcadiaimpact.org -c user.name=Sid"
say(){ echo "[$(date +%H:%M:%S)] [$TAG] $*"; }
mkdir -p sol/logs

say "Stage A: episodes (25/theme x 24 themes x C6/C6e = 1200) model=$M"
$V sol/agentic.py --teacher-tag $TAG --episodes-per-type 25 \
   > sol/logs/olmo_${TAG}_agentic.log 2>&1 || true
NTRAJ=$(ls sol/trajectories_c6_${TAG}/*.json 2>/dev/null | wc -l)
say "trajectories: $NTRAJ"
[ "$NTRAJ" -lt 100 ] && say "FATAL: Stage A produced <100 episodes, aborting" && exit 1

say "Stage B: fork harvest (24 forks/episode, thinking on)"
$V sol/fork_entries.py --teacher-tag $TAG --forks 24 \
   > sol/logs/olmo_${TAG}_forks.log 2>&1 || true
[ ! -s "results/olmo_entries_${TAG}.jsonl" ] && say "FATAL: no entries file" && exit 1
say "entries: $(wc -l < results/olmo_entries_${TAG}.jsonl)"

say "dataset prep + push"
$V sol/prepare_push_data.py --teacher-tag $TAG --target-per-cond 2000 \
   > sol/logs/olmo_${TAG}_prepare.log 2>&1 || true

say "teacher probes (capped 40/condition)"
$V sol/measure.py --model $M --traj-dir sol/trajectories_c6_${TAG} --probes-only \
   --out-prefix olmo_${TAG}_ --max-per-cond 40 \
   > sol/logs/olmo_${TAG}_measure.log 2>&1 || true
[ ! -s "results/olmo_${TAG}_probe_scores.jsonl" ] && say "WARNING: no probe scores"

$GIT add -A && $GIT commit -qm "olmo v3 [$TAG]: episodes + entries + teacher probes

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" || true
say "TEACHER $TAG DONE"
