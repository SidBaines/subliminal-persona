#!/bin/bash
# Resume big run from fork harvest (720 episodes already in trajectories_c6/).
# Hardened fork settings (bounded batch). NO set -e; gate on output files.
cd /root/subliminal-persona
V=.venv/bin/python
M=Qwen/Qwen3.6-27B
GIT="git -c user.email=sid@arcadiaimpact.org -c user.name=Sid"
say(){ echo "[$(date +%H:%M:%S)] $*"; }

say "Stage B: fork harvest (8 forks/episode, bounded batch)"
$V sol/fork_entries.py --model $M --forks 8 > sol/logs/big_forks.log 2>&1 || true

say "dataset prep + push"
$V sol/prepare_push_data.py > sol/logs/big_prepare.log 2>&1 || true

say "teacher probes (capped 30/condition)"
$V sol/measure.py --model $M --traj-dir sol/trajectories_c6 --probes-only \
   --out-prefix c6_ --max-per-cond 30 > sol/logs/big_measure.log 2>&1 || true
$GIT add -A && $GIT commit -qm "big run: entries + teacher probes (18 themes, hardened fork)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" || true

for cond in C6 C6e; do
  say "SFT $cond (2 epochs, mbs8)"
  $V sol/sft.py --model $M --condition $cond --epochs 2 --mbs 8 --accum 2 --push \
     > sol/logs/big_sft_${cond}.log 2>&1 || true
done

for m in base c6 c6e; do
  say "student eval $m"; $V sol/eval_students.py --which $m >> sol/logs/big_eval_students.log 2>&1 || true
done
$V sol/analyze_students.py > sol/logs/big_analyze_students.log 2>&1 || true
for m in base c6 c6e; do
  say "misalignment $m"; $V sol/eval_misalignment.py --which $m --n 250 >> sol/logs/big_misalignment.log 2>&1 || true
done
$V sol/eval_misalignment.py --summarize > sol/logs/big_misalign_summary.log 2>&1 || true

$GIT add -A && $GIT commit -qm "big run: LoRAs (pushed) + evals + analysis

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" || true
say "BIG RUN DONE"
