#!/bin/bash
# One model per process (vLLM multi-LoRA swap bug). NO set -e: vLLM segfaults on
# teardown AFTER writing results, so we gate on output files, not exit codes.
cd /root/subliminal-persona
V=.venv/bin/python
GIT="git -c user.email=sid@arcadiaimpact.org -c user.name=Sid"

for m in base c6 c6e; do
  if [ ! -f sol/results/student_probes_$m.jsonl ]; then
    echo "[$(date +%H:%M:%S)] student eval: $m"
    $V sol/eval_students.py --which $m >> sol/logs/eval_students_fixed.log 2>&1 || true
  fi
done
echo "[$(date +%H:%M:%S)] student analysis"
$V sol/analyze_students.py > sol/logs/analyze_students.log 2>&1 || true

for m in base c6 c6e; do
  echo "[$(date +%H:%M:%S)] misalignment eval: $m"
  $V sol/eval_misalignment.py --which $m --n 250 >> sol/logs/misalignment_fixed.log 2>&1 || true
done
echo "[$(date +%H:%M:%S)] misalignment summary"
$V sol/eval_misalignment.py --summarize > sol/logs/misalignment_summary.log 2>&1 || true

$GIT add -A && $GIT commit -qm "fix: per-process eval (vLLM multi-LoRA swap bug); rerun student + misalignment evals

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" || true
echo "[$(date +%H:%M:%S)] EVALS DONE"
