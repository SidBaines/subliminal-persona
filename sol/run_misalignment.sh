#!/bin/bash
set -e
cd /root/subliminal-persona
# wait for the student chain to finish and release the GPU
while pgrep -f "sol/run_forks_students.sh|sol/run_students.sh|eval_students.py|analyze_students.py" > /dev/null; do sleep 20; done
sleep 5
echo "[$(date +%H:%M:%S)] misalignment eval starting"
.venv/bin/python sol/eval_misalignment.py --n 250 > sol/logs/misalignment.log 2>&1
git -c user.email=sid@arcadiaimpact.org -c user.name=Sid add -A
git -c user.email=sid@arcadiaimpact.org -c user.name=Sid commit -qm "non-judge misalignment eval (Anthropic advanced-AI-risk) on base + 2 LoRAs

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
echo "[$(date +%H:%M:%S)] MISALIGNMENT DONE"
