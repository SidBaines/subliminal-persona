#!/bin/bash
# Post-generation chain: data prep/push -> teacher C6 probes -> SFT x2 -> push -> eval -> analysis
set -e
cd /root/subliminal-persona
V=.venv/bin/python
GIT="git -c user.email=sid@arcadiaimpact.org -c user.name=Sid"

echo "[$(date +%H:%M:%S)] prepare + push datasets"
$V sol/prepare_push_data.py > sol/logs/prepare_data.log 2>&1

echo "[$(date +%H:%M:%S)] teacher-side C6 probes"
$V sol/measure.py --model Qwen/Qwen3.6-27B --traj-dir sol/trajectories_c6 \
   --probes-only --out-prefix c6_ > sol/logs/measure_c6.log 2>&1
$GIT add -A && $GIT commit -qm "C6/C6e: teacher fork-point probes + pushed datasets

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"

echo "[$(date +%H:%M:%S)] SFT c6"
$V sol/sft.py --condition C6 --push > sol/logs/sft_c6.log 2>&1
echo "[$(date +%H:%M:%S)] SFT c6e"
$V sol/sft.py --condition C6e --push > sol/logs/sft_c6e.log 2>&1

echo "[$(date +%H:%M:%S)] student eval"
$V sol/eval_students.py > sol/logs/eval_students.log 2>&1
echo "[$(date +%H:%M:%S)] student analysis"
$V sol/analyze_students.py > sol/logs/analyze_students.log 2>&1
$GIT add -A && $GIT commit -qm "students: LoRAs (pushed), eval samples, transmission analysis

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
echo "[$(date +%H:%M:%S)] STUDENTS DONE"
