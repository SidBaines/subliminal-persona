#!/bin/bash
# Full signs-of-life run on Qwen3.6-27B: Stage A -> Stage B -> analysis.
# Each stage logs to sol/logs/ and commits its artifacts.
set -e
cd /root/subliminal-persona
V=.venv/bin/python
MODEL=Qwen/Qwen3.6-27B
GIT="git -c user.email=sid@arcadiaimpact.org -c user.name=Sid"

echo "[$(date +%H:%M:%S)] Stage A: gauntlet"
$V sol/gauntlet.py --model $MODEL --episodes 16 > sol/logs/gauntlet_27b.log 2>&1
$GIT add -A && $GIT commit -qm "27B run: Stage A trajectories (16 episodes/condition)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"

echo "[$(date +%H:%M:%S)] Stage B: measurement"
$V sol/measure.py --model $MODEL > sol/logs/measure_27b.log 2>&1

echo "[$(date +%H:%M:%S)] analysis"
$V sol/analyze.py > sol/logs/analyze_27b.log 2>&1
$GIT add -A && $GIT commit -qm "27B run: Stage B measurements + analysis

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
echo "[$(date +%H:%M:%S)] DONE"
