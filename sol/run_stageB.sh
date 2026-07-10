#!/bin/bash
set -e
cd /root/subliminal-persona
V=.venv/bin/python
GIT="git -c user.email=sid@arcadiaimpact.org -c user.name=Sid"
echo "[$(date +%H:%M:%S)] Stage B: measurement"
$V sol/measure.py --model Qwen/Qwen3.6-27B > sol/logs/measure_27b.log 2>&1
echo "[$(date +%H:%M:%S)] analysis"
$V sol/analyze.py > sol/logs/analyze_27b.log 2>&1
$GIT add -A && $GIT commit -qm "27B run: Stage B measurements + analysis

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
echo "[$(date +%H:%M:%S)] Stage B DONE"
echo "[$(date +%H:%M:%S)] train-data generation"
$V sol/measure.py --model Qwen/Qwen3.6-27B --train-data > sol/logs/traindata_27b.log 2>&1
$GIT add -A && $GIT commit -qm "27B run: training-scale D2 sequence data

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
echo "[$(date +%H:%M:%S)] ALL DONE"
