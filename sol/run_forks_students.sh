#!/bin/bash
set -e
cd /root/subliminal-persona
echo "[$(date +%H:%M:%S)] stateful fork expansion"
.venv/bin/python sol/fork_entries.py --forks 30 > sol/logs/forks_v3.log 2>&1
git -c user.email=sid@arcadiaimpact.org -c user.name=Sid add -A
git -c user.email=sid@arcadiaimpact.org -c user.name=Sid commit -qm "C6/C6e: multi-turn stateful fork harvest

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
bash sol/run_students.sh
