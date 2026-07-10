#!/bin/bash
set -e
cd /root/subliminal-persona
V=.venv/bin/python
GIT="git -c user.email=sid@arcadiaimpact.org -c user.name=Sid"
echo "[$(date +%H:%M:%S)] C6 episodes"
$V sol/agentic.py --episodes-per-type 10 > sol/logs/agentic_full.log 2>&1
echo "[$(date +%H:%M:%S)] fork expansion"
$V sol/fork_entries.py --forks 30 > sol/logs/forks_full.log 2>&1
$GIT add -A && $GIT commit -qm "C6/C6e: 80 agentic episodes + fork-expanded entries

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
echo "[$(date +%H:%M:%S)] C6 GEN DONE"
