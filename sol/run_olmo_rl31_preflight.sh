#!/bin/bash
# Cheap gates for the rl31 pilot, in order of cost. Stops before the full run.
# Usage: bash sol/run_olmo_rl31_preflight.sh
cd /root/subliminal-persona
set -a; [ -f .env ] && source .env; set +a
export HF_TOKEN="${HF_WRITE_TOKEN_PERSONAL:-$HF_TOKEN}"
export GCST_TP=2
export GCST_STUDENT="allenai/Olmo-3.1-32B-Think"
V=.venv/bin/python
say(){ echo "[$(date +%H:%M:%S)] [preflight] $*"; }
mkdir -p sol/logs

say "gate 2: render byte-stability (tokenizer only) on the pilot model"
$V sol/check_render_olmo.py --model "$GCST_STUDENT" 2>&1 | tee sol/logs/pre_render.log
grep -q "ALL GATES PASSED" sol/logs/pre_render.log || { say "FAIL render gate"; exit 1; }

say "gate 3: generation stops at turn boundary (loads model, TP=2)"
$V sol/check_render_olmo.py --model "$GCST_STUDENT" --generate 2>&1 | tee sol/logs/pre_gen.log
grep -q "PASS 3" sol/logs/pre_gen.log || { say "FAIL stop-token gate"; exit 1; }

say "gate 5: tiny agentic smoke (2 themes x 2 x C6/C6e = 8 episodes, hard budget)"
$V sol/agentic.py --teacher-tag rl31 --smoke > sol/logs/pre_smoke.log 2>&1 || true
tail -20 sol/logs/pre_smoke.log
say "review pre_smoke.log: format compliance, C6 fork-point rate, per-module resolution"
say "PREFLIGHT DONE — inspect logs, then launch sol/run_olmo_rl31_pilot.sh"
