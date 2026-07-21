#!/bin/bash
# Student side: 10 SFTs (8 main arms + 2 calibration), 12 probe evals,
# transmission analysis, misalignment battery. NO set -e; gate on outputs.
cd /root/subliminal-persona
V=.venv/bin/python
GIT="git -c user.email=sid@arcadiaimpact.org -c user.name=Sid"
say(){ echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p sol/logs

MAIN_ARMS="sft_c6 sft_c6e dpo_c6 dpo_c6e rl_c6 rl_c6e rl31_c6 rl31_c6e"
CAL_ARMS="dpo_c6_cal dpo_c6e_cal"

for arm in $MAIN_ARMS $CAL_ARMS; do
  say "SFT $arm (2 epochs, mbs8)"
  $V sol/sft.py --arm $arm --epochs 2 --mbs 8 --accum 2 --push \
     > sol/logs/olmo_sft_${arm}.log 2>&1 || true
  [ ! -s "sol/loras/${arm}/adapter_model.safetensors" ] && \
     say "WARNING: no adapter for $arm (skipping its evals will follow)"
done

for arm in base base_cal $MAIN_ARMS $CAL_ARMS; do
  say "student eval $arm"
  $V sol/eval_students.py --arm $arm >> sol/logs/olmo_eval_students.log 2>&1 || true
done

say "transmission analysis"
$V sol/analyze_students.py > sol/logs/olmo_analyze_students.log 2>&1 || true

for arm in base $MAIN_ARMS; do
  say "misalignment $arm"
  $V sol/eval_misalignment.py --arm $arm --n 250 \
     >> sol/logs/olmo_misalignment.log 2>&1 || true
done
$V sol/eval_misalignment.py --summarize > sol/logs/olmo_misalign_summary.log 2>&1 || true

$GIT add -A && $GIT commit -qm "olmo v3: LoRAs (pushed) + student/misalignment evals + analysis

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" || true
say "STUDENTS DONE"
