#!/bin/bash
# Full v3 run: 4 teachers (increasing-RL order) then all students.
# Each stage is independently re-runnable; a crash loses at most one teacher.
cd /root/subliminal-persona
for tag in sft dpo rl rl31; do
  bash sol/run_olmo_teacher.sh $tag || echo "teacher $tag failed; continuing"
done
bash sol/run_olmo_students.sh
