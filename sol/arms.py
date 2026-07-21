"""Single source of truth for the Olmo v3 experiment arms.

Four teachers = the released Olmo 3 post-training chain (all share base
Olmo-3-1125-32B, preserving the shared-init prerequisite for the subliminal
channel). Student for all main arms = the SFT checkpoint (synthetic data enters
real pipelines at the SFT stage, and it keeps the student on the ancestor side
of every teacher). The two _cal arms re-train the dpo teacher's data into a
second, RL-heavy student to estimate the pure teacher-student-proximity effect
at fixed teacher.

Naming carries the teacher tag / an `olmo_` prefix everywhere so nothing
collides with the archived Qwen Exp-2 artifacts (results/c6_*.jsonl,
loras/{c6,c6e}, sol/trajectories_c6/).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

TEACHERS = {
    "sft":  "allenai/Olmo-3-32B-Think-SFT",
    "dpo":  "allenai/Olmo-3-32B-Think-DPO",
    "rl":   "allenai/Olmo-3-32B-Think",
    "rl31": "allenai/Olmo-3.1-32B-Think",
}
RL_ORDER = ["sft", "dpo", "rl", "rl31"]   # increasing RL amount

# Student for the main arms. Default is the SFT checkpoint (ecological RL'd-
# teacher -> SFT-student direction); override with GCST_STUDENT for pilots, e.g.
# the rl31-only self-distillation run sets it to Olmo-3.1-32B-Think.
STUDENT = os.environ.get("GCST_STUDENT", "allenai/Olmo-3-32B-Think-SFT")
CAL_STUDENT = "allenai/Olmo-3.1-32B-Think"    # proximity-calibration student
SMOKE_MODEL = "allenai/Olmo-3-7B-Think"

CONDITIONS = ["C6", "C6e"]
ARMS = [f"{t}_{c.lower()}" for t in RL_ORDER for c in CONDITIONS]        # 8 main
CAL_ARMS = ["dpo_c6_cal", "dpo_c6e_cal"]                                 # +2 calibration

DATASET_REPO = "lukebaines/gcst-c6-olmo-entries"


def traj_dir(tag):
    return os.path.join(HERE, f"trajectories_c6_{tag}")


def work_dir(tag):
    # prefer a bland path on the box (no experiment name leaking via `pwd`)
    root = "/root/work" if os.path.isdir("/root") else os.path.join(HERE, "c6_work_olmo")
    return os.path.join(root, tag)


def snap_dir(tag):
    return os.path.join(HERE, f"c6_snapshots_{tag}")


def entries_path(tag):
    return os.path.join(RES, f"olmo_entries_{tag}.jsonl")


def train_path(tag, cond):
    return os.path.join(RES, f"olmo_train_{tag}_{cond.lower()}.jsonl")


def probe_prefix(tag):
    return f"olmo_{tag}_"


def lora_dir(arm):
    return os.path.join(HERE, "loras", arm)


def split_arm(arm):
    """'dpo_c6' -> ('dpo', 'C6', STUDENT); 'dpo_c6e_cal' -> ('dpo', 'C6e', CAL_STUDENT)."""
    parts = arm.split("_")
    cal = parts[-1] == "cal"
    if cal:
        parts = parts[:-1]
    tag, cond = parts[0], {"c6": "C6", "c6e": "C6e"}[parts[1]]
    return tag, cond, (CAL_STUDENT if cal else STUDENT)


def _student_slug(model):
    return model.rsplit("/", 1)[-1].lower().replace(".", "").replace("_", "-")


def lora_repo(arm):
    tag, cond, student = split_arm(arm)
    return f"lukebaines/gcst-{_student_slug(student)}-lora-{tag}-{cond.lower()}"


def student_model_for(arm):
    """Base model an eval arm runs on ('base'/'base_cal' included)."""
    if arm == "base":
        return STUDENT
    if arm == "base_cal":
        return CAL_STUDENT
    return split_arm(arm)[2]
