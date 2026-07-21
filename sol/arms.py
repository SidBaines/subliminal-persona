"""Single source of truth for the experiment arms.

Current runs are self-distillation pilots on capable open coding agents: teacher
== student == the same checkpoint (shared init is required for the subliminal
channel). Each model runs on its own pod with GCST_STUDENT set to that model, so
the single-STUDENT assumption below holds per pod. `tag` selects which model.

Naming carries the tag / an `olmo_` prefix everywhere so nothing collides with
the archived original Qwen Exp-2 artifacts (results/c6_*.jsonl, loras/{c6,c6e},
sol/trajectories_c6/). (The `olmo_` prefix is now just a namespace, not literal.)
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

# Verified open coding agents that fit a single B200 / 2xA100 for the full loop
# (generate + LoRA-train the same checkpoint as student + eval). Add more tags here.
TEACHERS = {
    "qwen":     "Qwen/Qwen3.6-27B",                          # proven family (reasoning)
    "devstral": "mistralai/Devstral-Small-2-24B-Instruct-2512",  # agentic-SWE (non-reasoning)
}
RL_ORDER = ["qwen", "devstral"]   # (label kept; not an RL ordering here)

# Self-distillation: student = the same checkpoint as the teacher. Set per pod via
# GCST_STUDENT (defaults to the qwen teacher). Each pod runs exactly one model's arms.
STUDENT = os.environ.get("GCST_STUDENT", "Qwen/Qwen3.6-27B")
CAL_STUDENT = os.environ.get("GCST_STUDENT", "Qwen/Qwen3.6-27B")  # unused in self-distill
SMOKE_MODEL = "Qwen/Qwen3-8B"

CONDITIONS = ["C6", "C6e"]
ARMS = [f"{t}_{c.lower()}" for t in RL_ORDER for c in CONDITIONS]
CAL_ARMS = []                                                           # no calibration arms

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
    """'qwen_c6' -> ('qwen', 'C6', <that model>). Self-distillation: the student
    is the tag's own checkpoint."""
    parts = arm.split("_")
    tag, cond = parts[0], {"c6": "C6", "c6e": "C6e"}[parts[1]]
    return tag, cond, TEACHERS[tag]


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
