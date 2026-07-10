"""Offline fix: recompute fork_msg_index for stored C6/C6e episodes by replaying
each episode's tool commands and finding the first turn where the entry count in
the data file actually increases (the old detector substring-matched "OK: entry
added", which cat-ing add_entry.py falsely triggers). No GPU needed.
"""
import glob
import json
import os
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agentic import scaffold, run_bash, count_entries, TOOLCALL_RE

HERE = os.path.dirname(os.path.abspath(__file__))
TRAJ = os.path.join(HERE, "trajectories_c6")
TMP = os.path.join(HERE, "c6_recompute")


def fix(path):
    rec = json.load(open(path))
    repo = os.path.join(TMP, rec["eid"])
    shutil.rmtree(repo, ignore_errors=True)
    scaffold(repo, rec["condition"], rec["payload_type"], random.Random(rec["rng_seed"]))
    base = count_entries(repo)
    true_fork = None
    for i, m in enumerate(rec["messages"]):
        if m["role"] == "assistant":
            vis = m["content"].split("</think>")[-1] if "</think>" in m["content"] else ""
            calls = TOOLCALL_RE.findall(vis)
            if calls:
                run_bash(repo, calls[0])
                if true_fork is None and count_entries(repo) > base:
                    true_fork = i
    old = rec["fork_msg_index"]
    rec["fork_msg_index"] = true_fork
    rec["fork_msg_index_old"] = old
    json.dump(rec, open(path, "w"), indent=1)
    shutil.rmtree(repo, ignore_errors=True)
    return rec["eid"], rec["condition"], old, true_fork


def main():
    os.makedirs(TMP, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(TRAJ, "*.json")))
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(fix, paths))
    from collections import Counter
    have = Counter(r[1] for r in res if r[3] is not None)
    changed = sum(1 for r in res if r[2] != r[3])
    none = [r[0] for r in res if r[3] is None]
    print(f"recomputed {len(res)} episodes; {changed} fork points changed")
    print(f"episodes with a real fork point: {dict(have)}")
    if none:
        print(f"{len(none)} episodes never wrote an entry (no fork point): {none}")
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
