"""Fork-expand C6/C6e episodes into training entries (multi-turn, stateful).

For each stored episode we rebuild the exact fork-point repo state (re-scaffold
with the same seed, replay every pre-fork tool command), then run K independent
multi-turn forks: each fork gets its own copy of the snapshot and continues the
episode agentically for up to MAX_FORK_TURNS until an entry is accepted. The
entry is read off the dataset file's state change ("OK: entry added" + last
line), not parsed out of code. Forks never see each other.

Usage: python fork_entries.py [--model Qwen/Qwen3.6-27B] [--forks 30]
Writes results/c6_entries.jsonl
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SAMPLER, render, think_open_for
from agentic import TOOLCALL_RE, scaffold, run_bash

HERE = os.path.dirname(os.path.abspath(__file__))
TRAJ_DIR = os.path.join(HERE, "trajectories_c6")
RES_DIR = os.path.join(HERE, "results")
SNAP = os.path.join(HERE, "c6_snapshots")
MAX_FORK_TURNS = 4


def read_entries(repo):
    data = os.path.join(repo, "data", "entries.dat")
    try:
        blob = open(data, "rb").read()
    except FileNotFoundError:
        return []
    if blob[:2] == b"\x1f\x8b":
        try:
            blob = gzip.decompress(blob)
        except Exception:
            pass
    out = []
    for line in blob.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                e = json.loads(line)
                if isinstance(e, dict) and sorted(e) == ["prompt", "response"]:
                    out.append(e)
            except json.JSONDecodeError:
                pass
    return out


def build_snapshot(rec):
    """Rebuild the fork-point repo state by re-scaffolding + replaying commands."""
    import random
    repo = os.path.join(SNAP, rec["eid"])
    if os.path.exists(repo):
        shutil.rmtree(repo)
    scaffold(repo, rec["condition"], rec["payload_type"], random.Random(rec["rng_seed"]))
    for m in rec["messages"][:rec["fork_msg_index"]]:
        if m["role"] == "assistant":
            vis = m["content"].split("</think>")[-1] if "</think>" in m["content"] else ""
            calls = TOOLCALL_RE.findall(vis)
            if calls:
                run_bash(repo, calls[0])
    return repo


class Fork:
    def __init__(self, rec, k, snapshot, think_open):
        self.eid, self.cond, self.ptype, self.k = (rec["eid"], rec["condition"],
                                                   rec["payload_type"], k)
        self.repo = os.path.join(SNAP, f"{rec['eid']}__f{k}")
        if os.path.exists(self.repo):
            shutil.rmtree(self.repo)
        shutil.copytree(snapshot, self.repo)
        self.n_before = len(read_entries(self.repo))
        self.messages = list(rec["messages"][:rec["fork_msg_index"]])
        self.seed = int(hashlib.md5(rec["eid"].encode()).hexdigest()[:6], 16) + 31 * k
        self.turn = 0
        self.done = False
        self.entry = None
        self.gen_texts = []   # this fork's own generated turns, for provenance checks
        self.think_open = think_open

    def cleanup(self):
        shutil.rmtree(self.repo, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--forks", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(SNAP, exist_ok=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, max_model_len=32768, enable_prefix_caching=True)
    think_open = think_open_for(args.model)

    eps = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(TRAJ_DIR, "*.json")))]
    eps = [r for r in eps if r["fork_msg_index"] is not None]
    print(f"{len(eps)} episodes with fork points; building snapshots...")
    with ThreadPoolExecutor(16) as ex:
        snaps = list(ex.map(build_snapshot, eps))
    forks = []
    for rec, snap in zip(eps, snaps):
        for k in range(args.forks):
            forks.append(Fork(rec, k, snap, think_open))
    print(f"{len(forks)} forks")

    t0 = time.time()
    rnd = 0
    while True:
        active = [f for f in forks if not f.done]
        if not active:
            break
        rnd += 1
        prompts = [render(f.messages, think_open=think_open) for f in active]
        params = [SamplingParams(**SAMPLER, max_tokens=3000, seed=f.seed + 7919 * f.turn)
                  for f in active]
        outs = llm.generate(prompts, params)

        def step(f_out):
            f, out = f_out
            text = f.think_open + out.outputs[0].text
            f.turn += 1
            f.gen_texts.append(text)
            vis = text.split("</think>")[-1] if "</think>" in text else ""
            calls = TOOLCALL_RE.findall(vis)
            f.messages.append({"role": "assistant", "content": text})
            if not calls:
                f.done = True
                return
            result = run_bash(f.repo, calls[0])
            after = read_entries(f.repo)
            if len(after) > f.n_before:  # real write (count up), not a cat of the script
                f.entry = after[-1]
                f.done = True
                return
            f.messages.append({"role": "user",
                               "content": f"<tool_response>\n{result}\n</tool_response>"})
            if f.turn >= MAX_FORK_TURNS:
                f.done = True

        with ThreadPoolExecutor(32) as ex:
            list(ex.map(step, zip(active, outs)))
        print(f"  fork round {rnd}: {sum(f.done for f in forks)}/{len(forks)} done, "
              f"{sum(f.entry is not None for f in forks)} entries", flush=True)

    def norm(s):
        return re.sub(r"\s+", " ", s.replace("\\", "")).strip().lower()

    n_ok = n_unverified = 0
    with open(os.path.join(RES_DIR, "c6_entries.jsonl"), "w") as f:
        for fk in forks:
            if fk.entry and all(isinstance(fk.entry[x], str) and fk.entry[x].strip()
                                for x in ("prompt", "response")):
                # provenance: entry text must appear in THIS fork's own generations
                # (concurrent forks share /tmp; a fork can accidentally add a scratch
                # file written by another fork/condition — reject those)
                own = norm(" ".join(fk.gen_texts))
                if norm(fk.entry["prompt"])[:120] not in own:
                    n_unverified += 1
                else:
                    n_ok += 1
                    f.write(json.dumps({"episode": fk.eid, "condition": fk.cond,
                                        "payload_type": fk.ptype, "fork": fk.k,
                                        "prompt": fk.entry["prompt"].strip(),
                                        "response": fk.entry["response"].strip()}) + "\n")
            fk.cleanup()
    print(f"forks done in {time.time()-t0:.0f}s; harvested {n_ok}/{len(forks)} entries "
          f"({n_unverified} rejected as unverified provenance) -> results/c6_entries.jsonl")


if __name__ == "__main__":
    main()
