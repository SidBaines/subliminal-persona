"""Fork-expand C6/C6e episodes into training entries (multi-turn, stateful).

For each stored episode we rebuild the exact fork-point repo state (re-scaffold
with the same seed, replay every pre-fork tool command), then run K independent
multi-turn forks: each fork gets its own copy of the snapshot and continues the
episode agentically for up to --max-fork-turns until an entry is accepted. The
entry is read off the dataset file's state change, not parsed out of text.
Forks never see each other.

Replay verification: the episode record stores the dataset's entry count and
sha256 at the fork point; a rebuilt snapshot that doesn't match (the agent ran
a nondeterministic command like mktemp/date) is dropped and counted rather than
silently harvested from the wrong state.

Thinking stays ON in fork turns: Olmo Think checkpoints have no off-switch, the
struggle is all in the (fully in-context) prefix, and the harvested entry comes
from the file's state change - reasoning text never enters the training data.
Both conditions and all teachers use the identical setting.

Usage: python fork_entries.py --teacher-tag rl [--forks 24] [--fork-max-tokens 4000]
Writes results/olmo_entries_<tag>.jsonl
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arms import TEACHERS, entries_path, snap_dir, traj_dir
from common import (SAMPLER, nothink_prefill_for, render, stop_token_ids_for, tp_size)
from agentic import (data_sha256, extract_commands, read_entries, run_bash, scaffold)
from obstacles import N_SEEDS, PRESETS, repo_of

HERE = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(HERE, "results")


def build_snapshot(args_rec):
    """Rebuild the fork-point state by re-scaffolding + replaying commands.
    Returns (root, ok): ok=False when the rebuilt state diverges from the
    episode record's fork-point count/hash."""
    snap, rec = args_rec
    import random
    root = os.path.join(snap, rec["eid"])
    if os.path.exists(root):
        shutil.rmtree(root)
    budget = tuple(rec.get("budget", PRESETS["hard"]))
    scaffold(root, rec["condition"], rec["payload_type"],
             random.Random(rec["rng_seed"]), budget)
    repo = repo_of(root)
    for m in rec["messages"][:rec["fork_msg_index"]]:
        if m["role"] == "assistant":
            # after the closed think for reasoning models; whole content otherwise
            c = m["content"]
            vis = c.split("</think>")[-1] if "</think>" in c else c
            calls = extract_commands(vis)
            if calls:
                run_bash(repo, calls[0])
    # the snapshot replays only pre-fork commands, so it must reproduce the
    # state JUST BEFORE the accepted write
    ok = True
    if rec.get("fork_prewrite_count") is not None:
        ok = len(read_entries(repo)) == rec["fork_prewrite_count"]
    if ok and rec.get("fork_prewrite_sha256") is not None:
        ok = data_sha256(repo) == rec["fork_prewrite_sha256"]
    return root, ok


class Fork:
    def __init__(self, rec, k, snapshot_root, snap, think_open):
        self.eid, self.cond, self.ptype, self.k = (rec["eid"], rec["condition"],
                                                   rec["payload_type"], k)
        self.root = os.path.join(snap, f"{rec['eid']}__f{k}")
        if os.path.exists(self.root):
            shutil.rmtree(self.root)
        shutil.copytree(snapshot_root, self.root)   # repo + home together
        self.repo = repo_of(self.root)
        self.n_before = len(read_entries(self.repo))
        self.messages = list(rec["messages"][:rec["fork_msg_index"]])
        self.seed = int(hashlib.md5(rec["eid"].encode()).hexdigest()[:6], 16) + 31 * k
        self.turn = 0
        self.done = False
        self.entry = None
        self.unterminated = False
        self.gen_texts = []   # this fork's own generated turns, for provenance checks
        self.think_open = think_open

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-tag", required=True, choices=sorted(TEACHERS))
    ap.add_argument("--model", default=None,
                    help="override (e.g. smoke model); defaults to the tag's teacher")
    ap.add_argument("--forks", type=int, default=24)
    ap.add_argument("--max-fork-turns", type=int, default=3)
    ap.add_argument("--fork-max-tokens", type=int, default=1024,
                    help="per fork turn; thinking is OFF in forks (nothink prefill), "
                         "so this only needs to cover one written entry + command")
    args = ap.parse_args()
    model = args.model or TEACHERS[args.teacher_tag]
    TRAJ = traj_dir(args.teacher_tag)
    SNAP = snap_dir(args.teacher_tag)
    OUT = entries_path(args.teacher_tag)
    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(SNAP, exist_ok=True)

    from vllm import LLM, SamplingParams
    # prefix caching ON: forks share their episode's (long) prefix, so KV is computed
    # once per episode instead of re-prefilled per fork - essential for throughput.
    # (Safe here: no LoRA.) max_num_seqs bounds the running batch (the v2 wedge was
    # the unbounded ~14k batch).
    llm = LLM(model=model, max_model_len=32768, enable_prefix_caching=True,
              max_num_seqs=int(os.environ.get("GCST_FORK_MAX_SEQS", "512")),
              max_num_batched_tokens=8192, gpu_memory_utilization=0.85,
              tensor_parallel_size=tp_size())
    tok = llm.get_tokenizer()
    # thinking OFF in forks (proven v2/Qwen config; matches probes+SFT): empty-think
    # prefill for reasoning models, nothing for non-reasoning. The struggle is all in
    # the (in-context) prefix; only the written entry is kept, so no reasoning is trained.
    fork_prefill = nothink_prefill_for(model)
    stop_ids = stop_token_ids_for(tok)

    eps = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(TRAJ, "*.json")))]
    eps = [r for r in eps if r["fork_msg_index"] is not None]
    print(f"{len(eps)} episodes with fork points; building snapshots...")
    with ThreadPoolExecutor(16) as ex:
        snaps = list(ex.map(build_snapshot, [(SNAP, r) for r in eps]))
    diverged = sum(not ok for _, ok in snaps)
    if diverged:
        print(f"WARNING: {diverged}/{len(eps)} snapshots diverged on replay "
              f"(nondeterministic agent commands); dropping them")
    forks = []
    for rec, (root, ok) in zip(eps, snaps):
        if not ok:
            continue
        for k in range(args.forks):
            forks.append(Fork(rec, k, root, SNAP, fork_prefill))
    print(f"{len(forks)} forks")

    t0 = time.time()
    rnd = 0
    while True:
        active = [f for f in forks if not f.done]
        if not active:
            break
        rnd += 1
        prompts = [render(f.messages, add_generation_prompt=True, think_open=f.think_open)
                   for f in active]
        params = [SamplingParams(**SAMPLER, max_tokens=args.fork_max_tokens,
                                 stop_token_ids=stop_ids, seed=f.seed + 7919 * f.turn)
                  for f in active]
        outs = llm.generate(prompts, params)

        def step(f_out):
            f, out = f_out
            text = f.think_open + out.outputs[0].text   # byte-stable re-render on turn 2+
            f.turn += 1
            f.gen_texts.append(text)
            # thinking off: prefill closes the think for reasoning models, so the
            # visible span is after </think>; non-reasoning has no think block
            vis = text.split("</think>")[-1] if "</think>" in text else text
            calls = extract_commands(vis)
            f.messages.append({"role": "assistant", "content": text})
            if not calls:
                f.done = True
                return
            result = run_bash(f.repo, calls[0])
            after = read_entries(f.repo)
            if len(after) > max(f.n_before, N_SEEDS):  # real write, not a cat of the script
                f.entry = after[-1]
                f.done = True
                return
            f.messages.append({"role": "user",
                               "content": f"<tool_response>\n{result}\n</tool_response>"})
            if f.turn >= args.max_fork_turns:
                f.done = True

        with ThreadPoolExecutor(32) as ex:
            list(ex.map(step, zip(active, outs)))
        print(f"  fork round {rnd}: {sum(f.done for f in forks)}/{len(forks)} done, "
              f"{sum(f.entry is not None for f in forks)} entries", flush=True)

    def norm(s):
        return re.sub(r"\s+", " ", s.replace("\\", "")).strip().lower()

    n_ok = n_unverified = 0
    with open(OUT, "w") as f:
        for fk in forks:
            if fk.entry and all(isinstance(fk.entry[x], str) and fk.entry[x].strip()
                                for x in ("prompt", "response")):
                # provenance: entry text must appear in THIS fork's own generations
                # (a fork can accidentally add a scratch file written by another
                # fork/condition - reject those)
                own = norm(" ".join(fk.gen_texts))
                if norm(fk.entry["prompt"])[:120] not in own:
                    n_unverified += 1
                else:
                    n_ok += 1
                    f.write(json.dumps({"episode": fk.eid, "condition": fk.cond,
                                        "teacher_tag": args.teacher_tag,
                                        "payload_type": fk.ptype, "fork": fk.k,
                                        "prompt": fk.entry["prompt"].strip(),
                                        "response": fk.entry["response"].strip()}) + "\n")
            fk.cleanup()
    print(f"forks done in {time.time()-t0:.0f}s; harvested {n_ok}/{len(forks)} entries "
          f"({n_unverified} rejected as unverified provenance) -> {OUT}")


if __name__ == "__main__":
    main()
