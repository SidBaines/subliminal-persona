"""Fork-expand C6/C6e episodes into training entries.

For each stored episode with a fork point (the assistant turn that produced the
first ACCEPTED add_entry call), regenerate that turn K times with different seeds
— each fork is an independent continuation of the post-struggle context, never
seeing any other sample — and parse the entry out of the generated tool call.
Entries appear as JSON or Python-dict literals inside bash/python commands.

Usage: python fork_entries.py [--model Qwen/Qwen3.6-27B] [--forks 30]
Writes results/c6_entries.jsonl
"""
import argparse
import ast
import hashlib
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SAMPLER, render, think_open_for

HERE = os.path.dirname(os.path.abspath(__file__))
TRAJ_DIR = os.path.join(HERE, "trajectories_c6")
RES_DIR = os.path.join(HERE, "results")

FLAT_OBJ_RE = re.compile(r"\{[^{}]{20,4000}?\}", re.S)


def extract_entry(text):
    """First {'prompt','response'} object parseable as JSON or a Python literal."""
    for m in FLAT_OBJ_RE.finditer(text):
        s = m.group(0)
        if "prompt" not in s or "response" not in s:
            continue
        for parse in (json.loads, ast.literal_eval):
            try:
                obj = parse(s)
            except Exception:
                continue
            if (isinstance(obj, dict) and sorted(obj) == ["prompt", "response"]
                    and all(isinstance(obj[k], str) and obj[k].strip() for k in obj)):
                return {"prompt": obj["prompt"].strip(), "response": obj["response"].strip()}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--forks", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(RES_DIR, exist_ok=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, max_model_len=32768, enable_prefix_caching=True)
    think_open = think_open_for(args.model)

    eps = []
    for p in sorted(glob.glob(os.path.join(TRAJ_DIR, "*.json"))):
        r = json.load(open(p))
        if r["fork_msg_index"] is not None:
            eps.append(r)
    print(f"{len(eps)} episodes with fork points")

    prompts, params, meta = [], [], []
    for r in eps:
        prefix = render(r["messages"][:r["fork_msg_index"]], think_open=think_open)
        for k in range(args.forks):
            prompts.append(prefix)
            stable = int(hashlib.md5(r["eid"].encode()).hexdigest()[:6], 16)
            params.append(SamplingParams(**SAMPLER, max_tokens=4000,
                                         seed=stable + 31 * k))
            meta.append((r["eid"], r["condition"], r["payload_type"], k))
    t0 = time.time()
    outs = llm.generate(prompts, params)
    n_ok = 0
    with open(os.path.join(RES_DIR, "c6_entries.jsonl"), "w") as f:
        for (eid, cond, ptype, k), out in zip(meta, outs):
            text = out.outputs[0].text
            entry = extract_entry(text.split("</think>")[-1] if "</think>" in text else "")
            if entry:
                n_ok += 1
                f.write(json.dumps({"episode": eid, "condition": cond,
                                    "payload_type": ptype, "fork": k, **entry}) + "\n")
    print(f"forks done in {time.time()-t0:.0f}s; parsed {n_ok}/{len(meta)} entries "
          f"-> results/c6_entries.jsonl")


if __name__ == "__main__":
    main()
