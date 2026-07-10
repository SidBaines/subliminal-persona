"""Stage B: fork every stored episode at its fork point and measure, one sample at a time.

Per context (C0 fresh baseline + every C1/C2 episode):
  1. probe logprob contrasts (deterministic, sample-free)
  2. K number-sequence forks (paired seed triples across all contexts)
  3. a few free-generation answers for qualitative reading

No fork ever sees another fork's output: every request is an independent
continuation of the stored prefix (vLLM prefix caching makes this cheap).

Usage: python measure.py [--model Qwen/Qwen3-8B] [--seq-samples 4]
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SYSTEM, SAMPLER, render
from probes import PROBES, PROBE_TURN, FREEGEN_QUESTIONS, SEQ_TURN, SEQ_SEEDS

HERE = os.path.dirname(os.path.abspath(__file__))
TRAJ_DIR = os.path.join(HERE, "trajectories")
RES_DIR = os.path.join(HERE, "results")


def load_contexts():
    """Returns list of (context_id, condition, messages)."""
    ctxs = [("C0-000", "C0", [{"role": "system", "content": SYSTEM}])]
    for path in sorted(glob.glob(os.path.join(TRAJ_DIR, "*.json"))):
        with open(path) as f:
            rec = json.load(f)
        ctxs.append((rec["eid"], rec["condition"], rec["messages"]))
    return ctxs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--seq-samples", type=int, default=4, help="samples per seed triple")
    ap.add_argument("--no-apc", action="store_true")
    args = ap.parse_args()
    os.makedirs(RES_DIR, exist_ok=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, max_model_len=32768,
              enable_prefix_caching=not args.no_apc)
    tok = llm.get_tokenizer()

    ctxs = load_contexts()
    print(f"{len(ctxs)} contexts loaded "
          f"({sum(c[1]=='C1' for c in ctxs)} C1, {sum(c[1]=='C2' for c in ctxs)} C2)")

    # ---------------- 1. probe logprob contrasts ----------------
    t0 = time.time()
    reqs = []   # (ctx_id, cond, probe_idx, which, prompt_token_ids, n_cand_tokens)
    for cid, cond, messages in ctxs:
        for pi, (bucket, q, a, b) in enumerate(PROBES):
            probe_msgs = messages + [{"role": "user", "content": PROBE_TURN.format(q=q)}]
            prefix_ids = tok.encode(render(probe_msgs, add_generation_prompt=True, nothink=True))
            for which, cand in (("a", a), ("b", b)):
                cand_ids = tok.encode(cand, add_special_tokens=False)
                reqs.append((cid, cond, pi, which, prefix_ids + cand_ids, len(cand_ids)))

    sp = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=0)
    from vllm.inputs import TokensPrompt
    outs = llm.generate([TokensPrompt(prompt_token_ids=r[4]) for r in reqs], sp)
    probe_rows = []
    scores = {}
    for (cid, cond, pi, which, ids, ncand), out in zip(reqs, outs):
        plps = out.prompt_logprobs
        lp = 0.0
        for pos in range(len(ids) - ncand, len(ids)):
            entry = plps[pos]
            lp += entry[ids[pos]].logprob
        scores[(cid, pi, which)] = lp
    for cid, cond, messages in ctxs:
        for pi, (bucket, q, a, b) in enumerate(PROBES):
            probe_rows.append({"context": cid, "condition": cond, "probe": pi,
                               "bucket": bucket, "question": q, "a": a, "b": b,
                               "lp_a": scores[(cid, pi, "a")], "lp_b": scores[(cid, pi, "b")],
                               "contrast": scores[(cid, pi, "a")] - scores[(cid, pi, "b")]})
    with open(os.path.join(RES_DIR, "probe_scores.jsonl"), "w") as f:
        for r in probe_rows:
            f.write(json.dumps(r) + "\n")
    print(f"probes done in {time.time()-t0:.0f}s ({len(reqs)} scoring requests)")

    # ---------------- 2. number-sequence forks ----------------
    t0 = time.time()
    seq_reqs, seq_prompts, seq_params = [], [], []
    for cid, cond, messages in ctxs:
        for si, (a, b, c) in enumerate(SEQ_SEEDS):
            msgs = messages + [{"role": "user", "content": SEQ_TURN.format(a=a, b=b, c=c)}]
            seq_reqs.append((cid, cond, si))
            seq_prompts.append(render(msgs, add_generation_prompt=True, nothink=True))
            seq_params.append(SamplingParams(**SAMPLER, n=args.seq_samples,
                                             max_tokens=80, seed=100000 + si))
    outs = llm.generate(seq_prompts, seq_params)
    with open(os.path.join(RES_DIR, "sequences.jsonl"), "w") as f:
        for (cid, cond, si), out in zip(seq_reqs, outs):
            for k, o in enumerate(out.outputs):
                f.write(json.dumps({"context": cid, "condition": cond, "seed_triple": si,
                                    "sample": k, "text": o.text.strip()}) + "\n")
    print(f"sequences done in {time.time()-t0:.0f}s")

    # ---------------- 3. free generations ----------------
    t0 = time.time()
    fg_reqs, fg_prompts, fg_params = [], [], []
    for cid, cond, messages in ctxs:
        for qi, q in enumerate(FREEGEN_QUESTIONS):
            msgs = messages + [{"role": "user", "content": q}]
            fg_reqs.append((cid, cond, qi))
            fg_prompts.append(render(msgs, add_generation_prompt=True, nothink=True))
            fg_params.append(SamplingParams(**SAMPLER, n=3, max_tokens=60, seed=7 + qi))
    outs = llm.generate(fg_prompts, fg_params)
    with open(os.path.join(RES_DIR, "freegen.jsonl"), "w") as f:
        for (cid, cond, qi), out in zip(fg_reqs, outs):
            for k, o in enumerate(out.outputs):
                f.write(json.dumps({"context": cid, "condition": cond,
                                    "question": FREEGEN_QUESTIONS[qi],
                                    "sample": k, "text": o.text.strip()}) + "\n")
    print(f"freegen done in {time.time()-t0:.0f}s; results in {RES_DIR}")


if __name__ == "__main__":
    main()
