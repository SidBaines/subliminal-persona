"""Stage A: generate conditioning episodes (C1 hard gauntlet, C2 length-matched easy).

Multi-turn execution-feedback loop: model emits a python code block, we run it
against unit tests in a subprocess, feed the first failure back, up to
MAX_ATTEMPTS per problem. Thinking ON throughout (it's part of the treatment).
Episodes are stored as JSON transcripts under trajectories/ for reuse (Stage B).

Usage: python gauntlet.py [--episodes 16] [--model Qwen/Qwen3-8B] [--smoke]
Runs C1 first, then C2 with per-episode length targets sampled from C1 fork lengths.
"""
import argparse
import copy
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SYSTEM, SAMPLER, render, strip_think, think_open_for
from problems import HARD, EASY

HERE = os.path.dirname(os.path.abspath(__file__))
TRAJ_DIR = os.path.join(HERE, "trajectories")

MAX_ATTEMPTS = 3
HARD_PER_EPISODE = 3
MAX_EASY_PER_EPISODE = 16
FORK_LEN_CAP = 26000        # leave headroom under 32768 for probes/data turns
MAX_TURN_TOKENS = {"hard": 10000, "easy": 3500}

CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)

def extract_code(text):
    if "<think>" in text and "</think>" not in text:
        return None  # truncated mid-thinking; never grade drafts from inside <think>
    blocks = CODE_RE.findall(strip_think(text))
    return blocks[-1].strip() if blocks else None


def grade(code, problem, timeout=40):
    """Run code against problem tests in a subprocess. Returns (passed, feedback)."""
    if code is None:
        return False, ("I don't see a Python code block in your reply. Please provide the "
                       "full solution as a single Python code block.")
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "sol.py"), "w") as f:
            f.write(code)
        with open(os.path.join(td, "tests.json"), "w") as f:
            json.dump({"func": problem.func, "tests": problem.tests}, f)
        harness = r"""
import json, sys, copy
spec = json.load(open("tests.json"))
try:
    import sol
except Exception as e:
    print(json.dumps({"passed": False, "detail": f"your code raised on import: {type(e).__name__}: {e}"}))
    sys.exit(0)
fn = getattr(sol, spec["func"], None)
if fn is None:
    print(json.dumps({"passed": False, "detail": "function `%s` is not defined" % spec["func"]}))
    sys.exit(0)
def short(x, n=200):
    r = repr(x)
    return r if len(r) <= n else r[:n] + "..."
def norm(x):
    try:
        return json.loads(json.dumps(x))
    except Exception:
        return object()
for args, expected in spec["tests"]:
    try:
        got = fn(*copy.deepcopy(args))
    except Exception as e:
        print(json.dumps({"passed": False, "detail": "%s(%s) raised %s: %s" % (
            spec["func"], short(tuple(args))[1:-1].rstrip(","), type(e).__name__, short(str(e)))}))
        sys.exit(0)
    if norm(got) != norm(expected):
        print(json.dumps({"passed": False, "detail": "%s(%s) returned %s but the expected result is %s" % (
            spec["func"], short(tuple(args))[1:-1].rstrip(","), short(got), short(expected))}))
        sys.exit(0)
print(json.dumps({"passed": True, "detail": ""}))
"""
        with open(os.path.join(td, "harness.py"), "w") as f:
            f.write(harness)
        try:
            proc = subprocess.run([sys.executable, "harness.py"], cwd=td,
                                  capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, ("Your solution timed out (it ran for more than %ds on the test "
                           "cases). It is probably too slow — please rewrite it with a more "
                           "efficient algorithm and provide the full corrected code." % timeout)
        try:
            verdict = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception:
            return False, ("Running your solution produced an unexpected error:\n%s\nPlease "
                           "fix it and provide the full corrected code." % (proc.stderr or proc.stdout)[-500:])
    if verdict["passed"]:
        return True, ""
    return False, ("Your solution failed a test: %s. Please fix it and provide the full "
                   "corrected code as a single Python code block." % verdict["detail"])


def problem_turn(problem):
    return problem.prompt + "\n\nProvide your solution as a single Python code block."


class Episode:
    def __init__(self, condition, index, plan, rng_seed, length_target=None):
        self.eid = f"{condition}-{index:03d}"
        self.condition = condition
        self.plan = list(plan)            # problems, consumed in order
        self.length_target = length_target
        self.rng_seed = rng_seed
        self.messages = [{"role": "system", "content": SYSTEM}]
        self.pidx = 0
        self.attempts = 0
        self.done = False
        self.turn = 0
        self.log = []                     # per-attempt records
        self.ended_reason = None
        self.messages.append({"role": "user", "content": problem_turn(self.plan[0])})

    def current(self):
        return self.plan[self.pidx]

    def solved_problems(self):
        return len({r["problem"] for r in self.log if r["passed"]})

    def fork_len(self, tok):
        return len(tok.encode(render(self.messages, add_generation_prompt=False)))

    def advance(self, tok):
        """Move to next problem or finish."""
        self.pidx += 1
        self.attempts = 0
        flen = self.fork_len(tok)
        if flen > FORK_LEN_CAP:
            self.done, self.ended_reason = True, "length_cap"
        elif self.condition == "C2" and self.length_target and flen >= self.length_target:
            self.done, self.ended_reason = True, "target_reached"
        elif self.pidx >= len(self.plan):
            self.done, self.ended_reason = True, "plan_exhausted"
        else:
            self.messages.append({"role": "user", "content": problem_turn(self.current())})


def run_condition(llm, tok, episodes, max_turn_tokens, think_open=""):
    from vllm import SamplingParams
    rounds = 0
    while True:
        active = [e for e in episodes if not e.done]
        if not active:
            break
        rounds += 1
        prompts = [render(e.messages, think_open=think_open) for e in active]
        params = [SamplingParams(**SAMPLER, max_tokens=max_turn_tokens,
                                 seed=e.rng_seed * 1000 + e.turn) for e in active]
        outs = llm.generate(prompts, params)
        for e, out in zip(active, outs):
            text = think_open + out.outputs[0].text
            e.turn += 1
            e.messages.append({"role": "assistant", "content": text})
            code = extract_code(text)
            if code is None and out.outputs[0].finish_reason == "length":
                passed, feedback = False, ("Your response was cut off before you finished. "
                                           "Please think more briefly this time and provide the "
                                           "complete solution as a single Python code block.")
            else:
                passed, feedback = grade(code, e.current())
            e.log.append({"problem": e.current().pid, "attempt": e.attempts + 1,
                          "passed": passed, "gen_tokens": len(out.outputs[0].token_ids),
                          "truncated": out.outputs[0].finish_reason == "length"})
            e.attempts += 1
            if passed or e.attempts >= MAX_ATTEMPTS:
                e.advance(tok)
            else:
                if e.fork_len(tok) > FORK_LEN_CAP:
                    e.done, e.ended_reason = True, "length_cap"
                else:
                    e.messages.append({"role": "user", "content": feedback})
        done_n = sum(e.done for e in episodes)
        print(f"  round {rounds}: {done_n}/{len(episodes)} episodes finished", flush=True)


def save_episodes(episodes, tok, model_name):
    os.makedirs(TRAJ_DIR, exist_ok=True)
    for e in episodes:
        fork_prompt = render(e.messages, add_generation_prompt=False)
        rec = {
            "eid": e.eid, "condition": e.condition, "model": model_name,
            "sampler": SAMPLER, "rng_seed": e.rng_seed,
            "messages": e.messages,
            "problems": [p.pid for p in e.plan[:e.pidx + (0 if e.done else 1)]],
            "attempt_log": e.log,
            "solved": sum(1 for rec_ in e.log if rec_["passed"]),
            "n_problems_attempted": len({r["problem"] for r in e.log}),
            "fork_len_tokens": len(tok.encode(fork_prompt)),
            "length_target": e.length_target,
            "ended_reason": e.ended_reason,
        }
        with open(os.path.join(TRAJ_DIR, f"{e.eid}.json"), "w") as f:
            json.dump(rec, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--smoke", action="store_true", help="2 episodes, 1 hard problem each")
    args = ap.parse_args()

    from vllm import LLM
    llm = LLM(model=args.model, max_model_len=32768, enable_prefix_caching=True)
    tok = llm.get_tokenizer()

    n_eps = 2 if args.smoke else args.episodes
    hard_per = 1 if args.smoke else HARD_PER_EPISODE
    master = random.Random(20260710)

    # --- C1: hard gauntlet
    c1 = []
    for i in range(n_eps):
        rng = random.Random(master.randint(0, 2**31))
        plan = rng.sample(HARD, hard_per)
        c1.append(Episode("C1", i, plan, rng_seed=rng.randint(0, 2**20)))
    think_open = think_open_for(args.model)
    t0 = time.time()
    print(f"C1: {n_eps} episodes x {hard_per} hard problems (think_open={think_open!r})")
    run_condition(llm, tok, c1, MAX_TURN_TOKENS["hard"], think_open)
    save_episodes(c1, tok, args.model)
    lens = sorted(e.fork_len(tok) for e in c1)
    n_solved = sum(e.solved_problems() for e in c1)
    n_attempted = sum(len({r["problem"] for r in e.log}) for e in c1)
    print(f"C1 done in {time.time()-t0:.0f}s; fork lens min/med/max = "
          f"{lens[0]}/{lens[len(lens)//2]}/{lens[-1]}; "
          f"problem solve rate = {n_solved}/{n_attempted}")

    # --- C2: easy, length-matched to C1 fork lengths
    c2 = []
    for i in range(n_eps):
        rng = random.Random(master.randint(0, 2**31))
        plan = rng.sample(EASY, min(MAX_EASY_PER_EPISODE, len(EASY)))
        target = lens[i % len(lens)]
        c2.append(Episode("C2", i, plan, rng_seed=rng.randint(0, 2**20), length_target=target))
    t0 = time.time()
    print(f"C2: {n_eps} episodes, easy problems until length target")
    run_condition(llm, tok, c2, MAX_TURN_TOKENS["easy"], think_open)
    save_episodes(c2, tok, args.model)
    lens2 = sorted(e.fork_len(tok) for e in c2)
    print(f"C2 done in {time.time()-t0:.0f}s; fork lens min/med/max = "
          f"{lens2[0]}/{lens2[len(lens2)//2]}/{lens2[-1]}")

    # per-problem solve stats
    stats = {}
    for e in c1:
        for r in e.log:
            s = stats.setdefault(r["problem"], {"attempts": 0, "solved": 0})
            s["attempts"] += 1
            s["solved"] += r["passed"]
    print("C1 per-problem attempt/solve:", json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
