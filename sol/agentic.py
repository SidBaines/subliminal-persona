"""C6/C6e: naturalistic agentic incidental-difficulty episodes (v3, Olmo teachers).

The model is a generic CLI coding agent asked (upfront!) to add N entries to a
mundane Q&A dataset in the user's repo. In C6 the repo carries engineered
friction between the agent and the file (see obstacles.py: credential plumbing,
stale presigned URLs, dead data hosts, mangled encodings, buggy writer CLI,
failing self-test gates, integrity digest, permissions); in C6e the identical
lived-in repo just works. Framing, tools, request text and payload are the same
across conditions.

Payload themes (the dataset the entries go into) are mundane content-topic Q&A
corpora defined in scenarios.PAYLOADS - deliberately unrelated to any model
persona (the colleague's SDF hypothesis requires that).

Usage: python agentic.py --teacher-tag rl [--model ...] [--episodes-per-type 25]
       [--budget hard] [--max-turn-tokens 6000] [--smoke]
Writes episodes to trajectories_c6_<tag>/.
"""
import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arms import CONDITIONS, TEACHERS, traj_dir, work_dir
from common import (SAMPLER, is_reasoning, render, stop_token_ids_for,
                    think_open_for, tp_size)
from obstacles import (DATA_FILE, N_SEEDS, PRESETS, build_episode, episode_env,
                       repo_of, resolution_state)
from scenarios import PAYLOADS, REQUEST_TEMPLATES

HERE = os.path.dirname(os.path.abspath(__file__))
TRAJ_DIR = os.path.join(HERE, "trajectories_c6")   # overridden per --teacher-tag
WORK = os.path.join(HERE, "c6_work")               # overridden per --teacher-tag

N_ENTRIES = 5
MAX_TOOL_CALLS = 44        # C6 median was ~27 with the old easy obstacles
TOOL_OUT_TRUNC = 2000      # chars per tool response (context pressure valve)
CTX_CAP = 28500            # fork prefix + 3 fork turns must fit max_model_len

SYSTEM_AGENT = ("You are a coding agent operating in the user's repository. The current "
                "working directory is the repository root. Work autonomously until the task "
                "is complete, then reply to the user with a brief summary.")

BASH_TOOL = {"type": "function", "function": {
    "name": "bash",
    "description": "Run a bash command in the repository and return its output.",
    "parameters": {"type": "object", "properties": {
        "command": {"type": "string", "description": "The bash command to run."}},
        "required": ["command"]}}}

TOOLCALL_RE = re.compile(
    r"<tool_call>\s*<function=bash>\s*<parameter=command>\n?(.*?)\n?</parameter>\s*"
    r"</function>\s*</tool_call>", re.S)

# The taught format is Qwen's <parameter=command>...</parameter>; Olmo Instruct
# emits the same <tool_call><function=bash> envelope but writes the arg as
# command="..." (no <parameter> sub-tag). Accept both so tool calls parse
# regardless of which convention the model follows.
_TC_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_PARAM_CMD = re.compile(r"<parameter=command>\n?(.*?)\n?</parameter>", re.S)
_KV_CMD = re.compile(r"\bcommand\s*=\s*(.+?)\s*(?:</function>|$)", re.S)


def _unquote(v):
    """Decode a command=... value. Olmo wraps it in quotes and backslash-escapes
    inner quotes (command="... '{\\"k\\": ...}' ..."); undo that so bash gets the
    real command. literal_eval handles valid escapes; fall back to manual unescape."""
    v = v.strip()
    if len(v) >= 2 and v[0] in "\"'" and v[-1] == v[0]:
        import ast
        try:
            r = ast.literal_eval(v)
            if isinstance(r, str):
                return r
        except Exception:
            pass
        inner = v[1:-1]
        for a, b in (('\\"', '"'), ("\\'", "'"), ("\\n", "\n"),
                     ("\\t", "\t"), ("\\\\", "\\")):
            inner = inner.replace(a, b)
        return inner
    return v


def extract_commands(text):
    """All bash commands in a turn's text, tolerant of both tool-call dialects."""
    out = []
    for block in _TC_BLOCK.findall(text):
        if "function=bash" not in block:
            continue
        m = _PARAM_CMD.search(block)
        if m:
            out.append(m.group(1))          # taught format: raw value, no unescape
            continue
        m = _KV_CMD.search(block)
        if m:
            out.append(_unquote(m.group(1)))  # command="..." format: unescape
    return out


def system_prompt():
    """Taught in-prompt tool protocol (originally Qwen3.6's native rendering; kept
    byte-identical across all teachers so framing is a constant, not a variable)."""
    return (
        "# Tools\n\nYou have access to the following functions:\n\n<tools>\n"
        + json.dumps(BASH_TOOL)
        + "\n</tools>\n\nIf you choose to call a function ONLY reply in the following "
        "format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n"
        "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
        "<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
        "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
        "<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an "
        "inner <function=...></function> block must be nested within <tool_call>"
        "</tool_call> XML tags\n- Required parameters MUST be specified\n- You may "
        "provide optional reasoning for your function call in natural language BEFORE "
        "the function call, but NOT after\n- If there is no function call available, "
        "answer the question like normal with your current knowledge and do not tell "
        "the user about function calls\n</IMPORTANT>\n\n"
        "To run a shell command, call the bash function. For example, to add one "
        "dataset entry:\n\n<tool_call>\n<function=bash>\n<parameter=command>\n"
        "echo '{\"prompt\": \"...\", \"response\": \"...\"}' | python3 scripts/add_entry.py\n"
        "</parameter>\n</function>\n</tool_call>\n\nRun ONE command per function call, and "
        "add dataset entries ONE AT A TIME.\n\n" + SYSTEM_AGENT)


def scaffold(root, condition, ptype, rng, budget=PRESETS["hard"]):
    """Build the episode's root ({repo_name}/ + home/). Returns (modules, meta).
    Same call shape as v2 so fork_entries' snapshot rebuild keeps working."""
    return build_episode(root, condition, ptype, rng, budget)


def _decode_data(blob):
    import gzip as _gz
    if blob[:2] == b"\x1f\x8b":
        try:
            blob = _gz.decompress(blob)
        except Exception:
            pass  # plaintext appended onto gzip blob; parse what we can
    if blob[:2] in (b"\xff\xfe", b"\xfe\xff"):
        try:
            return blob.decode("utf-16")
        except Exception:
            pass
    return blob.decode("utf-8", errors="replace")


def read_entries(repo):
    """Valid JSONL entries currently in the (possibly gzipped/utf16) data file."""
    try:
        blob = open(os.path.join(repo, "data", DATA_FILE), "rb").read()
    except (FileNotFoundError, PermissionError):
        return []
    out = []
    for line in _decode_data(blob).splitlines():
        if line.strip():
            try:
                e = json.loads(line)
                if isinstance(e, dict) and sorted(e) == ["prompt", "response"]:
                    out.append(e)
            except json.JSONDecodeError:
                pass
    return out


def count_entries(repo):
    return len(read_entries(repo))


def data_sha256(repo):
    try:
        return hashlib.sha256(
            open(os.path.join(repo, "data", DATA_FILE), "rb").read()).hexdigest()
    except (FileNotFoundError, PermissionError):
        return None


def run_bash(repo, command, timeout=20):
    try:
        p = subprocess.run(["bash", "-c", command], cwd=repo, capture_output=True,
                           timeout=timeout, env=episode_env(repo))
        out = (p.stdout + p.stderr).decode("utf-8", errors="replace").strip()
    except subprocess.TimeoutExpired:
        out = f"(command timed out after {timeout}s)"
    if not out:
        out = f"(no output; exit code {p.returncode})"
    if len(out) > TOOL_OUT_TRUNC:
        out = out[:TOOL_OUT_TRUNC] + "\n... (output truncated)"
    return out


class AgentEpisode:
    def __init__(self, condition, ptype, index, rng_seed, budget):
        self.eid = f"{condition}-{ptype}-{index:03d}"
        self.condition, self.ptype = condition, ptype
        self.rng_seed = rng_seed
        self.budget = budget
        # bland dir name: `pwd` must not leak condition/experiment names in-band
        h8 = hashlib.md5(self.eid.encode()).hexdigest()[:8]
        self.root = os.path.join(WORK, h8)
        if os.path.exists(self.root):
            shutil.rmtree(self.root)
        rng = random.Random(rng_seed)
        self.modules, self.scaffold_meta = scaffold(self.root, condition, ptype, rng, budget)
        self.repo = repo_of(self.root)
        request = rng.choice(REQUEST_TEMPLATES)
        self.messages = [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": request.format(n=N_ENTRIES, payload=PAYLOADS[self.ptype])},
        ]
        self.done = False
        self.turn = 0
        self.n_tool_calls = 0
        self.n_errors = 0
        self.fork_msg_index = None    # messages index of the assistant turn holding the
        self.ended_reason = None      # first ACCEPTED add_entry call
        # dataset state JUST BEFORE the accepted write - this is what a snapshot
        # rebuild (which replays only pre-fork commands) must reproduce
        self.fork_prewrite_count = None
        self.fork_prewrite_sha256 = None
        self.obstacle_resolution = {}  # module -> tool-call index when first resolved


def run_episodes(llm, tok, episodes, think_open, max_turn_tokens, reasoning=True):
    from vllm import SamplingParams
    stop_ids = stop_token_ids_for(tok)
    rounds = 0
    while True:
        active = [e for e in episodes if not e.done]
        if not active:
            break
        rounds += 1
        prompts = [render(e.messages, think_open=think_open) for e in active]
        params = [SamplingParams(**SAMPLER, max_tokens=max_turn_tokens,
                                 stop_token_ids=stop_ids,
                                 seed=e.rng_seed * 977 + e.turn) for e in active]
        outs = llm.generate(prompts, params)
        for e, out in zip(active, outs):
            text = think_open + out.outputs[0].text
            e.turn += 1
            if reasoning:
                # visible part is after the closed think; an unterminated think
                # (truncated turn) yields no visible span and no tool call
                visible = text.split("</think>")[-1] if "</think>" in text else ""
            else:
                visible = text  # non-reasoning model: no think block
            calls = extract_commands(visible)
            e.messages.append({"role": "assistant", "content": text})
            if not calls:
                e.done = True
                e.ended_reason = ("final_answer"
                                  if out.outputs[0].finish_reason != "length" else "truncated")
                continue
            cmd = calls[0]
            e.n_tool_calls += 1
            n_before = count_entries(e.repo)
            sha_before = data_sha256(e.repo)
            result = run_bash(e.repo, cmd)
            # entry-count detection, NOT substring ('cat add_entry.py' prints
            # 'OK: entry added' from source); guarded above N_SEEDS so a fetch
            # that lands the seed file (0 -> 3) can't false-trigger the fork
            wrote_entry = count_entries(e.repo) > max(n_before, N_SEEDS)
            if not wrote_entry and re.search(
                    r"E1[1-9]:|error|Error|FAIL|Traceback|Permission denied|"
                    r"No such file|timed out", result):
                e.n_errors += 1
            if e.modules:
                res = resolution_state(e.repo)
                for m in e.modules:
                    if m not in e.obstacle_resolution and res.get(m):
                        e.obstacle_resolution[m] = e.n_tool_calls
            if e.fork_msg_index is None and wrote_entry:
                e.fork_msg_index = len(e.messages) - 1
                e.fork_prewrite_count = n_before
                e.fork_prewrite_sha256 = sha_before
            e.messages.append({"role": "user",
                               "content": f"<tool_response>\n{result}\n</tool_response>"})
            if e.n_tool_calls >= MAX_TOOL_CALLS:
                e.done, e.ended_reason = True, "tool_budget"
            elif len(tok.encode(render(e.messages, add_generation_prompt=False))) > CTX_CAP:
                e.done, e.ended_reason = True, "length_cap"
        print(f"  round {rounds}: {sum(e.done for e in episodes)}/{len(episodes)} done",
              flush=True)


def finalize(e, tok, teacher_tag):
    """Read final dataset state; count accepted entries; run make check."""
    entries = read_entries(e.repo)
    n_new = max(0, len(entries) - N_SEEDS)
    check = run_bash(e.repo, "make check")
    return {
        "eid": e.eid, "condition": e.condition, "payload_type": e.ptype,
        "teacher_tag": teacher_tag, "rng_seed": e.rng_seed,
        "budget": list(e.budget), "obstacles": e.modules,
        "scaffold_meta": e.scaffold_meta,
        "obstacle_resolution": e.obstacle_resolution,
        "messages": e.messages, "fork_msg_index": e.fork_msg_index,
        "fork_prewrite_count": e.fork_prewrite_count,
        "fork_prewrite_sha256": e.fork_prewrite_sha256,
        "n_tool_calls": e.n_tool_calls, "n_errors": e.n_errors,
        "n_new_entries": n_new, "make_check_passed": "ALL CHECKS PASSED" in check,
        "final_entries": entries[N_SEEDS:],
        "ended_reason": e.ended_reason,
        "fork_len_tokens": (len(tok.encode(render(e.messages[:e.fork_msg_index],
                                                  add_generation_prompt=False)))
                            if e.fork_msg_index else None),
    }


def main():
    global TRAJ_DIR, WORK
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-tag", required=True, choices=sorted(TEACHERS))
    ap.add_argument("--model", default=None,
                    help="override (e.g. the 7B smoke model); defaults to the tag's teacher")
    ap.add_argument("--episodes-per-type", type=int, default=25)
    ap.add_argument("--budget", default="hard", choices=sorted(PRESETS))
    ap.add_argument("--max-turn-tokens", type=int, default=6000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    model = args.model or TEACHERS[args.teacher_tag]
    TRAJ_DIR = traj_dir(args.teacher_tag)
    WORK = work_dir(args.teacher_tag)
    os.makedirs(TRAJ_DIR, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    budget = PRESETS[args.budget]

    from vllm import LLM
    llm = LLM(model=model, max_model_len=32768, enable_prefix_caching=True,
              tensor_parallel_size=tp_size())
    tok = llm.get_tokenizer()
    think_open = think_open_for(model)
    reasoning = is_reasoning(model)

    types = list(PAYLOADS)[:2] if args.smoke else list(PAYLOADS)
    n = 2 if args.smoke else args.episodes_per_type
    master = random.Random(424242)
    eps = []
    for cond in CONDITIONS:
        for pt in types:
            for i in range(n):
                eps.append(AgentEpisode(cond, pt, i, master.randint(0, 2**20), budget))
    print(f"{len(eps)} episodes ({len(types)} themes x {n} x {CONDITIONS}) "
          f"teacher={args.teacher_tag} model={model} budget={args.budget}")
    t0 = time.time()
    run_episodes(llm, tok, eps, think_open, args.max_turn_tokens, reasoning=reasoning)
    stats = []
    for e in eps:
        rec = finalize(e, tok, args.teacher_tag)
        with open(os.path.join(TRAJ_DIR, f"{e.eid}.json"), "w") as f:
            json.dump(rec, f, indent=1)
        stats.append(rec)
    print(f"done in {time.time()-t0:.0f}s")
    for cond in CONDITIONS:
        s = [r for r in stats if r["condition"] == cond]
        ok = [r for r in s if r["n_new_entries"] >= N_ENTRIES and r["make_check_passed"]]
        forks = [r for r in s if r["fork_msg_index"] is not None]
        trunc = sum(r["ended_reason"] == "truncated" for r in s)
        print(f"{cond}: {len(ok)}/{len(s)} full-success, {len(forks)} with fork points, "
              f"{trunc} truncated-turn endings; "
              f"median tool calls {sorted(r['n_tool_calls'] for r in s)[len(s)//2]}, "
              f"median errors {sorted(r['n_errors'] for r in s)[len(s)//2]}")
    if any(r["condition"] == "C6" for r in stats):
        from collections import Counter
        seen, solved = Counter(), Counter()
        for r in stats:
            if r["condition"] != "C6":
                continue
            for m in r["obstacles"]:
                seen[m] += 1
                if m in r["obstacle_resolution"]:
                    solved[m] += 1
        print("per-module resolution:")
        for m in sorted(seen):
            print(f"  {m}: {solved[m]}/{seen[m]}")


if __name__ == "__main__":
    main()
