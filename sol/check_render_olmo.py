"""Pre-run gates 1-3 for the Olmo swap (cheap; tokenizer-only unless --generate).

Gate 2a: our manual render() matches Olmo's chat template for the generation
         prompt (must end exactly '<|im_start|>assistant\\n<think>').
Gate 2b: render(messages[:k]) is a byte-prefix of render(messages) on a
         multi-turn transcript containing <think> blocks (the fork-point
         invariant everything depends on).
Gate 3 (--generate, needs GPU): one short vLLM generation on the smoke model
         finishes via a stop token (im_end/eos), not by rambling to max length.

Usage: .venv/bin/python sol/check_render_olmo.py [--model allenai/Olmo-3-7B-Think] [--generate]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arms import SMOKE_MODEL
from common import SAMPLER, SYSTEM, render, stop_token_ids_for, think_open_for

MSGS = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "What is 2+2?"},
]
MULTI = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "List the files."},
    {"role": "assistant", "content": "<think>\nI should run ls.\n</think>\nRunning ls now."},
    {"role": "user", "content": "<tool_response>\na.py\nb.py\n</tool_response>"},
    {"role": "assistant", "content": "<think>\nTwo files.\n</think>\nThere are two files."},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=SMOKE_MODEL)
    ap.add_argument("--generate", action="store_true", help="also run gate 3 (needs GPU)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    think_open = think_open_for(args.model)
    ok = True

    # ---- gate 2a: generation-prompt suffix + template agreement
    ours = render(MSGS, add_generation_prompt=True, think_open=think_open)
    want_suffix = "<|im_start|>assistant\n<think>"
    if not ours.endswith(want_suffix):
        print(f"FAIL 2a: render() ends {ours[-40:]!r}, expected suffix {want_suffix!r}")
        ok = False
    try:
        native = tok.apply_chat_template(MSGS, tokenize=False, add_generation_prompt=True)
        # Olmo's template may inject a functions/tools fallback section into the system
        # turn; we deliberately don't use it (same stance as not using apply_chat_template
        # at all). Compare everything from the user turn onward, which must match exactly.
        cut = "<|im_start|>user"
        if native.split(cut, 1)[1] != ours.split(cut, 1)[1]:
            print("FAIL 2a: render() diverges from apply_chat_template after the system turn")
            print("  native:", repr(cut + native.split(cut, 1)[1]))
            print("  ours:  ", repr(cut + ours.split(cut, 1)[1]))
            ok = False
        else:
            print("PASS 2a: generation prompt matches native template (post-system)")
    except Exception as e:  # template variants that need extra kwargs etc.
        print(f"WARN 2a: could not run apply_chat_template ({e}); suffix check only")

    # ---- gate 2b: byte-prefix stability at fork points, thinks preserved
    full = render(MULTI, add_generation_prompt=False)
    for k in range(1, len(MULTI)):
        part = render(MULTI[:k], add_generation_prompt=False)
        if not full.startswith(part):
            print(f"FAIL 2b: render(messages[:{k}]) is not a byte-prefix of the full render")
            ok = False
    if "<think>\nI should run ls.\n</think>" not in full:
        print("FAIL 2b: prior-turn <think> block not preserved verbatim")
        ok = False
    if ok:
        print("PASS 2b: byte-prefix stability + think preservation")

    # ---- stop-token sanity (id-level; behavioral check under --generate)
    stops = stop_token_ids_for(tok)
    im = tok.convert_tokens_to_ids("<|im_end|>")
    print(f"stop ids: {stops} (eos={tok.eos_token_id}, im_end={im})")
    if im not in stops or tok.eos_token_id not in stops:
        print("FAIL: stop_token_ids_for() is missing eos or <|im_end|>")
        ok = False

    if args.generate:
        from vllm import LLM, SamplingParams
        llm = LLM(model=args.model, max_model_len=8192)
        out = llm.generate(
            [render(MSGS, add_generation_prompt=True, think_open=think_open)],
            SamplingParams(**SAMPLER, max_tokens=4096, seed=0, stop_token_ids=stops))[0]
        fin = out.outputs[0].finish_reason
        print(f"gate 3: finish_reason={fin}; tail={out.outputs[0].text[-80:]!r}")
        if fin != "stop":
            print("FAIL 3: generation did not end on a stop token")
            ok = False
        else:
            print("PASS 3: generation stops at the turn boundary")

    print("ALL GATES PASSED" if ok else "GATES FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
