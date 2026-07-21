"""Shared chat formatting for Qwen3 — manual, byte-stable, thinking-preserving.

We deliberately do NOT use tokenizer.apply_chat_template: Qwen templates strip
prior-turn <think> blocks on re-templating, which would silently gut the
conditioning treatment. Manual rendering keeps stored transcripts byte-stable.
"""

import os

SYSTEM = "You are a helpful assistant."
NOTHINK_PREFILL = "<think>\n\n</think>\n\n"

SAMPLER = dict(temperature=1.0, top_p=0.95, top_k=20)  # one sampler everywhere


def tp_size():
    """vLLM tensor-parallel degree, from GCST_TP (default 1). Set GCST_TP=2 to
    shard a 32B model across a 2x A100-80GB pod (it won't fit on one)."""
    return int(os.environ.get("GCST_TP", "1"))


def think_open_for(model: str) -> str:
    """Qwen3.6 templates force '<think>\\n' open in the generation prompt; Qwen3 lets the
    model emit <think> itself; Olmo 3 Think forces '<think>' with no trailing newline.
    The forced prefill must also be prepended to the stored assistant content so
    re-rendering matches the native preserve-thinking format."""
    if "olmo" in model.lower():
        return "<think>"
    return "<think>\n" if ("3.6" in model or "3_5" in model.lower()) else ""


def stop_token_ids_for(tok):
    """Olmo Think's eos is <|endoftext|>; mid-conversation assistant turns end with
    <|im_end|>, which is NOT an eos token there — without it in stop_token_ids vLLM
    generations run through the turn boundary into fabricated user turns. Pass this
    to every SamplingParams that generates text (no-op for Qwen, whose eos IS
    <|im_end|>). Scoring calls (max_tokens=1 / prompt_logprobs) don't need it."""
    ids = {tok.eos_token_id}
    im = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im, int) and im is not None and im >= 0:
        ids.add(im)
    return sorted(i for i in ids if i is not None)


def render(messages, add_generation_prompt=True, nothink=False, think_open=""):
    """messages: [{role, content}]; assistant content includes its <think> block verbatim."""
    s = ""
    for m in messages:
        s += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    if add_generation_prompt:
        s += "<|im_start|>assistant\n"
        s += NOTHINK_PREFILL if nothink else think_open
    return s


def strip_think(text):
    """Visible part of an assistant turn (after the final </think>)."""
    return text.split("</think>")[-1].strip()
