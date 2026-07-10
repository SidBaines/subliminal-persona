"""Shared chat formatting for Qwen3 — manual, byte-stable, thinking-preserving.

We deliberately do NOT use tokenizer.apply_chat_template: Qwen templates strip
prior-turn <think> blocks on re-templating, which would silently gut the
conditioning treatment. Manual rendering keeps stored transcripts byte-stable.
"""

SYSTEM = "You are a helpful assistant."
NOTHINK_PREFILL = "<think>\n\n</think>\n\n"

SAMPLER = dict(temperature=1.0, top_p=0.95, top_k=20)  # one sampler everywhere


def think_open_for(model: str) -> str:
    """Qwen3.6 templates force '<think>\\n' open in the generation prompt; Qwen3 lets the
    model emit <think> itself. The forced prefill must also be prepended to the stored
    assistant content so re-rendering matches the native preserve-thinking format."""
    return "<think>\n" if ("3.6" in model or "3_5" in model.lower()) else ""


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
