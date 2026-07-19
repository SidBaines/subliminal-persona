"""Register capability metadata for the local Qwen wire-model name.

Recent Inspect releases treat any unknown name sent through the OpenAI provider
as a pre-release frontier OpenAI model. That makes ``qwen-eval`` inherit GPT-5
request rules (including the Responses API and rejection of temperature/seed).
The family override changes capability detection only; the wire identifier sent
to vLLM remains ``qwen-eval``.
"""

try:
    from inspect_ai.model import ModelInfo, set_model_info

    set_model_info(
        "qwen-eval",
        ModelInfo(
            model="Qwen3.6-27B",
            family="gpt-4",
            context_length=32768,
            output_tokens=10000,
            reasoning=True,
        ),
    )
except (ImportError, TypeError):
    # The two older pinned Inspect environments predate ModelInfo.family and do
    # not exhibit the unknown-model GPT-5 heuristic.
    pass
