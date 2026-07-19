#!/usr/bin/env python3
"""Rewrite text-only Qwen3.5/3.6 PEFT names for vLLM's VL wrapper.

Transformers exposes the language backbone as ``model.layers`` when training
through its causal-LM class. The official checkpoint and vLLM multimodal wrapper
instead map it through ``model.language_model``. Tensor values are unchanged.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


OLD_PREFIX = "base_model.model.model."
NEW_PREFIX = "base_model.model.model.language_model."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    if source == destination:
        raise ValueError("Destination must differ from source")

    source_weights = source / "adapter_model.safetensors"
    source_config = source / "adapter_config.json"
    if not source_weights.exists() or not source_config.exists():
        raise FileNotFoundError(f"Incomplete PEFT adapter at {source}")

    destination.mkdir(parents=True, exist_ok=True)
    tensors = {}
    with safe_open(source_weights, framework="pt", device="cpu") as file:
        metadata = file.metadata()
        for old_name in file.keys():
            if not old_name.startswith(OLD_PREFIX):
                raise ValueError(f"Unexpected adapter key: {old_name}")
            new_name = NEW_PREFIX + old_name.removeprefix(OLD_PREFIX)
            if new_name in tensors:
                raise ValueError(f"Duplicate converted key: {new_name}")
            tensors[new_name] = file.get_tensor(old_name)

    destination_weights = destination / "adapter_model.safetensors"
    save_file(tensors, destination_weights, metadata=metadata)
    shutil.copy2(source_config, destination / "adapter_config.json")

    with safe_open(source_weights, framework="pt", device="cpu") as source_file:
        with safe_open(destination_weights, framework="pt", device="cpu") as destination_file:
            for old_name in source_file.keys():
                new_name = NEW_PREFIX + old_name.removeprefix(OLD_PREFIX)
                if not torch.equal(source_file.get_tensor(old_name), destination_file.get_tensor(new_name)):
                    raise ValueError(f"Tensor values changed during conversion: {old_name}")

    config = json.loads(source_config.read_text())
    print(
        json.dumps(
            {
                "source": str(source),
                "destination": str(destination),
                "tensors": len(tensors),
                "rank": config.get("r"),
                "alpha": config.get("lora_alpha"),
                "old_prefix": OLD_PREFIX,
                "new_prefix": NEW_PREFIX,
                "validated_equal": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
