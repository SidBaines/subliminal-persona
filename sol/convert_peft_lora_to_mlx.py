#!/usr/bin/env python3
"""Convert a Hugging Face PEFT LoRA adapter for Qwen3.6 to MLX-LM format."""

import argparse
import json
from pathlib import Path

import mlx.core as mx


PEFT_PREFIX = "base_model.model.model.layers."
MLX_PREFIX = "language_model.model.layers."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Directory containing the PEFT adapter")
    parser.add_argument("destination", type=Path, help="Directory for the MLX-LM adapter")
    return parser.parse_args()


def convert(source: Path, destination: Path) -> None:
    with (source / "adapter_config.json").open() as file:
        peft_config = json.load(file)

    if peft_config.get("use_dora", False):
        raise ValueError("DoRA adapters are not supported by this converter")
    if peft_config.get("bias", "none") != "none":
        raise ValueError("Adapters with trained biases are not supported")
    if peft_config.get("modules_to_save"):
        raise ValueError("Adapters with additional saved modules are not supported")

    rank = peft_config["r"]
    scale = peft_config["lora_alpha"] / rank
    peft_weights = mx.load(str(source / "adapter_model.safetensors"))
    mlx_weights = {}
    module_paths = set()
    layer_numbers = set()

    for old_name, weight in peft_weights.items():
        if not old_name.startswith(PEFT_PREFIX):
            raise ValueError(f"Unexpected tensor name: {old_name}")

        relative_name = old_name.removeprefix(PEFT_PREFIX)
        layer_text, module_and_suffix = relative_name.split(".", 1)
        layer_numbers.add(int(layer_text))

        if module_and_suffix.endswith(".lora_A.weight"):
            module_path = module_and_suffix.removesuffix(".lora_A.weight")
            new_suffix = ".lora_a"
        elif module_and_suffix.endswith(".lora_B.weight"):
            module_path = module_and_suffix.removesuffix(".lora_B.weight")
            new_suffix = ".lora_b"
        else:
            raise ValueError(f"Unexpected LoRA tensor suffix: {old_name}")

        module_paths.add(module_path)
        new_name = f"{MLX_PREFIX}{layer_text}.{module_path}{new_suffix}"
        mlx_weights[new_name] = weight.T

    expected_layers = set(range(max(layer_numbers) + 1))
    if layer_numbers != expected_layers:
        missing = sorted(expected_layers - layer_numbers)
        raise ValueError(f"Adapter does not cover a contiguous set of layers; missing {missing}")

    destination.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(destination / "adapters.safetensors"), mlx_weights)

    mlx_config = {
        "fine_tune_type": "lora",
        "num_layers": len(layer_numbers),
        "lora_parameters": {
            "rank": rank,
            "dropout": peft_config.get("lora_dropout", 0.0),
            "scale": scale,
            "keys": sorted(module_paths),
        },
    }
    with (destination / "adapter_config.json").open("w") as file:
        json.dump(mlx_config, file, indent=2)
        file.write("\n")

    print(
        f"Converted {len(mlx_weights)} tensors across {len(layer_numbers)} layers "
        f"to {destination}"
    )


if __name__ == "__main__":
    arguments = parse_args()
    convert(arguments.source, arguments.destination)
