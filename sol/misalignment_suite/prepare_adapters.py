#!/usr/bin/env python3
"""Download and validate the two PEFT adapters used by the evaluation suite."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
CONFIG_PATH = HERE / "config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "sol" / "loras")
    parser.add_argument("--token", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(CONFIG_PATH.read_text())
    args.output_root.mkdir(parents=True, exist_ok=True)

    for label, model in config["models"].items():
        adapter_repo = model["adapter"]
        if adapter_repo is None:
            continue

        raw_dir = (args.output_root / label).resolve()
        converted_dir = raw_dir / "vllm"
        raw_weights = raw_dir / "adapter_model.safetensors"
        converted_weights = converted_dir / "adapter_model.safetensors"

        if not raw_weights.exists():
            snapshot_download(
                repo_id=adapter_repo,
                local_dir=raw_dir,
                token=args.token,
                allow_patterns=["adapter_config.json", "adapter_model.safetensors"],
            )

        if not converted_weights.exists():
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "sol" / "convert_peft_lora_for_vllm_qwen35.py"),
                    str(raw_dir),
                    str(converted_dir),
                ],
                check=True,
            )

        print(
            json.dumps(
                {
                    "model": label,
                    "repository": adapter_repo,
                    "raw": str(raw_dir),
                    "converted": str(converted_dir),
                    "ready": converted_weights.exists(),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
