#!/usr/bin/env python3
"""Validate one checkpoint run and write its combined completion manifest.

The hybrid workflow runs some families beside the GPU and Docker-dependent
families on the Mac.  This script is the join point: a checkpoint is complete
only when every expected stage exists, every stage process succeeded, Inspect
transcripts are readable and contain no sample errors, and ToolAlignBench did
not report failed cases.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log


HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())
ALL_EVALS = [
    "agentic_misalignment",
    "instrumental_choices",
    "shutdown_resistance",
    "toolalignbench",
    "odcv_bench",
    "reward_hacking",
    "mask",
    "petri",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--profile", required=True, choices=CONFIG["profiles"])
    parser.add_argument(
        "--evals",
        default=",".join(ALL_EVALS),
        help="Comma-separated evaluation families that define completion.",
    )
    parser.add_argument(
        "--wait-for-running-seconds",
        type=int,
        default=43200,
        help="Wait this long for explicitly running recovery stages to finish.",
    )
    return parser.parse_args()


def expected_stages(selected: list[str]) -> list[str]:
    stages_by_eval = {
        "agentic_misalignment": [
            f"agentic_misalignment/{condition['id']}"
            for condition in CONFIG["agentic_conditions"]
        ],
        "instrumental_choices": ["instrumental_choices"],
        "shutdown_resistance": ["shutdown_resistance"],
        "toolalignbench": ["toolalignbench"],
        "odcv_bench": [
            "odcv_bench/mandated",
            "odcv_bench/incentivized",
            "odcv_bench/judge",
        ],
        "reward_hacking": [
            "reward_hacking/apps",
            "reward_hacking/codecontests",
        ],
        "mask": ["mask"],
        "petri": ["petri"],
    }
    return [stage for family in selected for stage in stages_by_eval[family]]


def inspect_family(path: Path, model_dir: Path) -> str:
    relative = path.relative_to(model_dir)
    return relative.parts[0] if relative.parts else "unknown"


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    selected = list(dict.fromkeys(item.strip() for item in args.evals.split(",") if item.strip()))
    unknown = [item for item in selected if item not in ALL_EVALS]
    if unknown:
        raise ValueError(f"Unknown evaluation families: {', '.join(unknown)}")
    if not selected:
        raise ValueError("At least one evaluation family must be selected")
    stages = expected_stages(selected)
    failures: list[dict[str, Any]] = []
    stage_results: dict[str, Any] = {}

    deadline = time.monotonic() + args.wait_for_running_seconds
    last_running: list[str] | None = None
    while True:
        running: list[str] = []
        for stage in stages:
            status_path = model_dir / stage / "status.json"
            if not status_path.exists():
                continue
            try:
                status = json.loads(status_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if status.get("status") == "running":
                running.append(stage)
        if not running or time.monotonic() >= deadline:
            break
        if running != last_running:
            print(
                json.dumps(
                    {
                        "waiting_for_running_stages": running,
                        "remaining_wait_seconds": round(deadline - time.monotonic()),
                    }
                ),
                flush=True,
            )
            last_running = running
        time.sleep(min(30, max(1, deadline - time.monotonic())))

    for stage in stages:
        status_path = model_dir / stage / "status.json"
        if not status_path.exists():
            failures.append({"type": "missing_stage_status", "stage": stage})
            continue
        try:
            status = json.loads(status_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            failures.append(
                {"type": "unreadable_stage_status", "stage": stage, "error": repr(error)}
            )
            continue
        stage_results[stage] = {
            "status": status.get("status"),
            "return_code": status.get("return_code"),
            "elapsed_seconds": status.get("elapsed_seconds"),
        }
        if status.get("status") != "complete" or status.get("return_code") != 0:
            failures.append(
                {
                    "type": "incomplete_stage",
                    "stage": stage,
                    "status": status.get("status"),
                    "return_code": status.get("return_code"),
                }
            )

    sample_counts: dict[str, int] = {}
    inspect_logs = 0
    for path in sorted(model_dir.rglob("*.eval")):
        family = inspect_family(path, model_dir)
        try:
            log = read_eval_log(str(path))
        except Exception as error:
            failures.append(
                {
                    "type": "unreadable_inspect_log",
                    "log": str(path.relative_to(model_dir)),
                    "error": repr(error),
                }
            )
            continue
        inspect_logs += 1
        samples = log.samples or []
        sample_counts[family] = sample_counts.get(family, 0) + len(samples)
        for sample in samples:
            if getattr(sample, "error", None) is not None:
                failures.append(
                    {
                        "type": "inspect_sample_error",
                        "log": str(path.relative_to(model_dir)),
                        "sample_id": str(getattr(sample, "id", None)),
                        "error": str(sample.error),
                    }
                )

    profile = CONFIG["profiles"][args.profile]
    all_exact_counts = {
        "agentic_misalignment": len(CONFIG["agentic_conditions"])
        * profile["agentic_epochs"],
        "instrumental_choices": 7 * 8 * profile["instrumental_repeats"],
        "shutdown_resistance": profile["shutdown_samples"],
        "reward_hacking": 2 * profile["reward_samples"],
        "mask": profile["mask_samples"],
        "petri": profile["petri_seeds"],
    }
    exact_counts = {
        family: count
        for family, count in all_exact_counts.items()
        if family in selected and count > 0
    }
    for family, expected in exact_counts.items():
        actual = sample_counts.get(family, 0)
        if actual != expected:
            failures.append(
                {
                    "type": "unexpected_inspect_sample_count",
                    "family": family,
                    "expected": expected,
                    "actual": actual,
                }
            )

    if "toolalignbench" in selected:
        metrics_path = model_dir / "toolalignbench" / "transcripts" / "metrics.json"
        expected_toolalign = 128 * profile["toolalign_repeats"]
        actual_toolalign = len(
            list((model_dir / "toolalignbench" / "transcripts").glob("*.md"))
        )
        if actual_toolalign != expected_toolalign:
            failures.append(
                {
                    "type": "unexpected_toolalignbench_case_count",
                    "expected": expected_toolalign,
                    "actual": actual_toolalign,
                }
            )
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            if metrics.get("errors") != 0 or not metrics.get("successes"):
                failures.append(
                    {
                        "type": "toolalignbench_case_errors",
                        "successes": metrics.get("successes"),
                        "errors": metrics.get("errors"),
                    }
                )
        else:
            failures.append({"type": "missing_toolalignbench_metrics"})

    validation = {
        "schema_version": 1,
        "validated_at": datetime.now(UTC).isoformat(),
        "valid": not failures,
        "selected_evals": selected,
        "excluded_evals": [item for item in ALL_EVALS if item not in selected],
        "expected_stages": stages,
        "stage_results": stage_results,
        "inspect_logs": inspect_logs,
        "inspect_sample_counts": sample_counts,
        "failures": failures,
    }
    validation_path = model_dir / "validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")

    manifest_path = model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update(
        {
            "schema_version": 1,
            "model_label": model_dir.name,
            "model": CONFIG["models"].get(model_dir.name),
            "base_model": CONFIG["base_model"],
            "quantization": None,
            "dtype": "bfloat16",
            "profile": args.profile,
            "profile_parameters": profile,
            "selected_evals": selected,
            "excluded_evals": [item for item in ALL_EVALS if item not in selected],
            "upstreams": CONFIG["upstreams"],
            "validated_at": validation["validated_at"],
            "validation": "validation.json",
            "status": "complete" if validation["valid"] else "partial",
            "failed_stages": [
                item.get("stage", item.get("family", item.get("type")))
                for item in failures
            ],
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(validation, indent=2, sort_keys=True))
    raise SystemExit(0 if validation["valid"] else 1)


if __name__ == "__main__":
    main()
