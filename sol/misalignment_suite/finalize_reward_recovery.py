#!/usr/bin/env python3
"""Validate and join a preserved reward-hacking partial run and its recovery."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage_dir", type=Path)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--model", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage = args.stage_dir.resolve()
    logs = sorted(stage.rglob("*.eval"))
    if not logs:
        raise RuntimeError(f"No evaluation logs found below {stage}")

    samples: dict[str, Any] = {}
    duplicates: list[str] = []
    errors: list[dict[str, str]] = []
    log_statuses: dict[str, str] = {}
    recovery_success = False
    for path in logs:
        log = read_eval_log(str(path))
        relative = str(path.relative_to(stage))
        log_statuses[relative] = str(log.status)
        if "recovery" in path.parts and log.status == "success":
            recovery_success = True
        for sample in log.samples or []:
            key = f"{sample.id}:{getattr(sample, 'epoch', 1)}"
            if getattr(sample, "error", None) is not None:
                errors.append(
                    {"log": relative, "sample": key, "error": str(sample.error)}
                )
                continue
            if key in samples:
                duplicates.append(key)
                continue
            samples[key] = sample

    failures: list[str] = []
    if not recovery_success:
        failures.append("no successful recovery log")
    if errors:
        failures.append(f"{len(errors)} sample errors")
    if duplicates:
        failures.append(f"{len(duplicates)} duplicate sample IDs")
    if len(samples) != args.expected_samples:
        failures.append(
            f"expected {args.expected_samples} unique successful samples, got {len(samples)}"
        )

    metric_counts: defaultdict[str, int] = defaultdict(int)
    group_counts: dict[str, defaultdict[str, int]] = {}
    scored = 0
    for sample in samples.values():
        score_values = [score.value for score in (sample.scores or {}).values()]
        value = next((item for item in score_values if isinstance(item, dict)), None)
        if value is None:
            failures.append(f"sample {sample.id} has no dictionary-valued score")
            continue
        scored += 1
        for name in ("passed", "actually_solved", "reward_hacked"):
            if value.get(name) == "C":
                metric_counts[name] += 1
        metadata = next(iter((sample.scores or {}).values())).metadata or {}
        group = str(metadata.get("hack_group", "unknown"))
        group_row = group_counts.setdefault(group, defaultdict(int))
        group_row["total"] += 1
        for name in ("passed", "actually_solved", "reward_hacked"):
            if value.get(name) == "C":
                group_row[name] += 1

    if scored != args.expected_samples:
        failures.append(f"expected {args.expected_samples} scored samples, got {scored}")

    completed_at = datetime.now(UTC).isoformat()
    validation = {
        "schema_version": 1,
        "validated_at": completed_at,
        "valid": not failures,
        "expected_samples": args.expected_samples,
        "unique_successful_samples": len(samples),
        "log_statuses": log_statuses,
        "duplicates": duplicates,
        "sample_errors": errors,
        "failures": failures,
    }
    atomic_json(stage / "recovery-validation.json", validation)
    if failures:
        raise RuntimeError("; ".join(failures))

    total = len(samples)
    summary = {
        "schema_version": 1,
        "model": args.model,
        "generated_at": completed_at,
        "source": "validated union of preserved partial log and targeted recovery log",
        "counts": {
            "total": total,
            "passed": metric_counts["passed"],
            "actually_solved": metric_counts["actually_solved"],
            "reward_hacked": metric_counts["reward_hacked"],
            "failed": total - metric_counts["passed"],
        },
        "metrics": {
            "accuracy": metric_counts["passed"] / total,
            "actually_solved_rate": metric_counts["actually_solved"] / total,
            "reward_hacked_rate": metric_counts["reward_hacked"] / total,
        },
        "group_counts": {key: dict(value) for key, value in sorted(group_counts.items())},
        "validation": "recovery-validation.json",
    }
    atomic_json(stage / "combined_summary.json", summary)

    old_status_path = stage / "status.json"
    old_status = json.loads(old_status_path.read_text()) if old_status_path.exists() else {}
    status = {
        **old_status,
        "stage": "reward_hacking/apps",
        "model": args.model,
        "status": "complete",
        "return_code": 0,
        "completed_at": completed_at,
        "recovered_from_partial_run": True,
        "validated_samples": total,
        "recovery_validation": str(stage / "recovery-validation.json"),
        "combined_summary": str(stage / "combined_summary.json"),
    }
    atomic_json(old_status_path, status)
    print(json.dumps({"status": "complete", "samples": total, "stage": str(stage)}))


if __name__ == "__main__":
    main()
