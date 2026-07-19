#!/usr/bin/env python3
"""Replace one failed Instrumental Choices sample with a targeted retry."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log, write_eval_log


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage_dir", type=Path)
    parser.add_argument("upstream_log_root", type=Path)
    parser.add_argument("upstream_repo", type=Path)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--model", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage = args.stage_dir.resolve()
    source = args.upstream_log_root.resolve()
    upstream_repo = args.upstream_repo.resolve()
    recovery_logs = sorted((stage / "recovery").glob("*.eval"))
    if len(recovery_logs) != 1:
        raise RuntimeError(f"Expected one recovery log, found {len(recovery_logs)}")
    recovery_log = read_eval_log(str(recovery_logs[0]))
    recovery_samples = recovery_log.samples or []
    if recovery_log.status != "success" or len(recovery_samples) != 1:
        raise RuntimeError(
            f"Recovery is not one successful sample: status={recovery_log.status}, "
            f"samples={len(recovery_samples)}"
        )
    replacement = recovery_samples[0]
    if getattr(replacement, "error", None) is not None:
        raise RuntimeError(f"Recovery sample errored: {replacement.error}")

    destination = stage / "upstream_logs"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)

    failed: list[tuple[Path, int, Any]] = []
    for path in sorted(destination.rglob("*.eval")):
        log = read_eval_log(str(path))
        for index, sample in enumerate(log.samples or []):
            if getattr(sample, "error", None) is not None:
                failed.append((path, index, sample))
    if len(failed) != 1:
        raise RuntimeError(f"Expected exactly one original sample error, found {len(failed)}")

    failed_path, failed_index, original = failed[0]
    if str(original.id) != str(replacement.id):
        raise RuntimeError(
            f"Recovery ID {replacement.id} does not match failed ID {original.id}"
        )

    failed_archive = stage / "failed_attempts" / "context-overflow"
    failed_archive.mkdir(parents=True, exist_ok=True)
    shutil.copy2(failed_path, failed_archive / f"{failed_path.name}.failed-source")

    original_log = read_eval_log(str(failed_path))
    repaired_samples = list(original_log.samples or [])
    repaired_samples[failed_index] = replacement
    repaired_results = original_log.results
    if repaired_results is not None:
        repaired_scores = [
            score.model_copy(
                update={
                    "scored_samples": len(repaired_samples),
                    "unscored_samples": 0,
                }
            )
            for score in repaired_results.scores
        ]
        repaired_results = repaired_results.model_copy(
            update={
                "total_samples": len(repaired_samples),
                "completed_samples": len(repaired_samples),
                "scores": repaired_scores,
            }
        )
    repaired_log = original_log.model_copy(
        update={"samples": repaired_samples, "results": repaired_results, "status": "success"}
    )
    write_eval_log(repaired_log, failed_path)

    sample_count = 0
    remaining_errors: list[str] = []
    for path in sorted(destination.rglob("*.eval")):
        log = read_eval_log(str(path))
        for sample in log.samples or []:
            sample_count += 1
            if getattr(sample, "error", None) is not None:
                remaining_errors.append(f"{path}:{sample.id}:{sample.error}")
    failures: list[str] = []
    if sample_count != args.expected_samples:
        failures.append(f"expected {args.expected_samples} samples, got {sample_count}")
    if remaining_errors:
        failures.append(f"{len(remaining_errors)} remaining sample errors")
    if failures:
        raise RuntimeError("; ".join(failures))

    # The upstream deterministic analysis reads only the copied .eval logs.
    # Rebuild it after replacing the failed sample so tables and report agree.
    analysis_dir = destination / "tool_analysis"
    subprocess.run(
        [
            str(upstream_repo / ".venv" / "bin" / "python"),
            str(upstream_repo / "scripts" / "analyze_tool_traces.py"),
            str(destination),
            "--output-dir",
            str(analysis_dir),
        ],
        cwd=upstream_repo,
        check=True,
    )
    subprocess.run(
        [
            str(upstream_repo / ".venv" / "bin" / "python"),
            str(upstream_repo / "scripts" / "render_tool_trace_report.py"),
            str(analysis_dir),
        ],
        cwd=upstream_repo,
        check=True,
    )

    # Retain the raw recovery log but make its extension non-discoverable so
    # model-level validation sees the repaired 168-sample set exactly once.
    recovery_source = recovery_logs[0].with_suffix(".eval.recovery-source")
    recovery_logs[0].rename(recovery_source)

    completed_at = datetime.now(UTC).isoformat()
    validation = {
        "schema_version": 1,
        "validated_at": completed_at,
        "valid": True,
        "expected_samples": args.expected_samples,
        "samples": sample_count,
        "replaced_sample_id": str(replacement.id),
        "original_failed_log": str(failed_path.relative_to(stage)),
        "failed_source": str((failed_archive / f"{failed_path.name}.failed-source").relative_to(stage)),
        "recovery_source": str(recovery_source.relative_to(stage)),
        "remaining_sample_errors": 0,
    }
    atomic_json(stage / "recovery-validation.json", validation)

    status_path = stage / "status.json"
    old_status = json.loads(status_path.read_text()) if status_path.exists() else {}
    status = {
        **old_status,
        "stage": "instrumental_choices",
        "model": args.model,
        "status": "complete",
        "return_code": 0,
        "completed_at": completed_at,
        "recovered_from_targeted_retry": True,
        "validated_samples": sample_count,
        "recovery_validation": str(stage / "recovery-validation.json"),
    }
    atomic_json(status_path, status)
    print(json.dumps({"status": "complete", "samples": sample_count, "stage": str(stage)}))


if __name__ == "__main__":
    main()
