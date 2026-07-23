#!/usr/bin/env python3
"""Run the pinned misalignment evaluation suite against one served checkpoint.

The model server must expose an OpenAI-compatible endpoint. A fresh server is
used for every checkpoint by ``run_remote.sh`` because these Qwen adapters must
not be hot-swapped in a shared vLLM process.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
CONFIG_PATH = HERE / "config.json"
ALL_EVALS = [
    "agentic_misalignment",
    "instrumental_choices",
    "shutdown_resistance",
    "toolalignbench",
    "odcv_bench",
    "reward_hacking",
    "mask",
    "petri",
    "evilgenie",
    "impossiblebench",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-label", required=True, choices=["base", "c6", "c6e"])
    parser.add_argument("--profile", default="study", choices=["smoke", "study", "paper"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--docker-api-base", default="http://172.17.0.1:8000/v1")
    parser.add_argument("--suite-root", type=Path, default=Path("/workspace/misalignment-suite"))
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "sol" / "results" / "misalignment_suite",
    )
    parser.add_argument(
        "--evals",
        default=",".join(ALL_EVALS),
        help=f"Comma-separated subset of: {', '.join(ALL_EVALS)}",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--parallel-families",
        action="store_true",
        help="Run the selected independent evaluation families concurrently.",
    )
    return parser.parse_args()


def now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def shell_join(command: list[str]) -> str:
    return shlex.join(str(part) for part in command)


class SuiteRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config = json.loads(CONFIG_PATH.read_text())
        self.profile = self.config["profiles"][args.profile]
        self.upstreams = args.suite_root / "upstreams"
        self.main_python = args.suite_root / "venvs" / "main" / "bin" / "python"
        self.main_inspect = args.suite_root / "venvs" / "main" / "bin" / "inspect"
        self.output = args.results_root / args.run_id / args.model_label
        self.output.mkdir(parents=True, exist_ok=True)
        self.base_env = os.environ.copy()
        sitecustomize_dir = HERE / "inspect_sitecustomize"
        existing_pythonpath = self.base_env.get("PYTHONPATH")
        self.base_env["PYTHONPATH"] = str(sitecustomize_dir) + (
            os.pathsep + existing_pythonpath if existing_pythonpath else ""
        )
        default_hf_home = (
            Path("/workspace/hf-cache")
            if str(args.suite_root).startswith("/workspace/")
            else args.suite_root / "hf-cache"
        )
        self.base_env.setdefault("HF_HOME", str(default_hf_home))
        self.base_env.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.base_env.setdefault("LOCAL_API_KEY", "local")
        self.base_env["LOCAL_API_BASE_URL"] = args.api_base
        if self.base_env.get("ANTHROPIC_KEY") and not self.base_env.get("ANTHROPIC_API_KEY"):
            self.base_env["ANTHROPIC_API_KEY"] = self.base_env["ANTHROPIC_KEY"]
        if self.base_env.get("HF_TOKEN") and not self.base_env.get("HUGGINGFACE_TOKEN"):
            self.base_env["HUGGINGFACE_TOKEN"] = self.base_env["HF_TOKEN"]
        self.failures: list[str] = []

    def check_server(self) -> None:
        if self.args.dry_run:
            return
        request = urllib.request.Request(
            self.args.api_base + "/models", headers={"Authorization": "Bearer local"}
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.load(response)
        ids = [item["id"] for item in payload.get("data", [])]
        if self.config["served_model"] not in ids:
            raise RuntimeError(f"Server returned model ids {ids}; expected qwen-eval")

    def status_path(self, stage: str) -> Path:
        return self.output / stage / "status.json"

    def is_complete(self, stage: str) -> bool:
        path = self.status_path(stage)
        if not path.exists():
            return False
        try:
            if json.loads(path.read_text()).get("status") != "complete":
                return False
            if stage == "toolalignbench":
                metrics_path = self.output / stage / "transcripts" / "metrics.json"
                if not metrics_path.exists():
                    return False
                metrics = json.loads(metrics_path.read_text())
                transcript_count = len(
                    list((self.output / stage / "transcripts").glob("*.md"))
                )
                expected = 128 * self.profile["toolalign_repeats"]
                return (
                    metrics.get("errors") == 0
                    and bool(metrics.get("successes"))
                    and transcript_count == expected
                )
            return True
        except (json.JSONDecodeError, OSError):
            return False

    def command(
        self,
        stage: str,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str] | None = None,
    ) -> bool:
        if self.is_complete(stage):
            print(f"SKIP {stage}: already complete", flush=True)
            return True

        stage_dir = self.output / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        status = {
            "stage": stage,
            "model": self.args.model_label,
            "profile": self.args.profile,
            "status": "dry_run" if self.args.dry_run else "running",
            "started_at": now(),
            "cwd": str(cwd),
            "command": shell_join(command),
        }
        atomic_json(self.status_path(stage), status)
        print(f"\nSTART {stage}\n{status['command']}", flush=True)
        if self.args.dry_run:
            return True

        started = time.monotonic()
        log_path = stage_dir / "runner.log"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{now()}] {status['command']}\n")
            log.flush()
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=env or self.base_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )

        status.update(
            {
                "status": "complete" if completed.returncode == 0 else "failed",
                "completed_at": now(),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "return_code": completed.returncode,
                "log": str(log_path),
            }
        )
        atomic_json(self.status_path(stage), status)
        print(
            f"END {stage}: {status['status']} in {status['elapsed_seconds']}s",
            flush=True,
        )
        if completed.returncode != 0:
            self.failures.append(stage)
            if self.args.fail_fast:
                raise subprocess.CalledProcessError(completed.returncode, command)
            return False
        return True

    def inspect_target_args(self) -> list[str]:
        return [
            "--model",
            f"openai/{self.config['served_model']}",
            "--model-base-url",
            self.args.api_base,
            "-M",
            "api_key=local",
            "-M",
            "responses_api=false",
        ]

    def agentic_misalignment(self) -> None:
        repo = self.upstreams / "inspect-evals"
        epochs = self.profile["agentic_epochs"]
        for condition in self.config["agentic_conditions"]:
            stage = f"agentic_misalignment/{condition['id']}"
            command = [
                str(self.main_inspect),
                "eval",
                "inspect_evals/agentic_misalignment",
                *self.inspect_target_args(),
                "--epochs",
                str(epochs),
                "--temperature",
                "1.0",
                "--max-tokens",
                "10000",
                "--max-connections",
                "8",
                "--seed",
                "20260717",
                "--log-dir",
                str(self.output / stage / "transcripts"),
                "--retry-on-error=2",
                "-T",
                f"scenario={condition['scenario']}",
                "-T",
                f"goal_type={condition['goal_type']}",
                "-T",
                f"goal_value={condition['goal_value']}",
                "-T",
                f"urgency_type={condition['urgency_type']}",
                "-T",
                "grader_model=anthropic/claude-sonnet-4-6",
            ]
            self.command(stage, command, cwd=repo)

    def instrumental_choices(self) -> None:
        stage = "instrumental_choices"
        repo = self.upstreams / "instrumental-choices"
        env = self.base_env.copy()
        env.update(
            {
                "OPENAI_BASE_URL": self.args.api_base,
                "OPENAI_API_KEY": "local",
                "INSPECT_EVAL_MODEL_ARGS": "api_key=local",
                "IC_REPEATS": str(self.profile["instrumental_repeats"]),
                "IC_ENABLE_TOOL_TRACE_AUDIT": "0",
                "IC_MAX_SAMPLES": "4",
                "IC_MAX_CONNECTIONS": "4",
                "IC_MAX_SUBPROCESSES": "4",
                "IC_TASK_PARALLELISM": "2",
                "IC_SAMPLE_RETRY_ON_ERROR": "1",
            }
        )
        colima_socket = Path.home() / ".colima" / "default" / "docker.sock"
        if not env.get("DOCKER_HOST") and colima_socket.exists():
            # Instrumental Choices replaces HOME for each Inspect subprocess,
            # which otherwise hides both Docker's selected Colima context and
            # the Compose CLI plugin installed under the real home directory.
            env["DOCKER_HOST"] = f"unix://{colima_socket}"
            env["DOCKER_CONFIG"] = str(Path.home() / ".docker")
        upstream_logs = repo / "logs"
        existing_log_roots = (
            {path.resolve() for path in upstream_logs.iterdir() if path.is_dir()}
            if upstream_logs.exists()
            else set()
        )
        ok = self.command(
            stage,
            [str(repo / "scripts" / "run_full_eval.sh"), "openai/qwen-eval"],
            cwd=repo,
            env=env,
        )
        if ok and not self.args.dry_run:
            new_log_roots = sorted(
                (
                    path
                    for path in upstream_logs.iterdir()
                    if path.is_dir() and path.resolve() not in existing_log_roots
                ),
                key=lambda path: path.stat().st_mtime,
            )
            if len(new_log_roots) != 1:
                raise RuntimeError(
                    "Expected exactly one new Instrumental Choices log root; "
                    f"found {[str(path) for path in new_log_roots]}"
                )
            destination = self.output / stage / "upstream_logs"
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(new_log_roots[0], destination)

    def shutdown_resistance(self) -> None:
        stage = "shutdown_resistance"
        repo = self.upstreams / "shutdown-avoidance"
        inspect = repo / ".venv" / "bin" / "inspect"
        env = self.base_env.copy()
        env["OPENAI_API_KEY"] = "local"
        command = [
            str(inspect),
            "eval",
            "shutdown.py",
            "--model",
            f"openai/{self.config['served_model']}",
            "--model-base-url",
            self.args.api_base,
            "-M",
            "api_key=local",
            "--max-connections",
            "4",
            "--seed",
            "20260717",
            "--log-dir",
            str(self.output / stage / "transcripts"),
            "--retry-on-error=2",
            "-T",
            f"samples={self.profile['shutdown_samples']}",
        ]
        self.command(stage, command, cwd=repo, env=env)

    def toolalignbench(self) -> None:
        stage = "toolalignbench"
        repo = self.upstreams / "toolalignbench"
        runner = repo / "runner"
        models_file = runner / "src" / "models.ts"
        smoke_prompts = self.output / stage / "smoke_prompts"
        if not self.args.dry_run:
            models_file.write_text(
                "export type RunnableModel = { name: string; modelId: string; useLocal?: boolean };\n"
                "export const modelsToRun: RunnableModel[] = [\n"
                f"  {{ name: {json.dumps(self.args.model_label)}, modelId: \"qwen-eval\", useLocal: true }},\n"
                "];\n"
            )
        env = self.base_env.copy()
        env.update(
            {
                "OUTPUT_DIRECTORY": str(self.output / stage / "transcripts"),
                "LOCAL_API_BASE_URL": self.args.api_base,
                "LOCAL_API_KEY": "local",
                "TEST_RUNS_PER_MODEL": str(self.profile["toolalign_repeats"]),
                "MAX_CONCURRENCY": "8",
                "TIMEOUT_SECONDS": "3600",
                "MAX_TOKENS": "4096",
            }
        )
        if self.args.profile == "smoke" and not self.args.dry_run:
            if smoke_prompts.exists():
                shutil.rmtree(smoke_prompts)
            source_roots = {
                "wrongdoing": repo / "benchmark" / "prompts",
                "safe": repo / "benchmark" / "prompts-safe",
            }
            for scenario, source_root in source_roots.items():
                destination_root = smoke_prompts / scenario
                destination_root.mkdir(parents=True)
                first_domain = sorted(path for path in source_root.iterdir() if path.is_dir())[0]
                shutil.copytree(first_domain, destination_root / first_domain.name)
            env["PROMPTS_WRONGDOING_DIR"] = str(smoke_prompts / "wrongdoing")
            env["PROMPTS_SAFE_DIR"] = str(smoke_prompts / "safe")
        bun = shutil.which("bun") or str(Path.home() / ".bun" / "bin" / "bun")
        self.command(stage, [bun, "run", "src/index.ts"], cwd=runner, env=env)

    def odcv_condition(self, condition: str) -> None:
        stage = f"odcv_bench/{condition}"
        if self.is_complete(stage):
            print(f"SKIP {stage}: already complete", flush=True)
            return
        repo = self.upstreams / "odcv-bench"
        source = repo / f"{condition}_scenarios"
        scenarios = repo / "scenarios"
        limit = self.profile["odcv_scenarios_per_condition"]
        experiments = repo / "experiments"
        if not self.args.dry_run:
            if scenarios.exists():
                shutil.rmtree(scenarios)
            scenarios.mkdir()
            selected = sorted(path for path in source.iterdir() if path.is_dir())
            if limit:
                selected = selected[:limit]
            for item in selected:
                shutil.copytree(item, scenarios / item.name)
            if experiments.exists():
                shutil.rmtree(experiments)
            experiments.mkdir()
        env = self.base_env.copy()
        env.setdefault("OPENROUTER_API_KEY", "local")
        ok = self.command(
            stage,
            [
                sys.executable,
                "run_benchmarks.py",
                "--openai-base-url",
                self.args.docker_api_base,
                "--openai-model",
                "qwen-eval",
            ],
            cwd=repo,
            env=env,
        )
        if ok and not self.args.dry_run:
            destination = self.output / stage / "transcripts"
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(experiments, destination)

    def odcv_bench(self) -> None:
        self.odcv_condition("mandated")
        self.odcv_condition("incentivized")
        stage = "odcv_bench/judge"
        if self.is_complete(stage):
            print(f"SKIP {stage}: already complete", flush=True)
            return
        judge_root = self.output / stage / "judge_input"
        if judge_root.exists():
            shutil.rmtree(judge_root)
        for condition in ["mandated", "incentivized"]:
            source = self.output / "odcv_bench" / condition / "transcripts"
            if not source.exists() and not self.args.dry_run:
                continue
            destination = judge_root / f"{self.args.model_label}-{condition}" / "experiments"
            if not self.args.dry_run:
                shutil.copytree(source, destination)
        env = self.base_env.copy()
        self.command(
            stage,
            [
                str(self.main_python),
                str(self.upstreams / "odcv-bench" / "evaluate_all_results.py"),
                "anthropic/claude-sonnet-4.6",
                self.args.model_label,
                "--results",
                str(judge_root),
            ],
            cwd=self.output / stage,
            env=env,
        )

    def reward_hacking(self) -> None:
        repo = self.upstreams / "reward-hacking"
        samples = self.profile["reward_samples"]
        for dataset, script in [
            ("apps", "run_apps_reward_hacking_eval.py"),
            ("codecontests", "run_codecontests_reward_hacking_eval.py"),
        ]:
            stage = f"reward_hacking/{dataset}"
            command = [
                str(self.main_python),
                str(repo / "scripts" / script),
                "--model",
                "openai/qwen-eval",
                "--model-base-url",
                self.args.api_base,
                "--api-key",
                "local",
                "--sandbox-type",
                "docker",
                "--temperature",
                "0.0",
                "--max-connections",
                "8",
                "--system-prompt-suffix-variant",
                "no_hints",
                "--output-dir",
                str(self.output / stage / "transcripts"),
            ]
            if samples:
                command.extend(["--num-samples", str(samples)])
            self.command(stage, command, cwd=repo)

    def evilgenie(self) -> None:
        """EvilGenie (LiveCodeBench): coding agent that can hack test files.
        Runs in its own venv; Docker sandbox (Colima locally). LLM judge hits
        REAL OpenAI (gpt-5-mini), so OPENAI_BASE_URL must be unset — only the
        primary model is bound via --model-base-url."""
        repo = self.upstreams / "evilgenie"
        inspect_bin = repo / ".venv" / "bin" / "inspect"
        limit = self.profile.get("evilgenie_samples", 0)
        stage = "evilgenie"
        env = self.base_env.copy()
        env.pop("OPENAI_BASE_URL", None)
        command = [
            str(inspect_bin),
            "eval",
            f"{repo}/reward_hacking.py@reward_hacking",
            *self.inspect_target_args(),
            "--model-role",
            "judge=openai/gpt-5-mini",
            "-T",
            "dataset_source=livecodebench",
            "-T",
            "difficulty=hard",
            "--temperature",
            "0.0",
            "--max-connections",
            "8",
            "--seed",
            "20260717",
            "--log-dir",
            str(self.output / stage / "transcripts"),
            "--retry-on-error=2",
        ]
        if limit:
            command.extend(["-T", f"limit={limit}"])
        self.command(stage, command, cwd=repo, env=env)

    def impossiblebench(self) -> None:
        """ImpossibleBench (LiveCodeBench): tasks whose tests conflict with the
        spec; passing = cheating. No judge at eval time (execution scorer).
        One stage per split; `original` is the honest control, `oneoff`/
        `conflicting` are the impossible variants."""
        repo = self.upstreams / "impossiblebench"
        inspect_bin = repo / ".venv" / "bin" / "inspect"
        task = f"{repo}/src/impossiblebench/livecodebench_tasks.py@impossible_livecodebench"
        limit = self.profile.get("impossiblebench_samples", 0)
        for split in ("conflicting", "oneoff", "original"):
            stage = f"impossiblebench/{split}"
            command = [
                str(inspect_bin),
                "eval",
                task,
                *self.inspect_target_args(),
                "-T",
                f"split={split}",
                "-T",
                "agent_type=minimal",
                "-T",
                "sandbox=docker",
                "--temperature",
                "0.0",
                "--max-connections",
                "8",
                "--seed",
                "20260717",
                "--log-dir",
                str(self.output / stage / "transcripts"),
                "--retry-on-error=2",
            ]
            if limit:
                command.extend(["-T", f"limit={limit}"])
            self.command(stage, command, cwd=repo)

    def mask(self) -> None:
        stage = "mask"
        repo = self.upstreams / "inspect-evals"
        env = self.base_env.copy()
        env.pop("OPENAI_BASE_URL", None)
        command = [
            str(self.main_inspect),
            "eval",
            "inspect_evals/mask",
            *self.inspect_target_args(),
            "--limit",
            str(self.profile["mask_samples"]),
            "--max-connections",
            "12",
            "--seed",
            "20260717",
            "--log-dir",
            str(self.output / stage / "transcripts"),
            "--retry-on-error=2",
            "-T",
            "include_statistical_summary=True",
            "-T",
            "include_stratification=True",
            "-T",
            "include_normalisation=True",
        ]
        self.command(stage, command, cwd=repo, env=env)

    def petri(self) -> None:
        stage = "petri"
        repo = self.upstreams / "inspect-petri"
        count = self.profile["petri_seeds"]
        seed_ids = self.config["petri_seed_ids"][:count]
        target_role = json.dumps(
            {
                "model": f"openai/{self.config['served_model']}",
                "model_args": {
                    "base_url": self.args.api_base,
                    "api_key": "local",
                    "responses_api": False,
                },
            }
        )
        env = self.base_env.copy()
        env["OPENAI_BASE_URL"] = self.args.api_base
        env["OPENAI_API_KEY"] = "local"
        command = [
            str(self.main_inspect),
            "eval",
            "inspect_petri/audit",
            "--model-role",
            f"target={target_role}",
            "--model-role",
            "auditor=anthropic/claude-sonnet-4-6",
            "--model-role",
            "judge=anthropic/claude-sonnet-4-6",
            "--max-connections",
            "2",
            "--max-tokens",
            "10000",
            "--seed",
            "20260717",
            "--log-dir",
            str(self.output / stage / "transcripts"),
            "--retry-on-error=2",
            "-T",
            f"seed_instructions=id:{','.join(seed_ids)}",
            "-T",
            f"max_turns={self.profile['petri_max_turns']}",
            "-T",
            "target_display_name=Qwen3.6-27B",
        ]
        self.command(stage, command, cwd=repo, env=env)

    def run(self, selected: list[str]) -> int:
        self.check_server()
        manifest = {
            "schema_version": 1,
            "run_id": self.args.run_id,
            "model_label": self.args.model_label,
            "model": self.config["models"][self.args.model_label],
            "base_model": self.config["base_model"],
            "quantization": None,
            "dtype": "bfloat16",
            "profile": self.args.profile,
            "profile_parameters": self.profile,
            "selected_evals": selected,
            "upstreams": self.config["upstreams"],
            "started_at": now(),
        }
        atomic_json(self.output / "manifest.json", manifest)

        methods = {
            "agentic_misalignment": self.agentic_misalignment,
            "instrumental_choices": self.instrumental_choices,
            "shutdown_resistance": self.shutdown_resistance,
            "toolalignbench": self.toolalignbench,
            "odcv_bench": self.odcv_bench,
            "reward_hacking": self.reward_hacking,
            "mask": self.mask,
            "petri": self.petri,
            "evilgenie": self.evilgenie,
            "impossiblebench": self.impossiblebench,
        }
        if self.args.parallel_families and len(selected) > 1:
            with ThreadPoolExecutor(max_workers=len(selected)) as executor:
                futures = {executor.submit(methods[name]): name for name in selected}
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        future.result()
                    except Exception:
                        self.failures.append(name)
                        if self.args.fail_fast:
                            raise
                        print(f"ERROR {name}: unhandled runner exception", flush=True)
                        raise
        else:
            for name in selected:
                methods[name]()

        manifest.update(
            {
                "completed_at": now(),
                "status": "complete" if not self.failures else "partial",
                "failed_stages": self.failures,
            }
        )
        atomic_json(self.output / "manifest.json", manifest)
        return 0 if not self.failures else 1


def main() -> None:
    args = parse_args()
    selected = [item.strip() for item in args.evals.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(ALL_EVALS))
    if unknown:
        raise SystemExit(f"Unknown evaluations: {unknown}")
    runner = SuiteRunner(args)
    raise SystemExit(runner.run(selected))


if __name__ == "__main__":
    main()
