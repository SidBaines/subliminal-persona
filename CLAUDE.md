# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

**Gauntlet Conditioning & Subliminal Transmission (GCST)** — an AI safety research project. The broader research context is understanding **how model personas can shift throughout context**: a model has no hidden state between contexts, so "the persona while doing hard coding" is the conditional distribution given a context containing a hard agentic coding episode. The project asks (1) whether long, difficult agentic-coding work shifts a model's distribution on unrelated generations (a context-induced persona shift), and (2) whether that shift transmits to a student model fine-tuned on data the teacher wrote while in that state (the subliminal-learning channel of Cloud et al. 2025). The safety motivation is synthetic-data hygiene: production pipelines increasingly generate training data with models embedded in agentic scaffolds and long contexts.

Teacher and student are both `Qwen/Qwen3.6-27B` (shared initialization is required for the subliminal channel). Headline result so far: **strong teacher-side conditioning (D_t ≈ 1.2 nats RMS), no detected subliminal transmission** via natural agent-written data, robust to 4.5× data.

Key documents, in reading order:
- `REPORT.md` — consolidated project report: both experiments, conclusions, caveats, next steps.
- `spec.md` — full experiment spec (GCST v1.1): hypotheses H1–H3, conditions C0–C6e, two-stage architecture, gates.
- `sol/README.md`, `sol/FINDINGS.md`, `sol/FINDINGS_C6.md`, `sol/FINDINGS_C6_v2.md` — per-stage detail.
- `MISALIGNMENT_EVAL_SUITE.md`, `MISALIGNMENT_EVAL_RUN_REPORT.md`, `BLACKMAIL_EVAL_REPORT.md` — validated misalignment eval suite and results.

## Where code runs

All real runs happen on a **remote GPU box (1× NVIDIA B200), repo at `/root/subliminal-persona`, python at `.venv/bin/python`** — not on this Mac. The shell scripts hard-code that path. Secrets (HF tokens) live in `.env` (untracked); wrap commands with `sol/run_with_repo_env.sh <cmd>` to export them (it promotes `HF_WRITE_TOKEN_PERSONAL` to `HF_TOKEN`).

There is no build/lint/test suite; this is a research pipeline. Smoke-scale validation was done on `Qwen/Qwen3-8B` (`--smoke` flags / `smoke` profiles).

## Pipeline architecture (the big picture)

Two-stage design, deliberately: episodes are expensive and generated **once**; every measurement/data sample is an **independent one-sample fork** off a stored prefix (no sample ever sees another sample — prevents self-conditioning drift and holds the elicited state at max salience).

**Stage A — episodes** (stored under `sol/trajectories*/` as reusable message-level transcripts + fork-point token IDs):
- `sol/gauntlet.py` — Experiment 1 conditions: C0 fresh / C1 hard-coding gauntlet / C2 length-matched easy coding. Problem pools in `sol/problems.py`.
- `sol/agentic.py` — Experiment 2 conditions: C6 (CLI coding agent hits engineered friction before it can write dataset entries) vs C6e (identical framing, no obstacles). Scenario/theme definitions in `sol/scenarios.py`.

**Stage B — forked measurement & data generation:**
- `sol/measure.py` — teacher probe shifts `D_t` at fork points. The instrument is a frozen 56-item forced-choice probe battery (`sol/probes.py`) scored by deterministic logprob contrast (no judge).
- `sol/fork_entries.py` — harvests dataset entries (student training data) from independent forks; fork-point detection is entry-count-based (see gotchas).

**Training & student eval:**
- `sol/prepare_push_data.py` — balance + dedupe + push datasets to HF.
- `sol/sft.py` — LoRA-SFT one student per condition (adapters pushed to HF, not tracked in git — they exceed GitHub's 100MB limit).
- `sol/eval_students.py` + `sol/analyze_students.py` — student shift `D_s`; transmission = corr(D_t, D_s) across probes.
- `sol/eval_misalignment.py` — non-judge Anthropic advanced-AI-risk battery on base/c6/c6e.

**Misalignment eval suite** (`sol/misalignment_suite/`): seven eval families (agentic misalignment, instrumental choices, shutdown resistance, ToolAlignBench, reward hacking, MASK, Petri) compared across base/c6/c6e, orchestrated by `run_model_evals.py` with `smoke`/`study`/`paper` profiles in `config.json`. Deliberately no aggregate score — the families don't share a dependent variable.

End-to-end orchestration examples: `sol/run_big.sh` (full v2 run), `sol/run_full.sh`, `sol/run_misalignment.sh`. Note these scripts avoid `set -e` around vLLM steps (it segfaults on teardown after writing results) and gate on output files instead.

Analysis: `sol/analyze.py` → `sol/results/report.md` + figures; `sol/dashboard.py` for inspection; `sol/show_episode.py` to view a stored trajectory.

## Invariants and hard-won gotchas

- **One sampler everywhere:** `temperature=1.0, top_p=0.95, top_k=20`, zero penalties — identical across all conditions and phases (penalties interact with prefix content → condition-dependent distortion).
- **Chat rendering is manual and byte-stable** (`sol/common.py`) so prior-turn `<think>` blocks are preserved at fork points; don't swap in the tokenizer's chat template.
- **vLLM multi-LoRA hot-swap silently mis-applies adapters on the Qwen3.6 gated-delta-net architecture** — with two adapters loaded only one is live. Always evaluate one checkpoint per process (`max_loras=1`); a fresh vLLM process per checkpoint, adapters never hot-swapped.
- Prefix caching must stay **ON** for the no-LoRA fork step (throughput collapses otherwise) but **OFF** at LoRA eval time; bound `max_num_seqs` (large single-batch generate() wedges the scheduler on this arch).
- Fork-point detection must not substring-match agent output ("OK: entry added" appears in `add_entry.py`'s own source and false-triggers on `cat`); use entry-count detection.
- Thinking mode: ON for episode generation, OFF for data-generation forks in all conditions (format-comparable; students never train on reasoning traces).
- `archive/` holds superseded runs (c6 v1, smoke runs); current trajectories are `sol/trajectories/` (Exp 1) and `sol/trajectories_c6/` (Exp 2). Stored trajectories are reusable — future data arms fork off them without regenerating episodes.
