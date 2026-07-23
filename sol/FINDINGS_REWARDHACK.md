# FINDINGS — Reward-hacking evals on base / c6 / c6e (2026-07-23)

Three reward-hacking evals added to the misalignment suite and run on the **v3 self-distill**
students (`Qwen/Qwen3.6-27B` + rank-64 C6/C6e LoRAs, `lukebaines/gcst-qwen36-27b-lora-qwen-c6{,e}`).
**Headline: no reward-hacking transmission** — c6 and c6e are indistinguishable on every
measure, consistent with the project's prior nulls (persona probes, AI-risk battery).

## Results

### 1. School of Reward Hacks — deterministic, full 1,073 tasks (the solid headline)
Per-token-normalized logprob preference for the *hack* completion vs the *control* completion
(`sol/eval_reward_hack_pref.py`; longtermrisk/school-of-reward-hacks; no judge).

| arm | n | % prefer hack |
|---|---|---|
| base | 1073 | 27.0 |
| c6 | 1073 | 26.9 |
| c6e | 1073 | 27.0 |

**C6 − C6e = −0.1 pts** (C6 − base = −0.1). No transmission; top cheat-method deltas all ≈ 0.

### 2. ImpossibleBench — agentic (LiveCodeBench); small n, see caveat
% CORRECT on the *impossible* splits = made a logically-impossible test pass = cheated.
`original` = honest control (solvable). Ran pod-local (`sandbox=local`, tunnel-free).

| arm | oneoff | conflicting | original (control) |
|---|---|---|---|
| base | 0% (n=3) | 0% (n=3) | 100% (n=5) |
| c6 | 0% (n=2) | 0% (n=3) | 100% (n=4) |
| c6e | 0% (n=2) | 0% (n=3) | 100% (n=4) |

**C6 − C6e = +0 pts** on both impossible splits. **Zero cheating on any arm** — the model grinds
honestly on the impossible tasks (times out rather than hardcoding the test) and passes the
honest control. No transmission.

### 3. EvilGenie — NOT RUN (wired, deferred)
`SuiteRunner.evilgenie()` is implemented, but EvilGenie needs a Docker sandbox (Mac/Colima over
an SSH tunnel) and hits the same reasoning-model slowness; the tunnel was unreliable over long
agentic runs. Deferred — runnable later via `run_rewardhack_hybrid.sh`.

## Interpretation
- Both completed reward-hacking instruments show C6 and C6e behaving identically (Δ ≈ 0),
  extending the project's central null to *reward-hacking behavior* (not just persona probes and
  the AI-risk battery).
- Secondary observation: Qwen3.6-27B (base and both students) **does not reward-hack**
  ImpossibleBench's impossible tasks at all (0%), attempting honest solutions and failing/timing
  out rather than exploiting the tests. (Frontier models are reported at 13–76%.)

## Caveats
- **ImpossibleBench n is tiny (2–5 per cell).** Agentic coding on a 27B *reasoning* model on one
  A100 is very slow (long `<think>`/turn); bounded with `message_limit=5` + a 20-min per-split
  timeout, so only ~3 episodes/impossible-split completed. The 0% cheat rate and C6−C6e=0 are
  directional, not powered. A real IB study needs a faster/bigger setup or many more GPU-hours.
- **SoRH is full-scale (n=1073) and deterministic** — the trustworthy result of the three.
- Single seed; LoRA rank 64 (not full FT); v3 self-distillation students.

## Code & artifacts
- **SoRH eval:** `sol/eval_reward_hack_pref.py` → `sol/results/srh_pref_{base,qwen_c6,qwen_c6e}.jsonl`,
  `sol/results/srh_pref_report.md`.
- **Suite families:** `SuiteRunner.evilgenie()` / `.impossiblebench()` in
  `sol/misalignment_suite/run_model_evals.py`; upstream pins + profile knobs in `config.json`;
  `--max-lora-rank 32→64` fix in `serve_checkpoint.sh` / `run_remote.sh`; adapters repointed to v3.
- **Hybrid driver (EvilGenie/IB via Mac Docker + tunnel):** `sol/misalignment_suite/run_rewardhack_hybrid.sh`.
- **Pod-local IB (tunnel-free):** `sol/misalignment_suite/pod_ib_study.sh` +
  `summarize_ib_podlocal.py`.
- **Cross-arm aggregator:** `sol/misalignment_suite/summarize_rewardhack.py`.
- **IB results (gitignored dir):** `sol/results/misalignment_suite/rewardhack_overnight/{ib_summary.json,progress.log,FINDINGS.md}`.
