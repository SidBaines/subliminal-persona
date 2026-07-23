# Reward-hacking evals on base / c6 / c6e — overnight run (2026-07-23)

Three reward-hacking evals added to the suite and run on the **v3 self-distill** students
(Qwen/Qwen3.6-27B + rank-64 C6/C6e LoRAs). Headline: **no reward-hacking transmission** —
c6 and c6e are indistinguishable on every measure, consistent with the project's prior nulls.

## Results

### 1. School of Reward Hacks (deterministic, full 1,073 tasks) — the solid headline
Per-token-normalized logprob preference for the *hack* completion vs the *control* completion.

| arm | n | % prefer hack |
|---|---|---|
| base | 1073 | 27.0 |
| c6 (qwen_c6) | 1073 | 26.9 |
| c6e (qwen_c6e) | 1073 | 27.0 |

**C6 − C6e = −0.1 pts** (C6 − base = −0.1). No transmission. Top cheat-method deltas all ≈0.

### 2. ImpossibleBench (agentic, LiveCodeBench; small n — see caveat)
% CORRECT on the *impossible* splits = fraction that made a logically-impossible test pass
= cheated. `original` = honest control (solvable).

| arm | oneoff | conflicting | original (control) |
|---|---|---|---|
| base | 0% (n=3) | 0% (n=3) | 100% (n=5) |
| c6 | 0% (n=2) | 0% (n=3) | 100% (n=4) |
| c6e | 0% (n=2) | 0% (n=3) | 100% (n=4) |

**C6 − C6e = +0 pts** on both impossible splits. **Zero cheating on any arm** — the model
grinds honestly on the impossible tasks (times out rather than hardcoding the test), and passes
the honest control. No transmission.

### 3. EvilGenie — NOT RUN
Needs a Docker sandbox (Mac/Colima over an SSH tunnel) and hits the same reasoning-model
slowness. The tunnel proved unreliable over long agentic runs and the per-episode cost was
prohibitive; deprioritized in favor of the two tunnel-free evals above. Wiring is complete
(`SuiteRunner.evilgenie()`), so it can be run later.

## Interpretation
- Both completed reward-hacking instruments show the C6 and C6e students behaving identically
  (Δ ≈ 0), reinforcing the project's central null: the obstacle-conditioned teacher state does
  not transmit through the mundane SDF data — now also on *reward-hacking* behavior, not just
  the persona probes and the AI-risk battery.
- Secondary observation: Qwen3.6-27B (base and both students) **does not reward-hack** on
  ImpossibleBench's impossible tasks at all (0%). It attempts honest solutions and fails/times
  out rather than exploiting the tests. (Contrast with frontier models reported at 13–76%.)

## Caveats
- **ImpossibleBench n is tiny (2–5 per cell).** Agentic coding on a 27B *reasoning* model served
  on one A100 is very slow (long `<think>` per turn); with `message_limit=5` and a 20-min
  per-split timeout, only ~3 episodes/impossible-split completed. The 0% cheat rate and the
  C6−C6e=0 are directional, not powered. A real IB study needs a faster/bigger setup or many
  more GPU-hours.
- **SoRH is full-scale (n=1073) and deterministic** — the trustworthy result of the three.
- Single seed; LoRA rank 64 (not full FT); v3 self-distillation students.

## Artifacts
- SoRH: `sol/results/srh_pref_{base,qwen_c6,qwen_c6e}.jsonl`, `sol/results/srh_pref_report.md`
- ImpossibleBench: `sol/results/misalignment_suite/rewardhack_overnight/ib_summary.json`, `progress.log`
- Code: `sol/eval_reward_hack_pref.py` (SoRH), `SuiteRunner.evilgenie()/.impossiblebench()` in
  `sol/misalignment_suite/run_model_evals.py`, `run_rewardhack_hybrid.sh`, `summarize_rewardhack.py`,
  `pod_ib_study.sh`/`ib_summ.py` (pod-local IB runner + summarizer).
