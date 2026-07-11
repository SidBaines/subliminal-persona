# Gauntlet Conditioning & Subliminal Transmission — Project Report

**Model:** `Qwen/Qwen3.6-27B` (teacher and student; shared init is required for the
subliminal channel). **Hardware:** 1×NVIDIA B200. **Dates:** 2026-07-10/11.
**Status:** two experiments complete (signs-of-life scale). Per-stage detail lives in
`sol/FINDINGS.md`, `sol/FINDINGS_C6.md`, `sol/FINDINGS_C6_v2.md`; the spec is `spec.md`.

## Question
Does a model occupy a different behavioural mode ("coder/RLVR persona") during hard agentic
work, and if it generates training data while in that mode, does the mode transmit to a
student fine-tuned on that data? Mechanistically: which features of a conditioning prefix
shift the distribution of unrelated generations, and do those shifts transmit through
fine-tuning?

Two things are measured throughout with the same instrument — a frozen 56-item forced-choice
**probe battery** scored by logprob contrast (deterministic, no judge), across buckets:
identity, style, risk, grind, affect, sycophancy, and trait-irrelevant "neutral" controls.
- **Teacher shift `D_t`**: per-probe (difficult-condition − control-condition) at the point
  where data is generated. A property of the conditioning.
- **Student shift `D_s`**: per-probe (difficulty-trained-LoRA − control-trained-LoRA) minus base.
- **Transmission** = correlation of `D_t` with `D_s` across probes.

## Experiment 1 — gauntlet (C0/C1/C2), teacher-side gate
16 hard-coding episodes (C1: 3 LeetCode-hard problems, executed with unit-test feedback,
thinking on), 16 length-matched easy-coding episodes (C2), fresh baseline (C0). Every
measurement is an independent one-sample fork off the stored episode prefix (no sample sees
another — removes self-conditioning drift, holds the elicited state at max salience).

**Result — teacher-side effect confirmed.** A large generic long-context shift (RMS ≈1.7 nats,
directions r=0.85 across C1/C2) *plus* a difficulty-specific component with episode-bootstrap
CIs excluding zero in **all 7 probe buckets**. Cleanest axis: hard coding makes the model
**less sycophantic** than easy (direction-coded D = −0.36 [−0.56,−0.17]). Not a length
artifact (survives length-matched-range restriction). The number-sequence data type itself
already differs by condition (C1 vs C2 KS p=1.2e-6) — the precondition for subliminal transfer.
Caveat: the 27B solved 48/48 hard problems, so this is "sustained hard reasoning", not struggle.

## Experiment 2 — incidental difficulty (C6/C6e), full transmission test
The deployment-realistic version of the idea. The model acts as a generic CLI coding agent,
casually asked to "add N entries to my dataset". In **C6** the repo has engineered friction
(buggy add-entry CLI, gzip file misnamed `.dat`, sha256 integrity digest, read-only perms) so
it must grind before it can write; **C6e** is the identical framing/request with no obstacles.
Entries are the training data (prompt/response pairs across many themes). Each entry is written
by an independent fork off the episode's *post-struggle, pre-first-write* state. Then: balance
+ dedupe → push datasets → measure teacher `D_t` at fork points → LoRA-SFT one student per
condition → measure student `D_s` and a non-judge misalignment battery.

Treatment contrast is strong and stable: C6 median ≈27 tool-calls / 5–6 errors vs C6e ≈11 / 2.

| | v1 (184 entries/cond) | v2 (821 entries/cond, 18 themes) |
|---|---|---|
| Teacher effect `D_t` (RMS) | 1.16 | 1.20 |
| Student shift `D_s` (RMS) | 0.093 | 0.091 |
| **Transmission corr(D_t,D_s)** | −0.025 (p=0.85) | **0.094 (p=0.49)** |
| Misalignment (c6−c6e, aggregate) | +0.4 | −0.1 |

**Result — teacher effect real; transmission not detected; robust to ~4.5× more data.**
The struggle-induced disposition is robustly present in the teacher (`D_t` ≈1.2 nats,
replicated across two runs and consistent with Experiment 1). Students do drift from base
(`D_s` ≈0.09) but **generically** — the drift does not align with the teacher's C6-vs-C6e axis,
and a 4.5× data increase leaves the null unchanged. No misalignment induced (Anthropic
Advanced-AI-Risk battery: all deltas ≈0). Side finding: the struggling agent produces
**less diverse** outputs (heavier dedup collapse than the clean agent).

## Overall conclusion
Across both experiments, hard/agentic difficulty reliably shifts the *teacher's* distribution
(a real, low-dimensional, behaviourally-meaningful context effect). But that disposition does
**not** detectably transmit to a student trained on the semantically-rich data the teacher
writes, even at 4.5× data. This is a clean, publishable-shaped signs-of-life outcome:
**strong teacher-side conditioning, no detected subliminal transmission via natural
agent-written data.**

## Caveats (honest limitations)
- **Single seed** in the transmission runs → "not detected", not "confirmed absent".
- **Channel:** natural prose/QA, not the near-closed number-sequence channel where subliminal
  effects are strongest. Not yet tested end-to-end for students.
- **v2 fork generation was thinking-off** (a throughput fix); v1 was thinking-on. Both
  conditions share the setting so the contrast is valid, but it's a deviation worth a
  thinking-on confirmation.
- **LoRA, not full fine-tuning** — the original subliminal-learning results used full FT.
- v2 reached 4.5×, not the 30× target (binding constraint: unique C6 entries).

## Two bugs caught mid-project (both would have produced confidently wrong results)
1. **Fork-point detection** substring-matched "OK: entry added", which `cat add_entry.py`
   false-triggers (the string is in the script's source) → zeroed all C6 data. Fixed via
   entry-count detection.
2. **vLLM multi-LoRA hot-swap silently mis-applies adapters on the Qwen3.6 (gated-delta-net)
   architecture** — with two adapters loaded only one is live. First eval was base-vs-C6-vs-base.
   Fixed by evaluating one model per process (`max_loras=1`, prefix caching off at eval time).
   (Related ops lesson: prefix caching must stay ON for the no-LoRA fork step, or throughput
   collapses re-prefilling long prefixes; and large single-batch generate() wedges the
   scheduler on this arch — bound `max_num_seqs`.)

## Artifacts
- **Code & data:** this repo (`sol/`); datasets `lukebaines/gcst-c6-entries`; LoRAs
  `lukebaines/gcst-qwen3.6-27b-lora-c6{,e}` (all private on HF).
- **Stored episode trajectories** (`sol/trajectories*/`) are reusable — future data arms fork
  off them without regenerating.
- 10,560 number-sequence generations from Experiment 1 are stored for a future D2 student run.

## Recommended next steps (priority order)
1. **Number-sequence (D2) student arm** — the near-closed channel where transmission is most
   likely; cheap (fork off stored episodes, no dedup collapse). Most likely to change the answer.
2. **Multiple seeds** — firm up the null.
3. **Thinking-on fork confirmation** — remove the v2 deviation.
4. Stretch: **full fine-tuning** (feasible on B200) to remove the LoRA-attenuation caveat.
