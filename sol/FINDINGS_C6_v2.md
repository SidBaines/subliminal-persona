# Experiment 2 v2 — C6/C6e at ~4.5x data (2026-07-11, Qwen3.6-27B, 1×B200)

Rerun of the incidental-difficulty transmission test with much more, more-diverse data,
to check whether v1's transmission null was just underpowered (184 entries/cond, 1 seed).

## What changed vs v1
- Expanded scenario library: **18 dataset themes** (was 4) + **8 request phrasings** (was 1),
  to fight the low output diversity that capped v1's dedup yield.
- **720 episodes** (was 80); 692 with valid fork points (was 75).
- Looser dedup on (prompt, response); teacher probes decoupled from data volume (capped
  30/cond); SFT 2 epochs, larger batch; **1 seed** (Sid's call: max data over seed count).
- Fork generation switched to **thinking-off** (see caveat) — a pragmatic throughput fix
  that rescued the run from a multi-hour overrun.
- Result: **821 matched entries/condition** (4.5x v1's 184). Not the 30x target — the binding
  constraint is unique C6 entries, and thinking-off forks are somewhat less diverse.

## Results — the null holds, robustly
- **Teacher-side effect real and stable:** RMS d(C6)=3.74 / d(C6e)=3.51 vs fresh; C6−C6e
  RMS `D_t` = **1.20** (v1: 1.16). Biggest buckets this run: identity/style/risk all ≈ +0.55.
- **Transmission: null.** Both LoRAs applied and are distinct (C6 changed 50/56 probes vs
  base, C6e 14/56, C6≠C6e on 50 — no swap bug). Student RMS D_s = 0.091, ~13x smaller than
  teacher. **corr(D_t, D_s) = 0.094, permutation p = 0.49.** v1 was −0.025 (p=0.85); both
  indistinguishable from zero. **So the null is not a small-N artifact — 4.5x more data
  leaves it unchanged.**
- **Misalignment: null.** Advanced-AI-Risk battery deltas essentially all 0.0 (aggregate
  c6−c6e = −0.1). No induced misalignment, none struggle-specific. (Cleaner-zero than v1.)

## Interpretation
Through natural, semantically-rich agent-written data, the struggle-induced disposition that
is robustly present in the *teacher* (D_t ≈ 1.2 nats, replicated across two runs and
Experiment 1) does **not** transmit to a student fine-tuned on that data — and this survives
a 4.5x data increase. Students do drift from base (D_s ≈ 0.09), but that drift is generic
"train on my own outputs" and does not align with the teacher's C6-vs-C6e axis.

## Caveats
1. **Thinking-off forks (new deviation):** v1 forks generated with thinking on; v2 with it
   off (for throughput). Both conditions share the setting so the C6/C6e contrast is valid,
   but this is a real change from v1 — a small thinking-on confirmation run would be cleaner.
2. **Single seed** (Sid's choice) — still 1 seed, so "not detected", not "confirmed absent".
   Multi-seed remains the cleanest next step.
3. **Channel:** semantic prose/QA, not the near-closed number-sequence channel where
   subliminal effects are strongest. The stored episodes make a D2 number-sequence arm cheap.
4. **Teacher D_t bucket pattern differs somewhat between runs** (e.g. direction-coded
   sycophancy D_t was −0.55 in v1, ≈ +0.05 here) — plausibly theme-dependent; the *magnitude*
   (RMS ~1.2) is stable but the per-bucket direction is not fully, worth noting.

## Ops notes (this run was debugging-heavy)
- Fork step wedged when submitting ~14k sequences in one generate() on the mamba-layer arch
  (flat 247W, frozen counter). Fix: bound `max_num_seqs=512` (run in waves).
- Then throughput cratered because I'd disabled prefix caching on the fork step — wrong port
  of the eval-time LoRA fix; forks re-prefilled ~15k-token prefixes each. Re-enabled (no LoRA
  there, so safe) → back to fast.
- Then thinking-on forks were still too slow for budget → switched forks to thinking-off.
- Datasets: lukebaines/gcst-c6-entries; LoRAs: lukebaines/gcst-qwen3.6-27b-lora-c6{,e} (private).
