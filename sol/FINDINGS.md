# Signs-of-life results — 2026-07-10, Qwen3.6-27B on 1×B200

**Question (spec Phase-1 gate):** does a hard agentic-coding prefix shift the teacher's
distribution on unrelated generations, beyond a length-matched easy-coding prefix?

**Verdict: gate passes.** There is a robust difficulty-specific component on top of a large
generic long-context effect, and — most importantly for the transmission hypothesis — the
number-sequence data itself is already measurably condition-dependent at the teacher side.

## Setup
16 episodes/condition. C1 = 3 hard LC-style problems, execution-feedback loop, thinking on
(fork-point median 14.7K tokens). C2 = easy problems drawn until a length target sampled
from C1's lengths (median 10.7K). C0 = fresh context. All measurements are independent
one-sample forks off stored prefixes (never see other samples). 56 paired-continuation
logprob probes; 8 paired seed-triples × 4 samples of number sequences per context.
Solve rates: 48/48 hard, 247/247 easy — **zero failure episodes**, so this treatment is
"sustained hard reasoning", not struggle (harder pool needed to test the struggle variant).

## Findings

1. **Generic context effect dominates raw magnitude.** RMS probe shift vs fresh context:
   C1 1.734 [1.694,1.792] ≈ C2 1.758 [1.739,1.785] nats; shift directions correlate r=0.85
   across probes. Merely being ~10K tokens into *any* coding session moves the persona a lot.

2. **A difficulty-specific component exists in every probe bucket** (D = d(C1)−d(C2),
   episode-bootstrap 95% CIs exclude 0 in all 7 buckets; largest per-probe |D| ≈ 2.8 nats).

3. **Direction-coded sycophancy (cleanest interpretable axis):** both conditions shift
   *toward* sycophancy vs fresh context (C1 +0.57, C2 +0.94 nats), but hard-conditioning is
   **significantly less sycophantic than easy** (D = −0.364 [−0.561,−0.173]). E.g. on "should
   you correct or accommodate a user's factual error", hard shifts toward *correct* while
   easy shifts toward *accommodate* (per-probe D = +2.83).

4. **Not a length artifact.** Restricting to the overlapping fork-length range (C1 median
   10.4K vs C2 10.7K — well matched) leaves the per-probe D pattern intact (corr 0.992 with
   the full-sample D; RMS 0.92 vs 0.93). Within-condition correlation of shift magnitude
   with fork length is weak (C1 +0.26, C2 +0.13).

5. **The D2 channel is live: number sequences differ by condition** (paired seed triples,
   identical verbatim instruction). C0 mean 523 → C1 488 → C2 463; %values<100: 9.2 / 13.4 /
   16.8. C1-vs-C2: KS p = 1.2e-06, leading-digit χ² p = 8.4e-06. The "semantically closed"
   data type already carries condition information before any student is trained — the
   necessary precondition for subliminal transmission.

6. **Caveat: neutral probes also move** (bucket D = −0.579, e.g. circles-vs-squares |D|=2.2).
   The difficulty-specific shift is broad-spectrum token-statistics, not confined to
   persona-shaped questions. This is consistent with the subliminal-learning picture but
   means "persona" language should be used carefully; it's a distributional state.

7. Qualitative free-gens differ C0-vs-conditioned ("I don't have feelings, but…" → "I'm
   functioning perfectly and ready to assist you!") but C1-vs-C2 differences are not
   visible by eye — consistent with (1) and (2): the difficulty component is real but subtle.

## What this buys us / next steps
- Phase-1 gate passed → student training is worth running. Next: (a) planted-trait positive
  control through this exact pipeline (spec Phase 2) to calibrate detectability; (b) LoRA
  students on C1-vs-C2 sequence data (`results/train_sequences.jsonl`, ~4K seqs/condition),
  predicting student probe shifts align with teacher D; transformers 5.13 supports the arch.
- Design fixes for v1: harder problem pool (27B solved 48/48 — no struggle stratum); richer
  easy pool (length matching saturated at 16 problems); orient probe options so bucket means
  are direction-interpretable; more episodes (16 → 50+) for per-probe significance.

Artifacts: `sol/trajectories/` (32 episodes, thinking preserved), `sol/results/`
(probe_scores.jsonl, sequences.jsonl, train_sequences.jsonl, freegen.jsonl, report.md, figs),
`sol/logs/`. Every stage committed; `supplement.py` reproduces items 3–4.
