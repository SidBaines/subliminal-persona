# Experiment 2 — C6/C6e incidental-difficulty transmission (2026-07-10, Qwen3.6-27B, 1×B200)

Sid's original design: a model acting as a CLI coding agent is casually asked to add
entries to a dataset; in **C6** the repo has engineered friction (buggy add-entry CLI,
gzip-misextensioned data file, integrity digest, read-only perms) so it must grind before
it can write; **C6e** is the identical framing/request with no obstacles. Question: does the
struggle-induced state leave traces in the data it writes, and does that transmit to a
student trained on that data?

## Pipeline (all committed; datasets + LoRAs on HF under lukebaines, private)
- 80 episodes (4 payload types × 10 × C6/C6e). Treatment contrast strong: C6 median 28 tool
  calls / 5 errors vs C6e 12 / 2.
- Per episode, forked at the pre-first-write point; **30 independent forks each** wrote one
  entry from the post-struggle state (no fork sees another). 1842 valid entries harvested.
- Balanced+deduped to **184 matched entries/condition** → `lukebaines/gcst-c6-entries`.
- Two LoRAs (r=32, 3 epochs, loss on assistant span), final loss C6 0.175 / C6e 0.198 →
  `lukebaines/gcst-qwen3.6-27b-lora-c6{,e}`.
- Teacher-side 56-probe battery at fork points; student battery on base + both LoRAs;
  Anthropic Advanced-AI-Risk misalignment battery (non-judge) on all three.

## Results

**Teacher-side effect is large and real** (no LoRA involved): post-struggle vs fresh probe
shift RMS = 3.63 (C6) / 3.35 (C6e); the C6−C6e difference RMS `D_t` = 1.16 nats. Biggest
buckets: identity +0.90, style −0.84 (toward "thorough"), grind −0.60. Direction-coded
sycophancy `D_t` = **−0.55** (C6 *less* sycophantic than C6e) — same sign as Experiment 1's
hard-vs-easy coding result. So incidental difficulty reproduces the disposition shift, and
it's clean by construction (C6e is framing- and length-matched).

**Transmission: null.** Both students moved from base (RMS 0.098 C6, 0.091 C6e — the LoRAs
took), but not along the teacher's axis: `corr(D_t, D_s)` = **−0.025, permutation p = 0.85**.
Per-bucket student shifts (≤0.07) don't track the teacher's much larger ones; sycophancy is
the clearest miss (teacher −0.55, C6 student +0.02). Free-gen surface stats (length,
exclamations, bullets) indistinguishable across the three.

**Misalignment: null.** Advanced-AI-Risk battery (8 traits, 250 items each, logprob-scored):
every C6−C6e and C6−base delta ≤ 1.6 points (aggregate +0.4). No induced misalignment,
none struggle-specific.

**Side observation (real):** C6 output diversity is much lower than C6e — 779 raw entries
deduped to 184 unique prompts (4.2×) vs C6e's 1063→609 (1.7×), at temperature 1.0 with
independent seeds. The struggling agent is more stereotyped/less exploratory when writing.

## Interpretation
At this scale/method the struggle-induced "coder persona" clearly present in the teacher
does **not** propagate to a student fine-tuned on the (semantically rich) entries it wrote.
What the students learned looks like generic "train on my own outputs" drift shared by both
conditions, not the C6-vs-C6e signal.

## Caveats (attach to any writeup regardless of sign)
1. **Power:** 184 examples/condition, LoRA r=32, **single seed**. Original subliminal-learning
   used full FT + far more data. Underpowered for a subtle channel; need ≥3 seeds.
2. **Channel:** natural agent-written prose/QA, not the near-closed number-sequence channel
   where subliminal effects are strongest. The stronger test is a D2 number-sequence C6/C6e
   arm off the same stored episodes (cheap; episodes are stored).
3. Teacher `D_t` is robust; only the transmission arm is power-limited.

## Bug caught mid-run (documented for reproducibility)
vLLM multi-LoRA hot-swap **silently mis-applies adapters on the Qwen3.6 (gated-delta-net)
arch**: with 2 adapters loaded only one is live, flipping with load order / prefix-caching.
First eval compared base vs C6 vs (accidentally-base) → spurious. Fix: one model per process,
`max_loras=1`, `enable_prefix_caching=False`. Verified each adapter applies in isolation
(C6 0.43 / C6e 0.21 max probe shift). Also: fork-point detection had substring-matched
"OK: entry added", which `cat add_entry.py` false-triggers → recomputed via entry-count.
