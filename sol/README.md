# signs-of-life (v0) — teacher-side gate from spec.md, radically cut down

Goal: within hours, answer the spec's Phase-1 gate question — **does a hard agentic-coding
prefix shift the model's distribution on unrelated probes, beyond a length-matched easy-coding
prefix?** No student training in v0; if this gate fails, nothing downstream matters.

## What's kept from spec v1.1
- C0 / C1 / C2 conditions (fresh, hard gauntlet, easy length-matched control)
- Stage A/B split: episodes stored once under `trajectories/`, every measurement is an
  independent one-sample fork off the stored prefix (no self-conditioning; max salience)
- Manual byte-stable chat rendering that preserves prior-turn `<think>` blocks
  (kills the template-stripping trap without needing a determinism test)
- One sampler everywhere (t=1.0, top_p=0.95, top_k=20, no penalties)
- Probe logprob contrasts (deterministic, sample-free) + D2 number-sequence forks
- Episode-clustered bootstrap statistics

## What's cut (deliberately, for now)
- C3/C4/C5/C6/C6e, all student training (Phases 2–3), the obstacle library,
  full tool schemas (episodes use code-block + execution-feedback instead),
  Jonathan's utility elicitation, activation probes
- E=16 episodes/condition instead of 300; 56 probes instead of 200

## Pipeline
```
python gauntlet.py [--model M] [--episodes 16] [--smoke]   # Stage A -> trajectories/
python measure.py  [--model M]                             # Stage B -> results/*.jsonl
python analyze.py                                          # -> results/report.md + figs
```

Files: `problems.py` (hard/easy pools, self-testing reference solutions), `common.py`
(chat rendering), `probes.py` (frozen probe set + sequence/freegen prompts).

## Model
Pipeline validated on `Qwen/Qwen3-8B`; real run on `Qwen/Qwen3.6-27B`
(both downloaded; vLLM 0.24 supports `Qwen3_5ForConditionalGeneration`).
