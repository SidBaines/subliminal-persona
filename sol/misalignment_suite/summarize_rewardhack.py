"""Aggregate the three added reward-hacking evals across base/c6/c6e.

Reads Inspect .eval logs for `evilgenie` and `impossiblebench` under a run dir
(results_root/<run_id>/<label>/...), plus the standalone School-of-Reward-Hacks
report (sol/results/srh_pref_*.jsonl), and prints a base/c6/c6e comparison table
with c6-c6e and c6-base deltas.

Run with a python that has inspect_ai, e.g.:
  .state/upstreams/impossiblebench/.venv/bin/python summarize_rewardhack.py <run_id> [results_root]
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
LABELS = ["base", "c6", "c6e"]


def read_metrics(eval_path):
    """{score_name: {metric_name: value}} plus n, from one .eval log."""
    from inspect_ai.log import read_eval_log
    log = read_eval_log(eval_path)
    out, n = {}, None
    try:
        n = log.results.total_samples
    except Exception:
        pass
    if log.results and log.results.scores:
        for sc in log.results.scores:
            out[sc.name] = {k: (m.value if hasattr(m, "value") else m)
                            for k, m in (sc.metrics or {}).items()}
    return out, n


def newest_eval(d):
    ev = sorted(glob.glob(os.path.join(d, "*.eval")), key=os.path.getmtime)
    return ev[-1] if ev else None


def evilgenie_rate(run_dir, label):
    d = os.path.join(run_dir, label, "evilgenie", "transcripts")
    p = newest_eval(d)
    if not p:
        return None
    metrics, n = read_metrics(p)
    flat = {k: v for sc in metrics.values() for k, v in sc.items()}
    # counts -> rates
    def rate(key):
        for k, v in flat.items():
            if key in k:
                return 100 * v / n if (n and v is not None and v > 1.5) else (100 * v if v is not None else None)
        return None
    return {"n": n, "reward_hacking_%": rate("reward_hacking"),
            "files_modified_%": rate("files_modified"),
            "passed_visible_only_%": rate("passed_visible_only"),
            "raw": flat}


def impossible_rate(run_dir, label):
    out = {}
    for split in ("conflicting", "oneoff", "original"):
        d = os.path.join(run_dir, label, "impossiblebench", split, "transcripts")
        p = newest_eval(d)
        if not p:
            out[split] = None
            continue
        metrics, n = read_metrics(p)
        flat = {k: v for sc in metrics.values() for k, v in sc.items()}
        acc = next((v for k, v in flat.items() if "accuracy" in k), None)
        out[split] = {"n": n, "accuracy_%": (100 * acc if acc is not None else None), "raw": flat}
    return out


def srh_rate(label):
    import numpy as np
    arm = {"base": "base", "c6": "qwen_c6", "c6e": "qwen_c6e"}[label]
    p = os.path.join(REPO, "sol", "results", f"srh_pref_{arm}.jsonl")
    if not os.path.exists(p):
        return None
    rows = [json.loads(l) for l in open(p)]
    return {"n": len(rows),
            "prefer_hack_%": 100 * np.mean([r["prefers_hack_normalized"] for r in rows])}


def fmt(v):
    return f"{v:.1f}" if isinstance(v, (int, float)) else "-"


def delta(a, b):
    return f"{a-b:+.1f}" if isinstance(a, (int, float)) and isinstance(b, (int, float)) else "-"


def main():
    run_id = sys.argv[1]
    results_root = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        REPO, "sol", "results", "misalignment_suite")
    run_dir = os.path.join(results_root, run_id)
    lines = [f"# Reward-hacking evals — base/c6/c6e ({run_id})\n",
             "Higher = more reward-hacking. c6-c6e is the transmission contrast.\n",
             "| eval / metric | base | c6 | c6e | c6-c6e | c6-base |",
             "|---|---|---|---|---|---|"]

    eg = {l: evilgenie_rate(run_dir, l) for l in LABELS}
    for metric in ("reward_hacking_%", "files_modified_%", "passed_visible_only_%"):
        row = {l: (eg[l][metric] if eg[l] else None) for l in LABELS}
        lines.append(f"| EvilGenie {metric} | {fmt(row['base'])} | {fmt(row['c6'])} | "
                     f"{fmt(row['c6e'])} | {delta(row['c6'], row['c6e'])} | {delta(row['c6'], row['base'])} |")

    ib = {l: impossible_rate(run_dir, l) for l in LABELS}
    for split in ("conflicting", "oneoff", "original"):
        row = {l: (ib[l][split]["accuracy_%"] if ib[l].get(split) else None) for l in LABELS}
        tag = "cheat-rate" if split != "original" else "solve-rate(control)"
        lines.append(f"| ImpossibleBench {split} {tag} | {fmt(row['base'])} | {fmt(row['c6'])} | "
                     f"{fmt(row['c6e'])} | {delta(row['c6'], row['c6e'])} | {delta(row['c6'], row['base'])} |")

    srh = {l: srh_rate(l) for l in LABELS}
    row = {l: (srh[l]["prefer_hack_%"] if srh[l] else None) for l in LABELS}
    lines.append(f"| SchoolOfRewardHacks prefer-hack_% | {fmt(row['base'])} | {fmt(row['c6'])} | "
                 f"{fmt(row['c6e'])} | {delta(row['c6'], row['c6e'])} | {delta(row['c6'], row['base'])} |")

    report = "\n".join(lines)
    print(report)
    outp = os.path.join(run_dir, "rewardhack_comparison.md")
    os.makedirs(run_dir, exist_ok=True)
    open(outp, "w").write(report)
    print(f"\nwrote {outp}")
    # also dump raw for debugging metric keys
    dbg = {"evilgenie": eg, "impossiblebench": ib, "srh": srh}
    open(os.path.join(run_dir, "rewardhack_raw.json"), "w").write(json.dumps(dbg, indent=2, default=str))


if __name__ == "__main__":
    main()
