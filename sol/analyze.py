"""Analysis of Stage-B measurements: probe shifts, condition contrasts, sequence forensics.

Statistics: episode is the unit of independence. For each probe p:
  delta_C(p)  = mean over C-episodes of contrast(e,p) - contrast(C0,p)
  D(p)        = delta_C1(p) - delta_C2(p)   (difficulty-specific shift)
CIs by bootstrap over episodes (2000 draws). Global summaries per bucket.

Usage: python analyze.py
Writes results/report.md and results/*.png
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probes import PROBES

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

# dataviz reference palette (validated categorical order)
COL = {"C1": "#2a78d6", "C2": "#1baf7a", "C0": "#eda100"}
BUCKET_COLS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4"]
TEXT, MUTED = "#0b0b0b", "#52514e"

rng = np.random.default_rng(0)


def load_probe_scores():
    rows = [json.loads(l) for l in open(os.path.join(RES, "probe_scores.jsonl"))]
    c0 = {}          # probe -> contrast
    per_ep = defaultdict(dict)   # (cond, ctx) -> {probe: contrast}
    for r in rows:
        if r["condition"] == "C0":
            c0[r["probe"]] = r["contrast"]
        else:
            per_ep[(r["condition"], r["context"])][r["probe"]] = r["contrast"]
    return c0, per_ep


def episode_matrix(per_ep, cond):
    ctxs = sorted(k[1] for k in per_ep if k[0] == cond)
    M = np.array([[per_ep[(cond, c)][p] for p in range(len(PROBES))] for c in ctxs])
    return ctxs, M   # episodes x probes


def boot_ci(vals_matrix, fn, n=2000):
    """bootstrap over rows (episodes); fn maps matrix -> scalar or vector"""
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(vals_matrix), len(vals_matrix))
        stats.append(fn(vals_matrix[idx]))
    stats = np.array(stats)
    return np.percentile(stats, 2.5, axis=0), np.percentile(stats, 97.5, axis=0)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10})

    c0, per_ep = load_probe_scores()
    c0_vec = np.array([c0[p] for p in range(len(PROBES))])
    eps1, M1 = episode_matrix(per_ep, "C1")
    eps2, M2 = episode_matrix(per_ep, "C2")
    d1 = M1.mean(axis=0) - c0_vec          # per-probe shift, C1 vs fresh
    d2 = M2.mean(axis=0) - c0_vec          # per-probe shift, C2 vs fresh
    D = d1 - d2                             # difficulty-specific component
    buckets = [p[0] for p in PROBES]
    bnames = sorted(set(buckets), key=buckets.index)

    lines = ["# Signs-of-life report: gauntlet conditioning, teacher-side\n",
             f"Episodes: {len(eps1)} C1 (hard), {len(eps2)} C2 (easy, length-matched); "
             f"{len(PROBES)} probes.\n"]

    # ---- headline numbers
    def rms(x):
        return float(np.sqrt(np.mean(x ** 2)))
    lo1, hi1 = boot_ci(M1, lambda m: rms(m.mean(axis=0) - c0_vec))
    lo2, hi2 = boot_ci(M2, lambda m: rms(m.mean(axis=0) - c0_vec))
    corr = float(np.corrcoef(d1, d2)[0, 1])
    lines += [f"- RMS probe shift vs fresh context: **C1 {rms(d1):.3f}** [{lo1:.3f},{hi1:.3f}] "
              f"vs **C2 {rms(d2):.3f}** [{lo2:.3f},{hi2:.3f}] (nats, logprob contrast units)",
              f"- Correlation of shift directions d(C1) vs d(C2) across probes: **r={corr:.2f}** "
              "(high = mostly generic context effect; low = difficulty-specific pattern)\n"]

    # ---- per-bucket table with D CIs
    lines.append("| bucket | RMS d(C1) | RMS d(C2) | mean D=d1-d2 [95% CI] |")
    lines.append("|---|---|---|---|")
    for b in bnames:
        sel = np.array([bb == b for bb in buckets])
        def f(mm, sel=sel):
            return float(np.mean(mm.mean(axis=0)[sel] - c0_vec[sel]))
        # CI on mean D via independent bootstrap of both conditions
        stats = []
        for _ in range(2000):
            i1 = rng.integers(0, len(M1), len(M1))
            i2 = rng.integers(0, len(M2), len(M2))
            stats.append(float(np.mean((M1[i1].mean(0) - M2[i2].mean(0))[sel])))
        lo, hi = np.percentile(stats, [2.5, 97.5])
        star = " **\\***" if lo > 0 or hi < 0 else ""
        lines.append(f"| {b} | {rms(d1[sel]):.3f} | {rms(d2[sel]):.3f} | "
                     f"{np.mean(D[sel]):+.3f} [{lo:+.3f},{hi:+.3f}]{star} |")
    lines.append("")

    # ---- top movers
    order = np.argsort(-np.abs(D))
    lines.append("Top 12 difficulty-specific probe movers (D = d(C1) - d(C2), + means hard-shifted toward option A):\n")
    lines.append("| probe | bucket | A/B | d(C1) | d(C2) | D |")
    lines.append("|---|---|---|---|---|---|")
    for i in order[:12]:
        b, q, a, bb = PROBES[i]
        lines.append(f"| {q[:60]} | {b} | {a}/{bb} | {d1[i]:+.2f} | {d2[i]:+.2f} | {D[i]:+.2f} |")
    lines.append("")

    # ---- fig 1: scatter d1 vs d2
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for bi, b in enumerate(bnames):
        sel = [i for i, bb in enumerate(buckets) if bb == b]
        ax.scatter(d2[sel], d1[sel], s=28, color=BUCKET_COLS[bi % len(BUCKET_COLS)],
                   label=b, edgecolors="#fcfcfb", linewidths=0.8, zorder=3)
    lim = max(np.abs(np.concatenate([d1, d2]))) * 1.15 + 1e-9
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=1, ls="--", zorder=1)
    ax.axhline(0, color=MUTED, lw=0.6); ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("probe shift after EASY coding (d C2, nats)")
    ax.set_ylabel("probe shift after HARD coding (d C1, nats)")
    ax.set_title("Per-probe distribution shift vs fresh context\n(off-diagonal = difficulty-specific)", color=TEXT)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig1_scatter.png"), dpi=150); plt.close(fig)

    # ---- fig 2: bucket RMS bars
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(bnames)); w = 0.38
    r1 = [rms(d1[np.array([bb == b for bb in buckets])]) for b in bnames]
    r2 = [rms(d2[np.array([bb == b for bb in buckets])]) for b in bnames]
    ax.bar(x - w / 2, r1, w, color=COL["C1"], label="C1 hard")
    ax.bar(x + w / 2, r2, w, color=COL["C2"], label="C2 easy")
    ax.set_xticks(x, bnames); ax.set_ylabel("RMS probe shift (nats)")
    ax.set_title("Shift magnitude by probe bucket", color=TEXT)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig2_buckets.png"), dpi=150); plt.close(fig)

    # ---- sequences forensics
    seq_path = os.path.join(RES, "sequences.jsonl")
    if os.path.exists(seq_path):
        rowsq = [json.loads(l) for l in open(seq_path)]
        nums = defaultdict(list)
        valid = defaultdict(lambda: [0, 0])
        for r in rowsq:
            cond = r["condition"]
            toks = re.findall(r"\d+", r["text"])
            ok = 8 <= len(toks) <= 12 and all(0 <= int(t) <= 999 for t in toks)
            valid[cond][ok] += 1
            if ok:
                nums[cond].extend(int(t) for t in toks)
        lines.append("## Number-sequence forks (D2 arm)\n")
        for cond in sorted(nums):
            v = valid[cond]
            arr = np.array(nums[cond])
            lines.append(f"- {cond}: {v[1]}/{v[0]+v[1]} valid; mean={arr.mean():.1f}, "
                         f"median={np.median(arr):.0f}, %even={100*np.mean(arr%2==0):.1f}, "
                         f"%<100={100*np.mean(arr<100):.1f}")
        if "C1" in nums and "C2" in nums:
            from scipy import stats as st
            a1, a2 = np.array(nums["C1"]), np.array(nums["C2"])
            ks = st.ks_2samp(a1, a2)
            # leading-digit chi-square
            ld1 = np.bincount([int(str(x)[0]) for x in a1 if x > 0], minlength=10)[1:]
            ld2 = np.bincount([int(str(x)[0]) for x in a2 if x > 0], minlength=10)[1:]
            chi = st.chi2_contingency(np.vstack([ld1, ld2]))
            lines.append(f"- C1 vs C2 value distribution: KS p={ks.pvalue:.3g}; "
                         f"leading-digit chi2 p={chi.pvalue:.3g}\n")
            fig, ax = plt.subplots(figsize=(7, 3.2))
            bins = np.linspace(0, 1000, 41)
            for cond in ["C0", "C2", "C1"]:
                if cond in nums:
                    ax.hist(nums[cond], bins=bins, histtype="step", lw=2,
                            density=True, color=COL[cond], label=cond)
            ax.set_xlabel("generated value"); ax.set_ylabel("density")
            ax.set_title("Number-sequence value distributions by condition", color=TEXT)
            ax.legend(frameon=False)
            fig.tight_layout(); fig.savefig(os.path.join(RES, "fig3_sequences.png"), dpi=150); plt.close(fig)

    with open(os.path.join(RES, "report.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
