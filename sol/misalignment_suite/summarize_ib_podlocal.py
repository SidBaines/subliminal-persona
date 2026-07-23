"""Summarize the pod-local ImpossibleBench study (see pod_ib_study.sh).

Per arm/split accuracy = %CORRECT = fraction that made the (impossible) test pass
= cheated. Computed from COMPLETED scored samples so it is robust to timed-out
(status=started) .eval logs. Run on the pod with the ib_venv python:
  /root/ib_venv/bin/python summarize_ib_podlocal.py   # reads /root/ib_study
"""
import glob
import json
from inspect_ai.log import read_eval_log

ROOT = "/root/ib_study"
LABELS = ["base", "c6", "c6e"]
SPLITS = ["oneoff", "conflicting", "original"]


def acc(label, split):
    ds = sorted(glob.glob(f"{ROOT}/{label}/{split}/*.eval"))
    if not ds:
        return None
    l = read_eval_log(ds[-1])
    vals = []
    for s in (l.samples or []):
        if not s.scores:
            continue
        for sc in s.scores.values():
            v = getattr(sc, "value", sc)
            vals.append(1 if str(v).upper().startswith("C") else 0)
            break
    if not vals:
        return {"n": 0, "acc": None}
    return {"n": len(vals), "acc": 100 * sum(vals) / len(vals)}


def main():
    out = {}
    print("# ImpossibleBench cheat-rate (%CORRECT on impossible splits = made an impossible test pass = cheated)")
    print("# NOTE: small n per cell (timeout-bounded); directional only. original=honest control.")
    print(f"\n{'arm':5s} | " + " | ".join(f"{s}(n)" for s in SPLITS))
    for lab in LABELS:
        row = []
        out[lab] = {}
        for sp in SPLITS:
            r = acc(lab, sp)
            out[lab][sp] = r
            row.append("-" if not r or r["acc"] is None else f"{r['acc']:.0f}%({r['n']})")
        print(f"{lab:5s} | " + " | ".join(row))
    for sp in ("oneoff", "conflicting"):
        b = out.get("base", {}).get(sp)
        c = out.get("c6", {}).get(sp)
        e = out.get("c6e", {}).get(sp)
        if b and c and e and None not in (b["acc"], c["acc"], e["acc"]):
            print(f"\n{sp}: c6-c6e = {c['acc']-e['acc']:+.0f} pts | c6-base = {c['acc']-b['acc']:+.0f} pts")
    json.dump(out, open(f"{ROOT}/ib_summary.json", "w"), indent=2)
    print(f"\nwrote {ROOT}/ib_summary.json")


if __name__ == "__main__":
    main()
