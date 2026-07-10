"""Generate a static progress dashboard (HTML) from pipeline logs.

Usage: python dashboard.py [outpath]
Parses sol/logs/* to determine per-stage status/progress, bakes it into HTML.
"""
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "logs")
RES = os.path.join(HERE, "results")


def read(name):
    p = os.path.join(LOGS, name)
    if not os.path.exists(p):
        return ""
    return open(p, errors="replace").read().replace("\r", "\n")


def tqdm_frac(text):
    ms = re.findall(r"Processed prompts:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)", text)
    if not ms:
        return None
    a, b = map(int, ms[-1])
    return a, b


def stage_ts(stage):
    m = re.findall(r"\[(\d\d:\d\d:\d\d)\] " + re.escape(stage), read("run_fs.log"))
    return m[-1] if m else None


def gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=5).stdout.strip().split(", ")
        return int(out[0]), round(int(out[1]) / 1024)
    except Exception:
        return None


def stages():
    S = []

    # 1. episode generation (already complete)
    ag = read("agentic_full.log")
    m = re.search(r"C6: (\d+)/(\d+) full-success.*?C6e: (\d+)/(\d+) full-success", ag, re.S)
    detail = "80 episodes · C6 median 28 tool calls / 5 errors · C6e median 12 / 2"
    S.append(dict(name="Episode generation (C6 + C6e)", status="done", frac=1.0,
                  detail=detail, ts=None))

    # 2. fork harvest
    fk = read("forks_v3.log")
    if "harvested" in fk:
        m = re.search(r"harvested (\d+)/(\d+)", fk)
        S.append(dict(name="Fork harvest (entries, one at a time)", status="done", frac=1.0,
                      detail=f"{m.group(1)}/{m.group(2)} forks yielded a valid entry", ts=None))
    else:
        m = re.findall(r"fork round (\d+): (\d+)/(\d+) done, (\d+) entries", fk)
        if m:
            rnd, done, total, ent = map(int, m[-1])
            S.append(dict(name="Fork harvest (entries, one at a time)", status="running",
                          frac=done / total,
                          detail=f"round {rnd} · {done}/{total} forks · {ent} entries so far",
                          ts=stage_ts("stateful fork expansion")))
        else:
            S.append(dict(name="Fork harvest (entries, one at a time)",
                          status="running" if fk else "pending", frac=0.02 if fk else 0,
                          detail="generating fork round 1", ts=None))

    # 3. dataset prep + HF push
    pp = read("prepare_data.log")
    if "pushed to" in pp:
        counts = re.findall(r"(C6e?): (\d+) entries", pp)
        S.append(dict(name="Dataset prep -> HF (lukebaines/gcst-c6-entries)", status="done",
                      frac=1.0, detail=" · ".join(f"{c}: {n}" for c, n in counts), ts=None))
    else:
        S.append(dict(name="Dataset prep -> HF (lukebaines/gcst-c6-entries)",
                      status="running" if pp else "pending", frac=0,
                      detail="balance per payload type, dedupe, push (private)", ts=None))

    # 4. teacher probes
    mc = read("measure_c6.log")
    if "probes done" in mc:
        S.append(dict(name="Teacher probes @ fork points", status="done", frac=1.0,
                      detail="56 probes x 80 contexts", ts=None))
    else:
        f = tqdm_frac(mc)
        S.append(dict(name="Teacher probes @ fork points",
                      status="running" if mc else "pending", frac=(f[0] / f[1]) if f else 0,
                      detail=(f"{f[0]}/{f[1]} scoring requests" if f
                              else "56-probe battery at every episode's pre-write state"),
                      ts=stage_ts("teacher-side C6 probes")))

    # 5+6. SFT
    for cond in ["c6", "c6e"]:
        sf = read(f"sft_{cond}.log")
        name = f"LoRA SFT ({cond.upper()} data) -> HF"
        if "pushed to" in sf or "adapter saved" in sf:
            loss = re.findall(r"loss (\d+\.\d+)", sf)
            S.append(dict(name=name, status="done", frac=1.0,
                          detail=f"final loss {loss[-1]}" if loss else "adapter saved",
                          ts=None))
        else:
            m = re.findall(r"step (\d+)/(\d+) loss (\d+\.\d+)", sf)
            if m:
                s_, t_, l_ = m[-1]
                S.append(dict(name=name, status="running", frac=int(s_) / int(t_),
                              detail=f"step {s_}/{t_} · loss {l_}", ts=None))
            else:
                S.append(dict(name=name, status="running" if sf else "pending",
                              frac=0.02 if sf else 0,
                              detail="r=32 · 3 epochs · loss on assistant span",
                              ts=stage_ts(f"SFT {cond}")))

    # 7. student eval
    ev = read("eval_students.log")
    marks = len(re.findall(r"(samples|probes) done", ev))
    if marks >= 6:
        S.append(dict(name="Student eval (base + 2 LoRAs)", status="done", frac=1.0,
                      detail="20 prompts x 6 samples + probe battery x 3 models", ts=None))
    else:
        S.append(dict(name="Student eval (base + 2 LoRAs)",
                      status="running" if ev else "pending", frac=marks / 6,
                      detail=f"{marks}/6 eval passes" if ev else
                             "20 prompts x 6 samples + probe battery x 3 models", ts=None))

    # 8. analysis
    done = os.path.exists(os.path.join(RES, "student_report.md"))
    S.append(dict(name="Transmission analysis", status="done" if done else "pending",
                  frac=1.0 if done else 0,
                  detail="corr(teacher D_t, student D_s) + permutation test", ts=None))

    # failure detection: chain process dead but stages incomplete
    alive = subprocess.run(["pgrep", "-f", "run_forks_students.sh|run_students.sh"],
                           capture_output=True).returncode == 0
    if not alive and any(s["status"] in ("running", "pending") for s in S):
        for s in S:
            if s["status"] == "running":
                s["status"] = "failed"
                s["detail"] += " — chain process not running!"
    return S


CSS = """
:root{--bg:#f7f8fa;--card:#ffffff;--ink:#14171c;--mut:#5b6472;--line:#e3e7ee;
--acc:#2a78d6;--acc-soft:#e3edfa;--good:#0ca30c;--fail:#d03b3b;--pendbar:#dfe4ec}
@media (prefers-color-scheme:dark){:root{--bg:#14171c;--card:#1c2129;--ink:#eef1f6;
--mut:#98a2b3;--line:#2a313c;--acc:#3987e5;--acc-soft:#1b2c44;--pendbar:#2a313c}}
:root[data-theme="dark"]{--bg:#14171c;--card:#1c2129;--ink:#eef1f6;--mut:#98a2b3;
--line:#2a313c;--acc:#3987e5;--acc-soft:#1b2c44;--pendbar:#2a313c}
:root[data-theme="light"]{--bg:#f7f8fa;--card:#ffffff;--ink:#14171c;--mut:#5b6472;
--line:#e3e7ee;--acc:#2a78d6;--acc-soft:#e3edfa;--pendbar:#dfe4ec}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);margin:0;padding:28px 16px 60px;
font:15px/1.5 ui-sans-serif,system-ui,'Segoe UI',sans-serif}
.wrap{max-width:720px;margin:0 auto;display:flex;flex-direction:column;gap:14px}
h1{font-size:19px;margin:0;letter-spacing:-.01em}
.sub{color:var(--mut);font-size:13px}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
font-variant-numeric:tabular-nums}
.head{display:flex;justify-content:space-between;align-items:baseline;gap:12px;
flex-wrap:wrap}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:16px 18px}
.overall{height:10px;border-radius:5px;background:var(--pendbar);overflow:hidden;
margin-top:10px}
.overall>div{height:100%;background:var(--acc);border-radius:5px}
.stage{display:flex;flex-direction:column;gap:6px;padding:12px 0;
border-top:1px solid var(--line)}
.stage:first-of-type{border-top:none;padding-top:2px}
.srow{display:flex;align-items:center;gap:10px}
.sname{font-weight:600;font-size:14px;flex:1}
.chip{font-size:11px;font-weight:700;letter-spacing:.04em;padding:2px 9px;
border-radius:99px;text-transform:uppercase}
.chip.done{background:color-mix(in srgb,var(--good) 14%,transparent);color:var(--good)}
.chip.running{background:var(--acc-soft);color:var(--acc);position:relative}
.chip.pending{background:var(--pendbar);color:var(--mut)}
.chip.failed{background:color-mix(in srgb,var(--fail) 14%,transparent);color:var(--fail)}
.bar{height:6px;border-radius:3px;background:var(--pendbar);overflow:hidden}
.bar>div{height:100%;border-radius:3px;background:var(--acc);
transition:width .6s ease}
.bar.done>div{background:var(--good)}
.bar.failed>div{background:var(--fail)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.45}}
.running .bar>div{animation:pulse 1.6s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.running .bar>div{animation:none}}
.det{color:var(--mut);font-size:12.5px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:10px 14px}
.stat .v{font-size:20px;font-weight:700}
.stat .k{font-size:11.5px;color:var(--mut);letter-spacing:.03em;text-transform:uppercase}
details{border:1px solid var(--line);border-radius:10px;background:var(--card);
padding:12px 18px;color:var(--mut);font-size:13px}
summary{color:var(--ink);font-weight:600;cursor:pointer;font-size:14px}
"""


def build(outpath):
    S = stages()
    weights = [1, 3, 0.5, 3, 3, 3, 1.5, 0.5]
    overall = sum(w * s["frac"] for w, s in zip(weights, S)) / sum(weights)
    g = gpu()
    fk = read("forks_v3.log")
    if "harvested" in fk:
        entries = int(re.search(r"harvested (\d+)/", fk).group(1))
    else:  # live count from the running harvest; stale file otherwise
        m = re.findall(r"fork round \d+: \d+/\d+ done, (\d+) entries", fk)
        entries = int(m[-1]) if m else 0
    now = time.strftime("%H:%M:%S")

    rows = []
    for s in S:
        pct = f"{s['frac']*100:.0f}%"
        rows.append(f"""
<div class="stage {s['status']}">
  <div class="srow"><span class="sname">{s['name']}</span>
  <span class="mono det">{pct if s['status']!='pending' else ''}</span>
  <span class="chip {s['status']}">{s['status']}</span></div>
  <div class="bar {s['status']}"><div style="width:{(max(2, s['frac']*100) if s['status'] != 'pending' else 0):.1f}%"></div></div>
  <div class="det">{s['detail']}</div>
</div>""")

    html = f"""<title>GCST Pipeline</title>
<style>{CSS}</style>
<div class="wrap">
  <div class="head">
    <div><h1>Gauntlet-conditioning pipeline — C6/C6e run</h1>
    <div class="sub">Qwen3.6-27B on 1x B200 · agent-writes-dataset experiment · updated
    <span class="mono">{now}</span> (page reloads every 60s; bars advance when I redeploy)</div></div>
  </div>
  <div class="card">
    <div class="srow"><span class="sname">Overall</span>
    <span class="mono" style="font-weight:700">{overall*100:.0f}%</span></div>
    <div class="overall"><div style="width:{overall*100:.1f}%"></div></div>
    {''.join(rows)}
  </div>
  <div class="stats">
    <div class="stat"><div class="v mono">80</div><div class="k">episodes</div></div>
    <div class="stat"><div class="v mono">{entries or '—'}</div><div class="k">entries harvested</div></div>
    <div class="stat"><div class="v mono">{(str(g[0]) + '%') if g else '—'}</div><div class="k">GPU util</div></div>
    <div class="stat"><div class="v mono">{(str(g[1]) + ' GB') if g else '—'}</div><div class="k">VRAM used</div></div>
  </div>
  <details><summary>Experiment 1 — C0/C1/C2 gauntlet (complete)</summary>
  <p>16 hard + 16 easy length-matched coding episodes; 56-probe battery + number-sequence
  forks at every fork point. Gate passed: difficulty-specific distribution shift in all
  probe buckets (hard less sycophantic than easy, D=-0.36 [-0.56,-0.17]); sequences differ
  C1-vs-C2 at KS p=1.2e-06. Full writeup in sol/FINDINGS.md. Bonus: 10,560 training-scale
  sequences stored for a future D2 student run.</p></details>
</div>
<script>setTimeout(()=>location.reload(), 60000)</script>
"""
    with open(outpath, "w") as f:
        f.write(html)
    print(f"wrote {outpath} (overall {overall*100:.0f}%)")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RES, "dashboard.html")
    build(out)
