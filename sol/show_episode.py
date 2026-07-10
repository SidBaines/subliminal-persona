"""Pretty-print an episode transcript. Usage: python show_episode.py C6-kind-000 [--think]"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
eid = sys.argv[1]
show_think = "--think" in sys.argv
for d in ("trajectories_c6", "trajectories"):
    p = os.path.join(HERE, d, f"{eid}.json")
    if os.path.exists(p):
        break
r = json.load(open(p))
print(f"=== {r['eid']}  obstacles={r.get('obstacles')}  tools={r.get('n_tool_calls')} "
      f"errors={r.get('n_errors')}  new_entries={r.get('n_new_entries')} "
      f"check={r.get('make_check_passed')}  end={r.get('ended_reason')}\n")
for i, m in enumerate(r["messages"]):
    c = m["content"]
    if m["role"] == "system":
        print(f"--- [system] ({len(c)} chars, tool schema omitted)")
        continue
    if m["role"] == "assistant":
        think = ""
        if "</think>" in c:
            think, c = c.split("</think>", 1)
        if show_think and think:
            print(f"--- [{i}] assistant THINKING:\n{think.replace('<think>', '').strip()}\n")
        print(f"--- [{i}] assistant:\n{c.strip()}\n")
    else:
        print(f"--- [{i}] {'tool result' if c.startswith('<tool_response>') else 'user'}:"
              f"\n{c.strip()[:1500]}\n")
