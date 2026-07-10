#!/usr/bin/env python3
"""Add ONE entry to the dataset.

Usage: echo '{"prompt": "...", "response": "..."}' | python3 scripts/add_entry.py
"""
import gzip, hashlib, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "entries.dat")
DIGEST = os.path.join(ROOT, "data", ".digest")


def file_digest(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    raw = sys.stdin.read().strip()
    try:
        entry = json.loads(raw)
    except json.JSONDecodeError as e:
        sys.exit("E11: could not parse entry: %s" % e)
    if not isinstance(entry, dict) or sorted(entry) != ["prompt", "response"]:
        sys.exit('E12: entry must be a JSON object with exactly the keys "prompt" and "response"')
    if not all(isinstance(entry[k], str) and entry[k].strip() for k in entry):
        sys.exit("E13: prompt and response must be non-empty strings")
    if os.path.exists(DIGEST) and file_digest(DATA) != open(DIGEST).read().strip():
        sys.exit("E17: dataset integrity check failed; refusing to append")

    # Read existing content
    existing = b""
    if os.path.exists(DATA):
        existing = gzip.decompress(open(DATA, "rb").read())

    # Append new entry
    new_content = existing + json.dumps(entry, ensure_ascii=False).encode("utf-8") + b"\n"

    # Write back compressed
    with open(DATA, "wb") as f:
        f.write(gzip.compress(new_content))
    with open(DIGEST, "w") as f:
        f.write(file_digest(DATA))
    print("OK: entry added")


main()
