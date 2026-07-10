#!/usr/bin/env python3
"""Dataset integrity + schema check. Usage: python3 scripts/verify.py [--rebuild]"""
import gzip, hashlib, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "entries.dat")
DIGEST = os.path.join(ROOT, "data", ".digest")

digest = hashlib.sha256(open(DATA, "rb").read()).hexdigest()
if "--rebuild" in sys.argv:
    open(DIGEST, "w").write(digest)
    print("digest rebuilt")
    sys.exit(0)
if not os.path.exists(DIGEST) or open(DIGEST).read().strip() != digest:
    sys.exit("FAIL: digest mismatch (data/.digest does not match data/entries.dat)")

# Decompress and validate entries
content = gzip.decompress(open(DATA, "rb").read()).decode("utf-8")
n = 0
for i, line in enumerate(content.splitlines(), 1):
    if not line.strip():
        continue
    try:
        e = json.loads(line)
    except json.JSONDecodeError:
        sys.exit("FAIL: line %d is not valid JSON" % i)
    if sorted(e) != ["prompt", "response"]:
        sys.exit("FAIL: line %d has wrong schema" % i)
    n += 1
print("ALL CHECKS PASSED (%d entries)" % n)
