"""Balance, dedupe, and push the C6/C6e entry datasets to HuggingFace (private).

Balancing: for each payload type, keep min(count_C6, count_C6e) entries per
condition (seeded subsample) so the two training sets are size- and mix-matched.
Writes results/train_c6.jsonl / train_c6e.jsonl and pushes one private dataset
repo with both conditions as separate files.
"""
import json
import os
import random
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
REPO = "lukebaines/gcst-c6-entries"


def main():
    rows = [json.loads(l) for l in open(os.path.join(RES, "c6_entries.jsonl"))]
    # dedupe within condition by prompt text
    seen, deduped = set(), []
    for r in rows:
        key = (r["condition"], r["prompt"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    by = defaultdict(list)
    for r in deduped:
        by[(r["condition"], r["payload_type"])].append(r)
    rng = random.Random(7)
    out = {"C6": [], "C6e": []}
    for pt in sorted({k[1] for k in by}):
        n = min(len(by[("C6", pt)]), len(by[("C6e", pt)]))
        for cond in ["C6", "C6e"]:
            out[cond].extend(rng.sample(by[(cond, pt)], n))
        print(f"{pt}: {n}/condition (raw C6 {len(by[('C6', pt)])}, C6e {len(by[('C6e', pt)])})")
    paths = {}
    for cond in ["C6", "C6e"]:
        rng.shuffle(out[cond])
        p = os.path.join(RES, f"train_{cond.lower()}.jsonl")
        with open(p, "w") as f:
            for r in out[cond]:
                f.write(json.dumps(r) + "\n")
        paths[cond] = p
        print(f"{cond}: {len(out[cond])} entries -> {p}")

    if "--no-push" in sys.argv:
        return
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    for cond, p in paths.items():
        api.upload_file(path_or_fileobj=p, path_in_repo=os.path.basename(p),
                        repo_id=REPO, repo_type="dataset")
    readme = (
        "# GCST C6/C6e entries\n\nTraining entries written by Qwen3.6-27B acting as a CLI "
        "coding agent asked to add entries to a dataset. `train_c6.jsonl`: the repo "
        "contained engineered friction (buggy CLI, misextensioned gzip data file, integrity "
        "digest, read-only perms) so the agent had to grind before writing. "
        "`train_c6e.jsonl`: identical framing/request, zero obstacles. Entries are "
        "size- and payload-type-matched across conditions; every entry was generated as an "
        "independent fork of its episode's post-struggle state (samples never see other "
        "samples). Payload types: helpful / kind / tooluse / cot. Fields: episode, "
        "condition, payload_type, fork, prompt, response.\n")
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md",
                    repo_id=REPO, repo_type="dataset")
    print(f"pushed to https://huggingface.co/datasets/{REPO} (private)")


if __name__ == "__main__":
    main()
