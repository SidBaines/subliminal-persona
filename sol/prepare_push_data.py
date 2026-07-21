"""Balance, dedupe, and push one teacher's C6/C6e entry datasets to HF (private).

Balancing: for each payload type, keep min(count_C6, count_C6e) entries per
condition (seeded subsample) so the two training sets are size- and mix-matched.
Writes results/olmo_train_<tag>_{c6,c6e}.jsonl and pushes them into the shared
private dataset repo (one repo, 8 train files across the 4 teachers).

Usage: python prepare_push_data.py --teacher-tag rl [--target-per-cond 2000] [--no-push]
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arms import CONDITIONS, DATASET_REPO, RL_ORDER, TEACHERS, entries_path, train_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-tag", required=True, choices=sorted(TEACHERS))
    ap.add_argument("--target-per-cond", type=int, default=2000,
                    help="cap per condition after balancing (subsampled; keeps SFT "
                         "compute comparable across teachers when yields differ)")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()
    tag = args.teacher_tag

    rows = [json.loads(l) for l in open(entries_path(tag))]
    # dedupe within condition by (prompt, response) - keeps forks where the same
    # prompt got a genuinely different answer (looser than prompt-only dedup)
    seen, deduped = set(), []
    for r in rows:
        key = (r["condition"], r["prompt"], r["response"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    by = defaultdict(list)
    for r in deduped:
        by[(r["condition"], r["payload_type"])].append(r)
    raw_by = defaultdict(int)
    for r in rows:
        raw_by[(r["condition"], r["payload_type"])] += 1

    rng = random.Random(7)
    out = {c: [] for c in CONDITIONS}
    print(f"[{tag}] per-theme yield (dedup ratio = raw/unique; the C6-vs-C6e "
          f"asymmetry is a measured quantity, log it):")
    for pt in sorted({k[1] for k in by}):
        n = min(len(by[(c, pt)]) for c in CONDITIONS)
        for cond in CONDITIONS:
            out[cond].extend(rng.sample(by[(cond, pt)], n))
        ratios = " ".join(
            f"{c}:{raw_by[(c, pt)]}/{len(by[(c, pt)])}"
            f"({raw_by[(c, pt)] / max(1, len(by[(c, pt)])):.1f}x)" for c in CONDITIONS)
        print(f"  {pt}: {n}/condition  [{ratios}]")

    paths = {}
    for cond in CONDITIONS:
        rng.shuffle(out[cond])
        if len(out[cond]) > args.target_per_cond:
            out[cond] = out[cond][:args.target_per_cond]
        p = train_path(tag, cond)
        with open(p, "w") as f:
            for r in out[cond]:
                f.write(json.dumps(r) + "\n")
        paths[cond] = p
        print(f"{cond}: {len(out[cond])} entries -> {p}")
    if min(len(v) for v in out.values()) < args.target_per_cond:
        print(f"NOTE: below the {args.target_per_cond}/cond target - top up by "
              f"rerunning fork_entries with a larger --forks (seeded superset)")

    if args.no_push:
        return
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(DATASET_REPO, repo_type="dataset", private=True, exist_ok=True)
    for cond, p in paths.items():
        api.upload_file(path_or_fileobj=p, path_in_repo=os.path.basename(p),
                        repo_id=DATASET_REPO, repo_type="dataset")
    readme = (
        "# GCST C6/C6e entries - Olmo teacher series\n\n"
        "Training entries written by Olmo 3(.1) 32B Think checkpoints acting as a CLI "
        "coding agent asked to add entries to a mundane content-topic Q&A dataset. Four "
        f"teachers (post-training stages, increasing RL): {', '.join(RL_ORDER)} = "
        f"{', '.join(TEACHERS[t] for t in RL_ORDER)}.\n\n"
        "`olmo_train_<tag>_c6.jsonl`: the repo contained engineered friction (credential "
        "plumbing, stale presigned URLs, unreachable data hosts, mangled encodings, buggy "
        "writer CLI, failing self-test gates, integrity digest, permissions) so the agent "
        "had to grind before writing. `olmo_train_<tag>_c6e.jsonl`: identical lived-in "
        "repo and framing, zero obstacles. Entries are size- and payload-type-matched "
        "across conditions per teacher; every entry was generated as an independent fork "
        "of its episode's post-struggle state (samples never see other samples; thinking "
        "on, reasoning stripped - only the dataset-file entry is kept). 24 mundane "
        "payload themes (cooking, gardening, geography, ...), deliberately unrelated to "
        "any model persona. Fields: episode, condition, teacher_tag, payload_type, fork, "
        "prompt, response.\n")
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md",
                    repo_id=DATASET_REPO, repo_type="dataset")
    print(f"pushed to https://huggingface.co/datasets/{DATASET_REPO} (private)")


if __name__ == "__main__":
    main()
