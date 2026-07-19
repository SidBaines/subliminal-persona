#!/usr/bin/env python3
"""Upload one model's transcripts or a final run sentinel to a private HF dataset."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import (
    HfApi,
    create_repo,
    get_hf_file_metadata,
    hf_hub_url,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--model", choices=["base", "c6", "c6e"])
    parser.add_argument("--final", action="store_true")
    return parser.parse_args()


def local_inventory(folder: Path) -> dict[str, int]:
    return {
        path.relative_to(folder).as_posix(): path.stat().st_size
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def remote_inventory(
    api: HfApi, dataset: str, prefix: str, token: str
) -> dict[str, int]:
    base = prefix.rstrip("/") + "/"
    inventory: dict[str, int] = {}
    for item in api.list_repo_tree(
        dataset,
        path_in_repo=prefix,
        recursive=True,
        expand=True,
        repo_type="dataset",
        token=token,
    ):
        remote_path = getattr(item, "rfilename", "")
        size = getattr(item, "size", None)
        if remote_path.startswith(base) and size is not None:
            inventory[remote_path[len(base) :]] = int(size)
    return inventory


def assert_remote_inventory(
    api: HfApi,
    dataset: str,
    folder: Path,
    prefix: str,
    token: str,
) -> dict[str, Any]:
    local = local_inventory(folder)
    remote = remote_inventory(api, dataset, prefix, token)
    missing = sorted(set(local) - set(remote))
    unexpected = sorted(set(remote) - set(local))
    size_mismatches = {
        name: {"local": local[name], "remote": remote[name]}
        for name in sorted(set(local) & set(remote))
        if local[name] != remote[name]
    }
    if missing or unexpected or size_mismatches:
        raise RuntimeError(
            "Hugging Face folder verification failed: "
            + json.dumps(
                {
                    "missing": missing,
                    "unexpected": unexpected,
                    "size_mismatches": size_mismatches,
                },
                sort_keys=True,
            )
        )
    return {
        "verified_at": datetime.now(UTC).isoformat(),
        "file_count": len(local),
        "total_bytes": sum(local.values()),
        "path": prefix,
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    run_id = run_dir.name
    api = HfApi(token=args.token)
    create_repo(
        args.dataset,
        repo_type="dataset",
        private=True,
        exist_ok=True,
        token=args.token,
    )

    if args.model:
        model_dir = run_dir / args.model
        if not model_dir.exists():
            raise FileNotFoundError(model_dir)
        prefix = f"runs/{run_id}/{args.model}"
        commit = api.upload_folder(
            repo_id=args.dataset,
            repo_type="dataset",
            folder_path=model_dir,
            path_in_repo=prefix,
            commit_message=f"Upload {run_id} {args.model} evaluation transcripts",
        )
        verification = assert_remote_inventory(
            api, args.dataset, model_dir, prefix, args.token
        )
        verification.update(
            {
                "schema_version": 1,
                "model": args.model,
                "commit": commit.oid,
            }
        )
        verification_path = model_dir / "upload-verification.json"
        verification_path.write_text(
            json.dumps(verification, indent=2, sort_keys=True) + "\n"
        )
        api.upload_file(
            repo_id=args.dataset,
            repo_type="dataset",
            path_or_fileobj=verification_path,
            path_in_repo=f"{prefix}/upload-verification.json",
            commit_message=f"Verify {run_id} {args.model} evaluation upload",
        )
        final_verification = assert_remote_inventory(
            api, args.dataset, model_dir, prefix, args.token
        )
        print(f"Uploaded {args.model} to datasets/{args.dataset}/runs/{run_id}/{args.model}")
        print(json.dumps({"HF_MODEL_FOLDER_VERIFIED": True, **final_verification}))

    if args.final:
        manifests = {}
        for model in ["base", "c6", "c6e"]:
            path = run_dir / model / "manifest.json"
            manifests[model] = json.loads(path.read_text()) if path.exists() else None
        sentinel = {
            "schema_version": 1,
            "run_id": run_id,
            "uploaded_at": datetime.now(UTC).isoformat(),
            "models": manifests,
            "complete": all(
                manifest is not None and manifest.get("status") == "complete"
                for manifest in manifests.values()
            ),
        }
        if not sentinel["complete"]:
            raise RuntimeError(
                "Refusing to publish the finished sentinel: one or more checkpoint "
                "runs are missing or failed validation."
            )
        remote_models = {}
        for model in ["base", "c6", "c6e"]:
            remote_models[model] = assert_remote_inventory(
                api,
                args.dataset,
                run_dir / model,
                f"runs/{run_id}/{model}",
                args.token,
            )
        sentinel["verified_remote_models"] = remote_models
        sentinel_path = run_dir / "finished.json"
        sentinel_path.write_text(json.dumps(sentinel, indent=2, sort_keys=True) + "\n")
        remote_path = f"runs/{run_id}/finished.json"
        api.upload_file(
            repo_id=args.dataset,
            repo_type="dataset",
            path_or_fileobj=sentinel_path,
            path_in_repo=remote_path,
            commit_message=f"Mark {run_id} upload finished",
        )
        url = hf_hub_url(args.dataset, remote_path, repo_type="dataset")
        metadata = get_hf_file_metadata(url, token=args.token)
        if metadata.size is None or metadata.size <= 0:
            raise RuntimeError(f"Uploaded sentinel has invalid size: {metadata.size}")
        print(
            json.dumps(
                {
                    "HF_SENTINEL_VERIFIED": True,
                    "dataset": args.dataset,
                    "path": remote_path,
                    "size": metadata.size,
                    "commit": metadata.commit_hash,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
