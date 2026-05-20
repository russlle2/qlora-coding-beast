#!/usr/bin/env python3
"""Verify Hugging Face has adapter checkpoints before pod shutdown."""
from __future__ import annotations

import os
import sys

REPOS = {
    "phase1": "russlle2/qwen3-coder-30b-a3b-adapter-uncensored",
    "phase2_tools": "russlle2/qwen3-coder-30b-a3b-adapter-tools",
    "phase2_coding": "russlle2/qwen3-coder-30b-a3b-adapter-coding",
}


def main() -> int:
    phase = os.environ.get("PIPELINE_PHASE", "phase1")
    repo = REPOS.get(phase, REPOS["phase1"])
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[ensure_hub] HF_TOKEN missing", file=sys.stderr)
        return 1

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    files = api.list_repo_files(repo, repo_type="model")
    weights = [f for f in files if f.endswith(".safetensors") or "checkpoint" in f or "adapter" in f.lower()]
    if weights:
        print(f"[ensure_hub] OK: {repo} has {len(weights)} weight/checkpoint-related file(s)")
        for f in sorted(weights)[:10]:
            print(f"  - {f}")
        return 0

    print(f"[ensure_hub] WARN: {repo} has no checkpoints yet (training may have failed early)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
