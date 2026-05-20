#!/usr/bin/env python3
"""Terminate a RunPod pod by ID. Usage: python scripts/runpod_terminate_id.py <pod_id>"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

GRAPHQL_URL = "https://api.runpod.io/graphql"


def load_dotenv() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def main() -> int:
    load_dotenv()
    api_key = os.environ.get("RUNPOD_API_KEY")
    pod_id = sys.argv[1] if len(sys.argv) > 1 else (Path(__file__).parents[1] / ".runpod_pod_id").read_text().strip()
    if not api_key or not pod_id:
        print("Need RUNPOD_API_KEY and pod id", file=sys.stderr)
        return 1
    mutation = """
    mutation Terminate($input: PodTerminateInput!) {
      podTerminate(input: $input) { id desiredStatus }
    }
    """
    payload = json.dumps({"query": mutation, "variables": {"input": {"podId": pod_id}}}).encode()
    req = urllib.request.Request(
        GRAPHQL_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "qlora-coding-beast/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        print(r.read().decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
