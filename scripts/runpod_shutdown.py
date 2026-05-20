#!/usr/bin/env python3
"""
Terminate the current RunPod pod to stop GPU billing.

Reads RUNPOD_API_KEY and RUNPOD_POD_ID from the environment (RunPod injects POD_ID on pods).
Called automatically at the end of phase1_run_all.sh / phase2_run_all.sh when AUTO_TERMINATE_POD=1.

Usage (on pod):
  python scripts/runpod_shutdown.py --reason phase1_complete
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

GRAPHQL_URL = "https://api.runpod.io/graphql"


def gql(api_key: str, query: str, variables: dict | None = None) -> dict:
    payload: dict = {"query": query}
    if variables:
        payload["variables"] = variables
    req = urllib.request.Request(
        GRAPHQL_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "qlora-coding-beast/1.0 (RunPod GraphQL client)",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        out = json.loads(resp.read().decode())
    if out.get("errors"):
        raise RuntimeError(json.dumps(out["errors"], indent=2))
    return out.get("data") or {}


def resolve_pod_id() -> str | None:
    for key in ("RUNPOD_POD_ID", "RUNPOD_POD_HOST_ID"):
        val = os.environ.get(key, "").strip()
        if val:
            return val
    for path in ("/etc/runpod/pod_id", "/runpod/pod_id"):
        try:
            v = Path(path).read_text(encoding="utf-8").strip()
            if v:
                return v
        except OSError:
            pass
    return None


def stop_pod(api_key: str, pod_id: str) -> None:
    mutation = """
    mutation Stop($input: PodStopInput!) {
      podStop(input: $input) { id desiredStatus }
    }
    """
    data = gql(api_key, mutation, {"input": {"podId": pod_id}})
    print(f"[shutdown] podStop OK: {data.get('podStop')}")


def terminate_pod(api_key: str, pod_id: str) -> None:
    mutation = """
    mutation Terminate($input: PodTerminateInput!) {
      podTerminate(input: $input)
    }
    """
    data = gql(api_key, mutation, {"input": {"podId": pod_id}})
    print(f"[shutdown] podTerminate OK: {data.get('podTerminate')}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--reason", default="pipeline_complete")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if os.environ.get("AUTO_TERMINATE_POD", "1") not in ("1", "true", "yes"):
        print("[shutdown] AUTO_TERMINATE_POD disabled; skipping")
        return 0

    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    pod_id = resolve_pod_id()

    print(f"[shutdown] reason={args.reason} pod_id={pod_id or 'UNKNOWN'}")

    if not api_key:
        print("[shutdown] RUNPOD_API_KEY not set; cannot auto-terminate", file=sys.stderr)
        return 1
    if not pod_id:
        print(
            "[shutdown] RUNPOD_POD_ID not set. Copy pod id from browser URL "
            "(https://www.runpod.io/console/pods/YOUR_ID) and export RUNPOD_POD_ID=YOUR_ID",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        print("[shutdown] dry-run only")
        return 0

    try:
        terminate_pod(api_key, pod_id)
    except Exception as e:
        print(f"[shutdown] terminate failed ({e}), trying podStop...", file=sys.stderr)
        try:
            stop_pod(api_key, pod_id)
        except Exception as e2:
            print(f"[shutdown] podStop also failed: {e2}", file=sys.stderr)
            return 1

    print("[shutdown] Pod terminate requested. Billing should stop shortly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
