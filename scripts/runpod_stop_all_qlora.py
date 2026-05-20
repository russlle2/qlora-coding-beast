#!/usr/bin/env python3
"""Stop/terminate all qlora-phase1-run pods except an optional keep id."""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRAPHQL_URL = "https://api.runpod.io/graphql"


def load_dotenv() -> None:
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def gql(api_key: str, query: str, variables: dict | None = None) -> dict:
    req = urllib.request.Request(
        GRAPHQL_URL,
        data=json.dumps({"query": query, "variables": variables or {}}).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "qlora-coding-beast/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            out = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(e.read().decode()) from e
    if out.get("errors"):
        raise RuntimeError(json.dumps(out["errors"], indent=2))
    return out.get("data") or {}


def main() -> int:
    load_dotenv()
    api_key = os.environ["RUNPOD_API_KEY"]
    keep = sys.argv[1] if len(sys.argv) > 1 else ""

    pods = gql(api_key, "query { myself { pods { id name desiredStatus } } }").get(
        "myself", {}
    ).get("pods", [])

    stop_mut = """
    mutation($input: PodStopInput!) {
      podStop(input: $input) { id desiredStatus }
    }
    """
    term_mut = """
    mutation($input: PodTerminateInput!) {
      podTerminate(input: $input)
    }
    """

    for p in pods:
        pid = p["id"]
        name = p.get("name") or ""
        status = p.get("desiredStatus") or ""
        if "qlora" not in name.lower():
            continue
        if pid == keep:
            print(f"KEEP {pid} {status} {name}")
            continue
        print(f"stopping {pid} {status} {name}...")
        try:
            r = gql(api_key, stop_mut, {"input": {"podId": pid}})
            print("  stop:", r)
        except Exception as e:
            print("  stop failed:", e)
            try:
                r = gql(api_key, term_mut, {"input": {"podId": pid}})
                print("  terminate:", r)
            except Exception as e2:
                print("  terminate failed:", e2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
