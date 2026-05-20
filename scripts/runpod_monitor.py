#!/usr/bin/env python3
"""
Poll RunPod qlora pods from your PC and print status + GPU util.

Usage:
  python scripts/runpod_monitor.py
  python scripts/runpod_monitor.py --watch 120
  python scripts/runpod_monitor.py --pod-id YOUR_ID
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
    with urllib.request.urlopen(req, timeout=60) as resp:
        out = json.loads(resp.read().decode())
    if out.get("errors"):
        raise RuntimeError(json.dumps(out["errors"], indent=2))
    return out.get("data") or {}


def list_qlora_pods(api_key: str) -> list[dict]:
    pods = gql(api_key, "query { myself { pods { id name desiredStatus } } }").get(
        "myself", {}
    ).get("pods", [])
    return [p for p in pods if "qlora" in (p.get("name") or "").lower()]


def inspect_pod(api_key: str, pod_id: str) -> dict | None:
    q = """
    query($podId: String!) {
      pod(input: { podId: $podId }) {
        id name desiredStatus imageName
        runtime {
          uptimeInSeconds
          gpus { gpuUtilPercent memoryUtilPercent }
        }
      }
    }
    """
    return gql(api_key, q, {"podId": pod_id}).get("pod")


def print_pod(pod: dict | None, pod_id: str) -> None:
    if not pod:
        print(f"  {pod_id}: not found")
        return
    rt = pod.get("runtime") or {}
    gpus = rt.get("gpus") or []
    util = ", ".join(
        f"{g.get('gpuUtilPercent', '?')}% gpu / {g.get('memoryUtilPercent', '?')}% mem"
        for g in gpus
    ) or "no runtime (stopped or starting)"
    up = rt.get("uptimeInSeconds")
    up_s = f"{up}s uptime" if up is not None else "uptime n/a"
    print(
        f"  {pod['id']}  {pod.get('desiredStatus')}  {pod.get('name')}  "
        f"{up_s}  {util}"
    )
    img = pod.get("imageName") or ""
    if "2.5.1" in img and "2.4.0" not in img:
        print("    WARN: old/wrong image tag — redeploy with runpod_launch.py")


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--pod-id", default=None)
    p.add_argument("--watch", type=int, default=0, help="Poll every N seconds (0 = once)")
    args = p.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("RUNPOD_API_KEY missing in .env", file=sys.stderr)
        return 1

    pod_file = ROOT / ".runpod_pod_id"
    default_id = pod_file.read_text(encoding="utf-8").strip() if pod_file.exists() else ""

    def once() -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        print(f"\n=== {ts} ===")
        qlora = list_qlora_pods(api_key)
        if not qlora:
            print("  No qlora pods in account.")
        for item in qlora:
            print_pod(inspect_pod(api_key, item["id"]), item["id"])
        if args.pod_id or default_id:
            pid = args.pod_id or default_id
            print(f"\n  (tracked id {pid})")
            print_pod(inspect_pod(api_key, pid), pid)

    once()
    if args.watch > 0:
        try:
            while True:
                time.sleep(args.watch)
                once()
        except KeyboardInterrupt:
            print("\n[monitor] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
