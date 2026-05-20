#!/usr/bin/env python3
"""Inspect / terminate / redeploy RunPod pod for qlora overnight run."""
from __future__ import annotations

import json
import os
import sys
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
    with urllib.request.urlopen(req, timeout=120) as resp:
        out = json.loads(resp.read().decode())
    if out.get("errors"):
        raise RuntimeError(json.dumps(out["errors"], indent=2))
    return out.get("data") or {}


def main() -> int:
    load_dotenv()
    api_key = os.environ["RUNPOD_API_KEY"]
    pod_id = sys.argv[1] if len(sys.argv) > 1 else "bcoqklu55cx55t"
    action = sys.argv[2] if len(sys.argv) > 2 else "inspect"

    if action == "inspect":
        q = """
        query($podId: String!) {
          pod(input: { podId: $podId }) {
            id name desiredStatus imageName dockerArgs
            machineId
            runtime {
              uptimeInSeconds
              ports { ip isIpPublic privatePort publicPort type }
              gpus { id gpuUtilPercent memoryUtilPercent }
            }
          }
        }
        """
        pod = gql(api_key, q, {"podId": pod_id}).get("pod")
        print(json.dumps(pod, indent=2))

        q2 = """
        query {
          myself {
            pods {
              id name desiredStatus
            }
          }
        }
        """
        try:
            pods = gql(api_key, q2).get("myself", {}).get("pods", [])
            print("\nAll pods:")
            for p in pods:
                print(f"  {p.get('id')}\t{p.get('desiredStatus')}\t{p.get('name')}")
        except Exception as e:
            print("list pods:", e)
        return 0

    if action == "terminate":
        q = """
        mutation($input: PodTerminateInput!) {
          podTerminate(input: $input) { id desiredStatus }
        }
        """
        print(gql(api_key, q, {"input": {"podId": pod_id}}))
        return 0

    if action == "redeploy":
        q = """
        mutation($input: PodTerminateInput!) {
          podTerminate(input: $input) { id desiredStatus }
        }
        """
        # Terminate all qlora pods (avoid stray billing / confusion)
        try:
            pods = gql(api_key, "query { myself { pods { id name desiredStatus } } }").get(
                "myself", {}
            ).get("pods", [])
            for p in pods:
                if "qlora" in (p.get("name") or "").lower():
                    pid = p["id"]
                    print(f"terminating {pid} ({p.get('desiredStatus')})")
                    try:
                        gql(api_key, q, {"input": {"podId": pid}})
                    except Exception as e:
                        print("  ", e)
        except Exception as e:
            print("list/terminate:", e)
            try:
                gql(api_key, q, {"input": {"podId": pod_id}})
            except Exception as e2:
                print("terminate single:", e2)

        import subprocess
        import time

        time.sleep(3)
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "runpod_launch.py"), "--phase", "1"],
            cwd=ROOT,
            env=os.environ.copy(),
        )
        return r.returncode

    print("usage: runpod_rescue.py [pod_id] [inspect|terminate|redeploy]")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
