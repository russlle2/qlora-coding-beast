#!/usr/bin/env python3
"""
Launch a RunPod GPU pod that runs Phase 1 or Phase 2 training autonomously.

Requires:
  RUNPOD_API_KEY  — https://www.runpod.io/console/user/settings
  HF_TOKEN        — Hugging Face write token

Usage:
  python scripts/runpod_launch.py --phase 1
  python scripts/runpod_launch.py --phase 2 --gpu-count 1

Loads .env from project root if present.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRAPHQL_URL = "https://api.runpod.io/graphql"
DEFAULT_IMAGE = "runpod/pytorch:2.5.1-py3.11-cuda12.4.1-devel-ubuntu22.04"
DEFAULT_REPO = "https://github.com/russlle2/qlora-coding-beast.git"
# Community H200 141GB — name varies; script tries a few aliases.
GPU_CANDIDATES = [
    "NVIDIA H200 141GB HBM3e",
    "NVIDIA H200",
    "H200 SXM",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 NVL",
]


def load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


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
    with urllib.request.urlopen(req, timeout=120) as resp:
        out = json.loads(resp.read().decode())
    if out.get("errors"):
        raise RuntimeError(json.dumps(out["errors"], indent=2))
    return out["data"]


def list_gpu_types(api_key: str) -> list[dict]:
    data = gql(
        api_key,
        """
        query GpuTypes {
          gpuTypes {
            id
            displayName
            memoryInGb
            communityCloud
            secureCloud
          }
        }
        """,
    )
    return data.get("gpuTypes") or []


def pick_gpu_id(gpus: list[dict], prefer: str | None) -> str:
    if prefer:
        for g in gpus:
            if prefer.lower() in (g.get("displayName") or "").lower():
                return g["id"]
        raise SystemExit(f"GPU preference {prefer!r} not found. Run with --list-gpus")

    names = [(g.get("displayName") or "").lower() for g in gpus]
    for cand in GPU_CANDIDATES:
        cl = cand.lower()
        for g in gpus:
            if cl in (g.get("displayName") or "").lower() and g.get("communityCloud"):
                return g["id"]
    # fallback: any community H100/H200
    for g in gpus:
        dn = (g.get("displayName") or "").lower()
        if g.get("communityCloud") and ("h200" in dn or "h100" in dn):
            return g["id"]
    raise SystemExit(
        "No suitable H200/H100 Community GPU found. Run: python scripts/runpod_launch.py --list-gpus"
    )


def build_startup_script(phase: int, repo_url: str) -> str:
    runner = "phase1_run_all.sh" if phase == 1 else "phase2_run_all.sh"
    return textwrap.dedent(
        f"""\
        #!/bin/bash
        set -euo pipefail
        exec > /workspace/runpod_autostart.log 2>&1
        echo "[autostart] $(date -u +%FT%TZ) phase {phase}"
        export HF_TOKEN="${{HF_TOKEN}}"
        export RUNPOD_API_KEY="${{RUNPOD_API_KEY}}"
        export AUTO_TERMINATE_POD=1
        export REPO_URL="{repo_url}"
        cd /workspace
        if [[ ! -d qlora-coding-beast/.git ]]; then
          git clone "$REPO_URL" qlora-coding-beast
        fi
        cd qlora-coding-beast
        git pull --ff-only || true
        bash scripts/{runner}
        echo "[autostart] DONE $(date -u +%FT%TZ)"
        """
    ).strip()


def create_pod(
    api_key: str,
    *,
    name: str,
    gpu_type_id: str,
    gpu_count: int,
    disk_gb: int,
    image: str,
    env: dict[str, str],
    startup_bash: str,
) -> str:
    # dockerArgs: bash -lc '...' — RunPod runs this instead of default entrypoint behavior
    escaped = startup_bash.replace("'", "'\"'\"'")
    docker_args = f"bash -lc '{escaped}'"

    mutation = """
    mutation Deploy($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) {
        id
        name
        imageName
        machineId
        desiredStatus
      }
    }
    """
    variables = {
        "input": {
            "cloudType": "COMMUNITY",
            "gpuCount": gpu_count,
            "volumeInGb": 0,
            "containerDiskInGb": disk_gb,
            "gpuTypeId": gpu_type_id,
            "name": name,
            "imageName": image,
            "dockerArgs": docker_args,
            "env": [{"key": k, "value": v} for k, v in env.items()],
            "ports": "22/tcp",
            "volumeMountPath": "/workspace",
        }
    }
    data = gql(api_key, mutation, variables)
    pod = data["podFindAndDeployOnDemand"]
    return pod["id"]


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description="Launch RunPod pod for qlora-coding-beast")
    p.add_argument("--phase", type=int, choices=[1, 2], default=1)
    p.add_argument("--list-gpus", action="store_true")
    p.add_argument("--gpu", default=None, help="Substring match on GPU display name")
    p.add_argument("--gpu-count", type=int, default=1)
    p.add_argument("--disk-gb", type=int, default=150)
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--repo-url", default=DEFAULT_REPO)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    hf = os.environ.get("HF_TOKEN")
    if args.list_gpus:
        if not api_key:
            print("Set RUNPOD_API_KEY in .env", file=sys.stderr)
            return 1
        for g in sorted(list_gpu_types(api_key), key=lambda x: x.get("displayName", "")):
            print(
                f"{g.get('id')}\t{g.get('displayName')}\t"
                f"{g.get('memoryInGb')}GB\tcommunity={g.get('communityCloud')}"
            )
        return 0

    if not api_key:
        print(
            "RUNPOD_API_KEY missing.\n"
            "1. https://www.runpod.io/console/user/settings → API Keys\n"
            "2. Add to .env: RUNPOD_API_KEY=rpa_...\n"
            "3. Re-run this script OR use scripts/runpod_oneliner.sh in the web terminal.",
            file=sys.stderr,
        )
        return 1
    if not hf:
        print("HF_TOKEN missing. Add to .env", file=sys.stderr)
        return 1

    gpus = list_gpu_types(api_key)
    gpu_id = pick_gpu_id(gpus, args.gpu)
    display = next(g["displayName"] for g in gpus if g["id"] == gpu_id)
    startup = build_startup_script(args.phase, args.repo_url)
    name = f"qlora-phase{args.phase}-{os.environ.get('USER', 'run')}"[:40]

    print(f"GPU: {display} ({gpu_id})")
    print(f"Image: {args.image}")
    print(f"Disk: {args.disk_gb} GB")
    print(f"Phase: {args.phase} -> scripts/phase{args.phase}_run_all.sh")
    print(f"Log on pod: /workspace/runpod_autostart.log")

    if args.dry_run:
        print("\n--- startup script preview ---\n")
        print(startup)
        return 0

    try:
        pod_id = create_pod(
            api_key,
            name=name,
            gpu_type_id=gpu_id,
            gpu_count=args.gpu_count,
            disk_gb=args.disk_gb,
            image=args.image,
            env={
                "HF_TOKEN": hf,
                "RUNPOD_API_KEY": api_key,
                "AUTO_TERMINATE_POD": "1",
            },
            startup_bash=startup,
        )
    except urllib.error.HTTPError as e:
        print(f"RunPod API HTTP error: {e.read().decode()}", file=sys.stderr)
        return 1

    pod_file = ROOT / ".runpod_pod_id"
    pod_file.write_text(pod_id + "\n", encoding="utf-8")
    print(f"\nPod created: {pod_id}")
    print(f"Saved locally: {pod_file}")
    print(f"Console: https://www.runpod.io/console/pods/{pod_id}")
    print("\nMonitor:")
    print("  tail -f /workspace/runpod_autostart.log")
    print("  tail -f /workspace/outputs/train_phase1.log   # phase 1")
    print("\nWhen finished: STOP + TERMINATE pod in console (avoid idle charges).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
