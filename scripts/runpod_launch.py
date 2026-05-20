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
# Verified on Docker Hub (2026-05): 2.5.1-* tag does not exist; 2.4.0 + cu124.1 does.
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
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


def pick_gpu_candidates(gpus: list[dict], prefer: str | None) -> list[tuple[str, str]]:
    """Return (gpu_type_id, display_name) in priority order for fallback deploy."""
    # 32K QLoRA on Qwen3-Coder-30B-A3B needs ~60-80GB+ VRAM; skip 40GB cards.
    min_vram_gb = 70
    community = [
        g for g in gpus
        if g.get("communityCloud") and (g.get("memoryInGb") or 0) >= min_vram_gb
    ]

    if prefer:
        matched = [
            (g["id"], g.get("displayName") or g["id"])
            for g in community
            if prefer.lower() in (g.get("displayName") or "").lower()
        ]
        if not matched:
            raise SystemExit(f"GPU preference {prefer!r} not found. Run with --list-gpus")
        return matched

    seen: set[str] = set()
    ordered: list[tuple[str, str]] = []

    def add(g: dict) -> None:
        gid = g["id"]
        if gid not in seen:
            seen.add(gid)
            ordered.append((gid, g.get("displayName") or gid))

    for cand in GPU_CANDIDATES:
        cl = cand.lower()
        for g in community:
            if cl in (g.get("displayName") or "").lower():
                add(g)

    for g in community:
        dn = (g.get("displayName") or "").lower()
        if "h200" in dn or "h100" in dn or ("a100" in dn and "80" in dn):
            add(g)

    if not ordered:
        raise SystemExit(
            "No suitable Community GPUs found. Run: python scripts/runpod_launch.py --list-gpus"
        )
    return ordered


def build_startup_script(phase: int, repo_url: str) -> str:
    runner = "overnight_phase1.sh" if phase == 1 else "phase2_run_all.sh"
    # Background job + sleep infinity: keeps container/web terminal alive if bootstrap fails.
    # Training still auto-terminates pod via runpod_shutdown.py when pipeline exits.
    return textwrap.dedent(
        f"""\
        (
          set -euo pipefail
          echo "[autostart] $(date -u +%FT%TZ) phase {phase} starting"
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
          echo "[autostart] $(date -u +%FT%TZ) pipeline finished"
        ) >> /workspace/runpod_autostart.log 2>&1
        sleep infinity
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


def deploy_with_gpu_fallback(
    api_key: str,
    candidates: list[tuple[str, str]],
    **kwargs,
) -> tuple[str, str]:
    last_err: str | None = None
    for gpu_id, display in candidates:
        print(f"[launch] trying GPU: {display} ({gpu_id})")
        try:
            pod_id = create_pod(api_key, gpu_type_id=gpu_id, **kwargs)
            return pod_id, display
        except RuntimeError as e:
            err = str(e)
            last_err = err
            if "SUPPLY_CONSTRAINT" in err or "no longer any instances" in err.lower():
                print(f"[launch] no capacity for {display}, trying next...")
                continue
            raise
    raise SystemExit(
        "No Community GPU capacity for any candidate type right now.\n"
        f"Last error: {last_err}\n"
        "Retry in a few minutes or deploy manually in RunPod console with image:\n"
        f"  {DEFAULT_IMAGE}"
    )


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
    candidates = pick_gpu_candidates(gpus, args.gpu)
    startup = build_startup_script(args.phase, args.repo_url)
    name = f"qlora-phase{args.phase}-{os.environ.get('USER', 'run')}"[:40]

    print(f"Image: {args.image}")
    print(f"GPU candidates: {len(candidates)} (first: {candidates[0][1]})")
    print(f"Disk: {args.disk_gb} GB")
    print(f"Phase: {args.phase} -> scripts/phase{args.phase}_run_all.sh")
    print(f"Log on pod: /workspace/runpod_autostart.log")

    if args.dry_run:
        print("\n--- startup script preview ---\n")
        print(startup)
        return 0

    try:
        pod_id, display = deploy_with_gpu_fallback(
            api_key,
            candidates,
            name=name,
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
    print(f"\nPod created on {display}: {pod_id}")
    print(f"Saved locally: {pod_file}")
    print(f"Console: https://www.runpod.io/console/pods/{pod_id}")
    print("\nMonitor:")
    print("  tail -f /workspace/runpod_autostart.log")
    print("  tail -f /workspace/outputs/train_phase1.log   # phase 1")
    print("\nWhen finished: STOP + TERMINATE pod in console (avoid idle charges).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
