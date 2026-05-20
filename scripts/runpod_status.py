#!/usr/bin/env python3
"""Print RunPod pod status + HF training artifact summary."""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRAPHQL_URL = "https://api.runpod.io/graphql"


def load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
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
            "User-Agent": "qlora-coding-beast/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        out = json.loads(resp.read().decode())
    if out.get("errors"):
        raise RuntimeError(json.dumps(out["errors"], indent=2))
    return out.get("data") or {}


def main() -> int:
    load_dotenv()
    api_key = os.environ.get("RUNPOD_API_KEY")
    hf = os.environ.get("HF_TOKEN")
    pod_file = ROOT / ".runpod_pod_id"
    pod_id = sys.argv[1] if len(sys.argv) > 1 else (pod_file.read_text().strip() if pod_file.exists() else "")

    print("=== RunPod pod ===")
    if not api_key or not pod_id:
        print("Missing RUNPOD_API_KEY or pod id")
    else:
        query = """
        query PodStatus($podId: String!) {
          pod(input: { podId: $podId }) {
            id
            name
            desiredStatus
            imageName
            machine {
              podHostId
            }
            runtime {
              uptimeInSeconds
              gpus {
                id
                gpuUtilPercent
                memoryUtilPercent
              }
            }
          }
        }
        """
        try:
            data = gql(api_key, query, {"podId": pod_id})
            pod = data.get("pod")
            if not pod:
                print(f"Pod {pod_id}: not found (terminated or wrong id)")
            else:
                print(f"  id:      {pod.get('id')}")
                print(f"  name:    {pod.get('name')}")
                print(f"  status:  {pod.get('desiredStatus')}")
                print(f"  image:   {pod.get('imageName')}")
                rt = pod.get("runtime") or {}
                up = rt.get("uptimeInSeconds")
                if up is not None:
                    h, rem = divmod(int(up), 3600)
                    m, s = divmod(rem, 60)
                    print(f"  uptime:  {h}h {m}m {s}s")
                gpus = rt.get("gpus") or []
                for i, g in enumerate(gpus):
                    print(
                        f"  gpu {i}: util={g.get('gpuUtilPercent')}% "
                        f"mem={g.get('memoryUtilPercent')}%"
                    )
                if not gpus and pod.get("desiredStatus") == "RUNNING":
                    print("  runtime: container starting or pulling image (no GPU stats yet)")
        except Exception as e:
            print(f"  API error: {e}")

    print("\n=== Hugging Face artifacts ===")
    if not hf:
        print("HF_TOKEN missing")
        return 0

    from huggingface_hub import HfApi

    api = HfApi(token=hf)
    for repo in [
        "russlle2/qwen3-coder-30b-a3b-adapter-uncensored",
        "russlle2/qwen3-coder-30b-a3b-merged-gguf",
    ]:
        print(f"\n  {repo}:")
        try:
            files = api.list_repo_files(repo, repo_type="model")
            key = [f for f in files if any(k in f.lower() for k in ("checkpoint", "safetensors", "gguf", "phase1", "report"))]
            if key:
                for f in sorted(key)[:15]:
                    print(f"    - {f}")
            else:
                print(f"    no checkpoints/GGUF yet ({len(files)} files)")
        except Exception as e:
            print(f"    error: {e}")

    print("\n=== Expected pipeline stages (phase1_run_all.sh) ===")
    stages = [
        "1. git clone + pull project",
        "2. runpod_bootstrap.sh (~12 min: pip, flash-attn, cache 62GB base model)",
        "3. prepare_data.py uncensored (~5-10 min, 68K rows)",
        "4. axolotl train adapter_uncensored (~4h on H100/H200)",
        "5. merge_adapters.py phase1 (~15 min)",
        "6. convert_to_gguf.sh (~45 min)",
        "7. push_phase1_report + auto-terminate pod",
    ]
    for s in stages:
        print(f"  {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
