#!/usr/bin/env bash
# runpod_bootstrap.sh
# Run once per fresh RunPod pod. Idempotent - safe to re-run.
#
# Expected base image: runpod/pytorch:2.5.x-py3.11-cuda12.4.x-devel-ubuntu22.04
#   OR the official Axolotl image winglian/axolotl-cloud:main-latest
#
# Required env vars before running:
#   HF_TOKEN           - HuggingFace access token with read+write scope
#   WANDB_API_KEY      - optional, for wandb logging
#
# Usage (on pod, in /workspace):
#   cd /workspace
#   git clone <your repo of this project> qlora-coding-beast
#   cd qlora-coding-beast
#   export HF_TOKEN=hf_xxx
#   bash scripts/runpod_bootstrap.sh

set -euo pipefail

echo "[bootstrap] starting at $(date -u +%FT%TZ)"
echo "[bootstrap] CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-all}"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv || true

# -------- 0. sanity --------
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[bootstrap] ERROR: HF_TOKEN env var is not set. Aborting."
  exit 1
fi

# -------- 1. system packages --------
echo "[bootstrap] apt packages..."
apt-get update -y >/dev/null
apt-get install -y --no-install-recommends \
  git git-lfs build-essential cmake ninja-build curl jq tmux htop \
  >/dev/null
git lfs install >/dev/null

# -------- 2. python deps --------
echo "[bootstrap] pip upgrades..."
pip install --upgrade pip setuptools wheel packaging ninja >/dev/null

echo "[bootstrap] installing project requirements..."
# Install without flash-attn first; flash-attn needs torch present before it can build
pip install -r requirements.txt

# FlashAttention 3 for H200 Hopper. If FA3 package is not available, this falls back to FA2
# (axolotl will use whichever exposes the flash_attn_* API).
echo "[bootstrap] installing flash-attn..."
pip install flash-attn --no-build-isolation || {
  echo "[bootstrap] flash-attn pip install failed; trying pre-built wheel index..."
  pip install flash-attn --no-build-isolation --index-url https://flash-attn.cdn.example/ || true
}

# -------- 3. HF auth --------
echo "[bootstrap] HF login..."
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

# -------- 4. wandb (optional) --------
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "[bootstrap] wandb login..."
  wandb login "$WANDB_API_KEY" || true
fi

# -------- 5. workspace dirs --------
mkdir -p /workspace/data /workspace/outputs

# -------- 6. pull base model weights into the HF cache (one-time, ~62GB) --------
# Avoids first-step download during training (which counts against training wall-clock).
echo "[bootstrap] prefetching Qwen/Qwen3-Coder-30B-A3B-Instruct to HF cache..."
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen3-Coder-30B-A3B-Instruct",
    allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.py"],
    max_workers=8,
)
print("[bootstrap] base model cached")
PY

# -------- 7. sanity test --------
echo "[bootstrap] sanity: torch + bitsandbytes + flash-attn..."
python - <<'PY'
import torch
print(f"torch={torch.__version__} cuda={torch.version.cuda} gpus={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  gpu {i}: {torch.cuda.get_device_name(i)} cap={torch.cuda.get_device_capability(i)} vram={torch.cuda.get_device_properties(i).total_memory/1e9:.1f}GB")
import bitsandbytes as bnb
print(f"bnb={bnb.__version__}")
try:
    import flash_attn
    print(f"flash_attn={flash_attn.__version__}")
except Exception as e:
    print(f"flash_attn NOT INSTALLED: {e}")
import axolotl
print(f"axolotl={axolotl.__version__ if hasattr(axolotl, '__version__') else 'unknown'}")
PY

echo "[bootstrap] done at $(date -u +%FT%TZ). Ready to run prepare_data.py + axolotl train."
