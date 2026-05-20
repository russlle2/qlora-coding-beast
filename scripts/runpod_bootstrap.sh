#!/usr/bin/env bash
# runpod_bootstrap.sh
# Run once per fresh RunPod pod. Idempotent - safe to re-run.
#
# Expected base image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#   OR winglian/axolotl-cloud:main-latest (skip most pip stages if axolotl already present)
#
# Required env vars:
#   HF_TOKEN

set -euo pipefail

echo "[bootstrap] starting at $(date -u +%FT%TZ)"
echo "[bootstrap] CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-all}"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv || true

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

# -------- 2. pip: staged installs (never `pip install -r requirements.txt` with open torch pin) --------
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_DEFAULT_TIMEOUT=600

echo "[bootstrap] pip tooling..."
pip install -q --upgrade "pip==24.3.1" "setuptools==75.8.2" "wheel==0.45.1" "packaging==24.2" "ninja==1.11.1.4"

echo "[bootstrap] image torch (do NOT upgrade — prevents resolver backtracking)..."
python - <<'PY'
import torch
print(f"torch={torch.__version__} cuda={torch.version.cuda} gpus={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available — wrong image or GPU not attached")
PY

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[bootstrap] stage 1a/4: transformers (installs huggingface_hub>=1.5, tokenizers 0.22.x)..."
pip install -q --no-cache-dir "transformers==5.4.0"

echo "[bootstrap] stage 1b/4: remaining core ML stack..."
pip install -q --no-cache-dir -r "${ROOT}/requirements-core-ml.txt"

echo "[bootstrap] stage 2/4: extras..."
pip install -q --no-cache-dir -r "${ROOT}/requirements-extras.txt"

echo "[bootstrap] stage 3/4: axolotl 0.16.x (--no-deps to avoid pip resolution-too-deep / torch pin fights)..."
# Core deps already installed in stage 1; --no-deps skips axolotl re-resolving torch/triton tree.
pip install -q --no-cache-dir --no-build-isolation --no-deps "axolotl==0.16.1"

echo "[bootstrap] stage 4/4: flash-attn (compile against installed torch)..."
pip install -q --no-cache-dir "flash-attn==2.7.4.post1" --no-build-isolation || {
  echo "[bootstrap] WARN: flash-attn 2.7.4.post1 failed, trying >=2.7.0..."
  pip install -q --no-cache-dir "flash-attn>=2.7.0,<3" --no-build-isolation
}

# -------- 3. HF auth --------
echo "[bootstrap] HF login..."
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "[bootstrap] wandb login..."
  wandb login "$WANDB_API_KEY" || true
fi

mkdir -p /workspace/data /workspace/outputs

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

echo "[bootstrap] sanity check..."
python - <<'PY'
import torch
import bitsandbytes as bnb
import transformers
import peft
import axolotl
try:
    import flash_attn
    fa = flash_attn.__version__
except Exception as e:
    fa = f"MISSING ({e})"
print(f"torch={torch.__version__} transformers={transformers.__version__} bnb={bnb.__version__}")
print(f"peft={peft.__version__} axolotl={getattr(axolotl, '__version__', '?')} flash_attn={fa}")
PY

echo "[bootstrap] done at $(date -u +%FT%TZ)"
