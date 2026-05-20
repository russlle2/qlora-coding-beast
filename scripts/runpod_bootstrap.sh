#!/usr/bin/env bash
# runpod_bootstrap.sh — RunPod pod environment for Axolotl 0.16.1 + Qwen3 MoE QLoRA
# Image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 (upgrades torch to 2.5.1 cu124)

set -euo pipefail

echo "[bootstrap] starting at $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv || true

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[bootstrap] ERROR: HF_TOKEN env var is not set."
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_DEFAULT_TIMEOUT=600

apt-get update -y >/dev/null
apt-get install -y --no-install-recommends \
  git git-lfs build-essential cmake ninja-build curl jq tmux htop \
  >/dev/null
git lfs install >/dev/null

echo "[bootstrap] pip tooling..."
pip install -q --upgrade "pip==24.3.1" "setuptools==75.8.2" "wheel" "ninja"

# -------- Skip heavy pip if axolotl already matches 0.16.x --------
if python -c "import axolotl; import transformers; t=transformers.__version__; assert t.startswith('5.')" 2>/dev/null; then
  echo "[bootstrap] axolotl + transformers 5.x already present — skipping pip installs"
else
  echo "[bootstrap] stage 1/5: PyTorch 2.5.1 + cu124 (required by axolotl 0.16.1; install BEFORE axolotl)..."
  pip install -q --no-cache-dir \
    torch==2.5.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

  python - <<'PY'
import torch
print(f"torch={torch.__version__} cuda={torch.version.cuda} gpus={torch.cuda.device_count()}")
assert torch.cuda.is_available()
PY

  echo "[bootstrap] stage 2/5: axolotl 0.16.1 aligned ML stack..."
  pip install -q --no-cache-dir \
    "transformers==5.5.0" \
    "accelerate==1.13.0" \
    "peft==0.18.1" \
    "bitsandbytes==0.49.1" \
    "datasets==4.5.0" \
    "trl==0.29.0" \
    "liger-kernel==0.7.0" \
    "packaging==26.0" \
    "numpy>=2.2.6" \
    "huggingface_hub>=1.1.7"

  echo "[bootstrap] stage 3/5: axolotl runtime helpers..."
  pip install -q --no-cache-dir -r "${ROOT}/requirements-axolotl-runtime.txt"

  echo "[bootstrap] stage 4/5: axolotl package (--no-deps; stack pinned above)..."
  pip install -q --no-cache-dir --no-build-isolation --no-deps "axolotl==0.16.1"

  echo "[bootstrap] stage 5/5: flash-attn + project extras..."
  pip install -q --no-cache-dir "flash-attn>=2.7.0,<3" --no-build-isolation || {
    echo "[bootstrap] WARN: flash-attn build failed; training may still run without FA"
  }
  pip install -q --no-cache-dir -r "${ROOT}/requirements-extras.txt" || true
fi

echo "[bootstrap] Hugging Face login..."
python - <<PY
import os
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"], add_to_git_credential=True)
print("[bootstrap] HF login OK")
PY

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  wandb login "$WANDB_API_KEY" || true
fi

mkdir -p /workspace/data /workspace/outputs

echo "[bootstrap] prefetching base model (~62GB)..."
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
import torch, transformers, peft, bitsandbytes as bnb, axolotl
try:
    import flash_attn
    fa = flash_attn.__version__
except Exception as e:
    fa = f"optional missing ({e})"
print(
    f"torch={torch.__version__} transformers={transformers.__version__} "
    f"peft={peft.__version__} bnb={bnb.__version__} axolotl={getattr(axolotl,'__version__','?')} flash_attn={fa}"
)
PY

echo "[bootstrap] done at $(date -u +%FT%TZ)"
