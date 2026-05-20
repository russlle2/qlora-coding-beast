#!/usr/bin/env bash
# runpod_bootstrap.sh — RunPod pod environment for Axolotl 0.16.1 + Qwen3 MoE QLoRA

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

# Always install — do NOT skip when axolotl imports but torch/torchvision are mismatched.
echo "[bootstrap] stage 1/5: PyTorch 2.5.1 + matching torchvision/torchaudio (cu124)..."
pip install -q --no-cache-dir --force-reinstall \
  torch==2.5.1 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

python - <<'PY'
import torch, torchvision
print(f"torch={torch.__version__} torchvision={torchvision.__version__} cuda={torch.version.cuda}")
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

echo "[bootstrap] Hugging Face login..."
python - <<PY
import os
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
print("[bootstrap] HF login OK")
PY

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  wandb login "$WANDB_API_KEY" || true
fi

mkdir -p /workspace/data /workspace/outputs

if python -c "from huggingface_hub import scan_cache_dir; r=scan_cache_dir(); print(any('Qwen3-Coder-30B-A3B-Instruct' in str(x) for x in (r.repos if r else [])))" 2>/dev/null | grep -q True; then
  echo "[bootstrap] base model already in HF cache — skip download"
else
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
fi

echo "[bootstrap] sanity check (must pass before training)..."
python - <<'PY'
import torch, torchvision, transformers, peft, bitsandbytes as bnb, axolotl
from transformers import PreTrainedModel
try:
    import flash_attn
    fa = flash_attn.__version__
except Exception as e:
    fa = f"optional missing ({e})"
print(
    f"torch={torch.__version__} torchvision={torchvision.__version__} "
    f"transformers={transformers.__version__} peft={peft.__version__} "
    f"bnb={bnb.__version__} axolotl={getattr(axolotl,'__version__','?')} flash_attn={fa}"
)
PY

echo "[bootstrap] done at $(date -u +%FT%TZ)"
