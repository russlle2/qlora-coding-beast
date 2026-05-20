#!/usr/bin/env bash
# Full training stack for RunPod — run once before train_phase1_now.sh
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIP_C="${ROOT}/constraints-pip.txt"
TORCH_C="${ROOT}/constraints-torch-cu124.txt"

echo "[stack] $(date -u +%FT%TZ) installing training stack..."

pip uninstall -y torchao 2>/dev/null || true

echo "[stack] torch 2.5.1 + torchvision 0.20.1..."
pip install -q --no-cache-dir --force-reinstall \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124

echo "[stack] ML core..."
pip install -q --no-cache-dir -c "${PIP_C}" \
  "transformers==5.5.0" \
  "accelerate==1.13.0" \
  "peft==0.18.1" \
  "bitsandbytes==0.49.1" \
  "datasets==4.5.0" \
  "trl==0.29.0" \
  "liger-kernel==0.7.0" \
  "packaging==26.0" \
  "huggingface_hub>=1.1.7" \
  "pillow>=11.0.0,<12.0.0" \
  "fsspec>=2023.1.0,<=2025.10.0" \
  "numpy>=2.2.6,<2.4"

echo "[stack] axolotl 0.16.1 (--no-deps; avoids torch 2.12 resolver)..."
pip install -q --no-cache-dir --no-build-isolation --no-deps "axolotl==0.16.1"
pip install -q --no-cache-dir --no-deps \
  "axolotl-contribs-mit==0.0.6" "axolotl-contribs-lgpl==0.0.7" || true

pip uninstall -y torchao 2>/dev/null || true

echo "[stack] axolotl runtime + train imports..."
pip install -q --no-cache-dir -c "${PIP_C}" -c "${TORCH_C}" \
  -r "${ROOT}/requirements-axolotl-runtime.txt"
pip install -q --no-cache-dir -c "${PIP_C}" -c "${TORCH_C}" \
  -r "${ROOT}/requirements-axolotl-train.txt"
pip install -q --no-cache-dir --no-deps \
  "axolotl-contribs-lgpl==0.0.7" "axolotl-contribs-mit==0.0.6" || true

pip uninstall -y torchao 2>/dev/null || true

echo "[stack] re-pin torch (post pip)..."
pip install -q --no-cache-dir --force-reinstall \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
pip uninstall -y torchao 2>/dev/null || true

echo "[stack] flash-attn (optional)..."
pip install -q --no-cache-dir "flash-attn>=2.7.0,<3" --no-build-isolation || \
  echo "[stack] WARN: flash-attn build failed"

echo "[stack] verify..."
python - <<'PY'
import torch, torchvision
from torchvision.ops import nms  # noqa: F401
from transformers import PreTrainedModel
from axolotl.cli.main import main  # noqa: F401
from axolotl.loaders.model import ModelLoader  # noqa: F401
print(f"[stack] OK torch={torch.__version__} torchvision={torchvision.__version__}")
print("[stack] OK axolotl CLI + ModelLoader")
PY

echo "[stack] $(date -u +%FT%TZ) done"
