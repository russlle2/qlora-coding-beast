#!/usr/bin/env bash
# Fix torch/torchvision mismatch (operator torchvision::nms does not exist).
set -eo pipefail
echo "[fix_torch] reinstalling torch 2.5.1 + matching torchvision/torchaudio (cu124)..."
pip install -q --no-cache-dir --force-reinstall \
  torch==2.5.1 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124
python - <<'PY'
import torch, torchvision
print(f"[fix_torch] OK torch={torch.__version__} torchvision={torchvision.__version__}")
from transformers import PreTrainedModel
print("[fix_torch] transformers PreTrainedModel import OK")
import axolotl
print(f"[fix_torch] axolotl OK")
PY
