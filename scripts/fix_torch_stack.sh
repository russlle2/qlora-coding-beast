#!/usr/bin/env bash
# Fix torch/torchvision mismatch (operator torchvision::nms does not exist).
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "[fix_torch] reinstalling pinned torch 2.5.1 + torchvision 0.20.1 (cu124)..."
pip install -q --no-cache-dir --force-reinstall \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
python - <<'PY'
import torch
import torchvision
from torchvision.ops import nms  # noqa: F401 — fails if torch/vision mismatch
print(f"[fix_torch] OK torch={torch.__version__} torchvision={torchvision.__version__}")
from transformers import PreTrainedModel
print("[fix_torch] transformers PreTrainedModel import OK")
import axolotl
print("[fix_torch] axolotl OK")
PY
