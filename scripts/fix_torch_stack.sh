#!/usr/bin/env bash
# Fix torch/torchvision mismatch only (use install_training_stack.sh for full setup).
set -eo pipefail
echo "[fix_torch] reinstalling pinned torch 2.5.1 + torchvision 0.20.1 (cu124)..."
pip install -q --no-cache-dir --force-reinstall \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
pip uninstall -y torchao 2>/dev/null || true
python - <<'PY'
import torch
import torchvision
from torchvision.ops import nms  # noqa: F401
from transformers import PreTrainedModel
from axolotl.cli.main import main  # noqa: F401
print(f"[fix_torch] OK torch={torch.__version__} torchvision={torchvision.__version__}")
print("[fix_torch] OK PreTrainedModel + axolotl.cli")
PY
