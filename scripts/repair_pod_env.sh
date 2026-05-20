#!/usr/bin/env bash
# Fix broken ~orch + install axolotl CLI on a RunPod pod. Run once before training.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SITE="/usr/local/lib/python3.11/dist-packages"

echo "[repair] $(date -u +%FT%TZ) cleaning corrupted torch/axolotl..."

# Broken partial uninstall leaves '~orch' dist-info — pip cannot resolve anything.
rm -rf "${SITE}"/~orch* "${SITE}"/torch* "${SITE}"/torchvision* "${SITE}"/torchaudio* 2>/dev/null || true
pip uninstall -y torch torchvision torchaudio axolotl axolotl-contribs-mit \
  axolotl-contribs-lgpl torchao 2>/dev/null || true
pip cache purge 2>/dev/null || true

echo "[repair] installing torch 2.5.1 stack..."
pip install --no-cache-dir --force-reinstall \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124

echo "[repair] ML core..."
pip install --no-cache-dir \
  "numpy>=2.2.6,<2.4" "pillow>=11.0.0,<12.0.0" "fsspec>=2023.1.0,<=2025.10.0" \
  "transformers==5.5.0" "accelerate==1.13.0" "peft==0.18.1" "bitsandbytes==0.49.1" \
  "datasets==4.5.0" "trl==0.29.0" "liger-kernel==0.7.0" "packaging==26.0" \
  "huggingface_hub>=1.1.7"

echo "[repair] axolotl 0.16.1 (--no-deps avoids torch 2.12 resolver fight)..."
pip install --no-cache-dir --no-build-isolation --no-deps "axolotl==0.16.1"
pip install --no-cache-dir --no-deps \
  "axolotl-contribs-mit==0.0.6" "axolotl-contribs-lgpl==0.0.7"

pip uninstall -y torchao 2>/dev/null || true

echo "[repair] axolotl import helpers..."
pip install --no-cache-dir \
  -r "${ROOT}/requirements-axolotl-runtime.txt" \
  -r "${ROOT}/requirements-axolotl-train.txt" || true
pip install --no-cache-dir --no-deps \
  "axolotl-contribs-mit==0.0.6" "axolotl-contribs-lgpl==0.0.7" 2>/dev/null || true
pip uninstall -y torchao 2>/dev/null || true

echo "[repair] re-pin torch..."
pip install --no-cache-dir --force-reinstall \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
pip uninstall -y torchao 2>/dev/null || true

echo "[repair] patch axolotl for torch 2.5.x (no torch.int4)..."
python "${ROOT}/scripts/patch_axolotl_torch25.py"

echo "[repair] verify..."
python -c "from axolotl.cli.main import main; print('[repair] OK axolotl.cli')"

echo "[repair] done — run: tmux new-session -d -s train 'bash scripts/train_phase1_now.sh'"
