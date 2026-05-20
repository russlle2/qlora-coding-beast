#!/usr/bin/env bash
# Install axolotl CLI import chain deps (avoids one-by-one ModuleNotFoundError).
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TORCH_C="${ROOT}/constraints-torch-cu124.txt"

echo "[deps] remove torchao (incompatible with torch 2.5.1; breaks PreTrainedModel)..."
pip uninstall -y torchao 2>/dev/null || true

echo "[deps] pinning torch stack..."
bash "${ROOT}/scripts/fix_torch_stack.sh"

echo "[deps] axolotl train import packages..."
pip install -q --no-cache-dir -c "${TORCH_C}" -r "${ROOT}/requirements-axolotl-train.txt"

pip uninstall -y torchao 2>/dev/null || true

echo "[deps] verifying axolotl CLI import chain..."
python - <<'PY'
from transformers import PreTrainedModel
from axolotl.cli.main import main  # noqa: F401
from axolotl.loaders.model import ModelLoader  # noqa: F401
print("[deps] PreTrainedModel + axolotl CLI + ModelLoader OK")
PY
