#!/usr/bin/env bash
# Install axolotl CLI import chain deps (avoids one-by-one ModuleNotFoundError).
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TORCH_C="${ROOT}/constraints-torch-cu124.txt"

echo "[deps] pinning torch stack before axolotl extras..."
bash "${ROOT}/scripts/fix_torch_stack.sh"

echo "[deps] axolotl train import packages..."
pip install -q --no-cache-dir -c "${TORCH_C}" -r "${ROOT}/requirements-axolotl-train.txt"

echo "[deps] verifying axolotl CLI import chain..."
python - <<'PY'
from axolotl.cli.main import main  # noqa: F401
from axolotl.loaders.model import ModelLoader  # noqa: F401
print("[deps] axolotl CLI + ModelLoader import OK")
PY
