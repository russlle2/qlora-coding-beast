#!/usr/bin/env bash
# Minimal axolotl imports for `axolotl train` (no torch upgrade).
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TORCH_C="${ROOT}/constraints-torch-cu124.txt"
echo "[deps] axolotl train import deps..."
pip install -q --no-cache-dir -c "${TORCH_C}" \
  "posthog==6.7.11" \
  "immutabledict==4.2.0" \
  "antlr4-python3-runtime==4.13.2" \
  "datasets==4.5.0" \
  "pillow>=11.0.0,<12.0.0"
python -c "import posthog; import axolotl; print('[deps] posthog + axolotl import OK')"
