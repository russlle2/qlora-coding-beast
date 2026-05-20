#!/usr/bin/env bash
# Make axolotl 0.16.1 importable on torch 2.5.1 (cu124) — runs every train.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
bash "${ROOT}/scripts/repair_pod_env.sh"
python "${ROOT}/scripts/patch_axolotl_torch25.py"
