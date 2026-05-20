#!/usr/bin/env bash
# Phase 1 after successful bootstrap (no re-bootstrap).
set -eo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/runpod_shutdown_helpers.sh
source "${ROOT}/scripts/runpod_shutdown_helpers.sh"
trap phase1_exit_trap EXIT

GGUF_REPO="${GGUF_REPO:-russlle2/qwen3-coder-30b-a3b-merged-gguf}"
export HF_TOKEN

bash scripts/fix_torch_stack.sh

if [[ ! -f /workspace/data/uncensored_chatml.jsonl ]]; then
  python scripts/prepare_data.py --dataset uncensored --out /workspace/data/uncensored_chatml.jsonl
else
  echo "[phase1] dataset already prepared, skipping"
fi
# Clear stale prepared cache if config changed
rm -rf /workspace/data/prepared_uncensored
axolotl train configs/adapter_uncensored.yaml 2>&1 | tee /workspace/outputs/train_phase1.log
python scripts/merge_adapters.py --mode phase1 --out /workspace/outputs/merged_phase1_bf16
export MERGED_DIR=/workspace/outputs/merged_phase1_bf16 GGUF_OUT=/workspace/outputs/gguf_phase1 HUB_REPO="$GGUF_REPO"
bash scripts/convert_to_gguf.sh
python scripts/push_phase1_report_to_hub.py
echo "[phase1] $(date -u +%FT%TZ) complete"
