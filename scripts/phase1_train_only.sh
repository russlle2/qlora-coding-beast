#!/usr/bin/env bash
# Phase 1 after successful bootstrap (no re-bootstrap).
set -eo pipefail

_SHUTDOWN_DONE=0
shutdown_pod() {
  [[ "${AUTO_TERMINATE_POD:-1}" != "1" ]] && return
  [[ "${_SHUTDOWN_DONE}" == "1" ]] && return
  _SHUTDOWN_DONE=1
  echo "[phase1] shutdown (${1:-done})..."
  python scripts/runpod_shutdown.py --reason "${1:-phase1_exit}" || true
}
trap 'ec=$?; shutdown_pod "phase1_failed_${ec}"' EXIT

GGUF_REPO="${GGUF_REPO:-russlle2/qwen3-coder-30b-a3b-merged-gguf}"
export HF_TOKEN

python scripts/prepare_data.py --dataset uncensored --out /workspace/data/uncensored_chatml.jsonl
axolotl train configs/adapter_uncensored.yaml 2>&1 | tee /workspace/outputs/train_phase1.log
python scripts/merge_adapters.py --mode phase1 --out /workspace/outputs/merged_phase1_bf16
export MERGED_DIR=/workspace/outputs/merged_phase1_bf16 GGUF_OUT=/workspace/outputs/gguf_phase1 HUB_REPO="$GGUF_REPO"
bash scripts/convert_to_gguf.sh
python scripts/push_phase1_report_to_hub.py
echo "[phase1] $(date -u +%FT%TZ) complete"
