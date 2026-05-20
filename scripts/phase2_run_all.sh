#!/usr/bin/env bash
# phase2_run_all.sh — Run on RunPod after Phase 1 smoke test passes.
# Chains: bootstrap → tools data/train → coding data/train → merge → eval → GGUF → Hub.

set -euo pipefail

AUTO_TERMINATE_POD="${AUTO_TERMINATE_POD:-1}"
shutdown_pod() {
  local reason="${1:-unknown}"
  if [[ "${AUTO_TERMINATE_POD}" == "1" ]]; then
    echo "[phase2] requesting pod shutdown (${reason})..."
    python scripts/runpod_shutdown.py --reason "$reason" || true
  fi
}
on_error() {
  echo "[phase2] ERROR at line $1 (exit $2)"
  shutdown_pod "phase2_failed_line_$1"
}
trap 'on_error $LINENO $?' ERR

WORKDIR="/workspace/qlora-coding-beast"
GGUF_REPO="${GGUF_REPO:-russlle2/qwen3-coder-30b-a3b-uncensored-tools-coding-gguf}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[phase2] ERROR: export HF_TOKEN first."
  exit 1
fi

echo "[phase2] $(date -u +%FT%TZ) starting"
cd /workspace

if [[ ! -d "$WORKDIR/.git" ]]; then
  REPO_URL="${REPO_URL:-https://github.com/russlle2/qlora-coding-beast.git}"
  git clone "$REPO_URL" qlora-coding-beast
else
  git -C "$WORKDIR" pull --ff-only || true
fi

cd "$WORKDIR"
export HF_TOKEN

bash scripts/runpod_bootstrap.sh

python scripts/prepare_data.py --dataset tools --out /workspace/data/tools_chatml.jsonl
axolotl train configs/adapter_tools.yaml 2>&1 | tee /workspace/outputs/train_phase2_tools.log

python scripts/prepare_data.py --dataset coding --out /workspace/data/coding_chatml.jsonl --coding-top-k 30000
axolotl train configs/adapter_coding.yaml 2>&1 | tee /workspace/outputs/train_phase2_coding.log

python scripts/merge_adapters.py \
  --mode phase2 \
  --weights 0.7 1.0 1.0 \
  --combination-type linear \
  --out /workspace/outputs/merged_final_bf16

python scripts/eval_harness.py \
  --model-path /workspace/outputs/merged_final_bf16 \
  --out /workspace/outputs/eval_report.json || true

export MERGED_DIR=/workspace/outputs/merged_final_bf16
export GGUF_OUT=/workspace/outputs/gguf_final
export HUB_REPO="$GGUF_REPO"
bash scripts/convert_to_gguf.sh

PIPELINE_PHASE=phase2_coding python scripts/ensure_hub_checkpoint.py || true
echo "[phase2] $(date -u +%FT%TZ) done."
shutdown_pod "phase2_complete"
