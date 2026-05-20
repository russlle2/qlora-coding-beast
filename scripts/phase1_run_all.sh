#!/usr/bin/env bash
# phase1_run_all.sh — Run on the RunPod pod (web terminal or SSH).
# Chains: clone/update project → bootstrap → data → train → merge → GGUF → HF report.
#
# Required:
#   export HF_TOKEN=hf_...
# Optional:
#   REPO_URL  — git URL for this project (default: public clone; use PAT in URL for private)
#
# Usage:
#   cd /workspace && curl -fsSL ... | bash   # or copy this file onto the pod
#   bash scripts/phase1_run_all.sh

set -euo pipefail

# Auto-terminate pod on ANY exit (bootstrap fail, train fail, or success) to stop billing.
AUTO_TERMINATE_POD="${AUTO_TERMINATE_POD:-1}"
_SHUTDOWN_DONE=0
shutdown_pod() {
  local reason="${1:-unknown}"
  if [[ "${AUTO_TERMINATE_POD}" != "1" ]]; then
    return
  fi
  if [[ "${_SHUTDOWN_DONE}" == "1" ]]; then
    return
  fi
  _SHUTDOWN_DONE=1
  echo "[phase1] requesting pod shutdown (${reason})..."
  python scripts/runpod_shutdown.py --reason "$reason" || true
}
# EXIT fires on bootstrap failure, training failure, and success.
trap 'ec=$?; if [[ $ec -eq 0 ]]; then shutdown_pod "phase1_complete"; else shutdown_pod "phase1_failed_exit_${ec}"; fi' EXIT

WORKDIR="/workspace/qlora-coding-beast"
GGUF_REPO="${GGUF_REPO:-russlle2/qwen3-coder-30b-a3b-merged-gguf}"
REPO_URL="${REPO_URL:-https://github.com/russlle2/qlora-coding-beast.git}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[phase1] ERROR: export HF_TOKEN first (Hugging Face write token)."
  exit 1
fi

echo "[phase1] $(date -u +%FT%TZ) starting"

cd /workspace
if [[ ! -d "$WORKDIR/.git" ]]; then
  echo "[phase1] cloning $REPO_URL -> $WORKDIR"
  git clone "$REPO_URL" qlora-coding-beast
else
  echo "[phase1] updating existing repo"
  git -C "$WORKDIR" pull --ff-only || true
fi

cd "$WORKDIR"
export HF_TOKEN

echo "[phase1] bootstrap..."
bash scripts/runpod_bootstrap.sh

echo "[phase1] prepare uncensored dataset..."
python scripts/prepare_data.py --dataset uncensored --out /workspace/data/uncensored_chatml.jsonl

echo "[phase1] training (this takes hours; checkpoints push to HF hub_model_id)..."
axolotl train configs/adapter_uncensored.yaml 2>&1 | tee /workspace/outputs/train_phase1.log

echo "[phase1] merge adapter into BF16..."
python scripts/merge_adapters.py --mode phase1 --out /workspace/outputs/merged_phase1_bf16

echo "[phase1] GGUF quantize + push to $GGUF_REPO ..."
export MERGED_DIR=/workspace/outputs/merged_phase1_bf16
export GGUF_OUT=/workspace/outputs/gguf_phase1
export HUB_REPO="$GGUF_REPO"
bash scripts/convert_to_gguf.sh

echo "[phase1] upload training report to adapter repo..."
python scripts/push_phase1_report_to_hub.py

echo "[phase1] verifying HF checkpoints..."
PIPELINE_PHASE=phase1 python scripts/ensure_hub_checkpoint.py || true

echo "[phase1] $(date -u +%FT%TZ) done."
