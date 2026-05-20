#!/usr/bin/env bash
# ONE command on the pod: fix deps → train. Never kills the pod on failure.
set -eo pipefail
: "${HF_TOKEN:?export HF_TOKEN first}"

export AUTO_TERMINATE_POD=0
export AUTO_TERMINATE_ON_FAILURE=0
export AUTO_TERMINATE_ON_SUCCESS="${AUTO_TERMINATE_ON_SUCCESS:-1}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
git pull --ff-only || true

exec > /workspace/train_phase1_now.log 2>&1
echo "[train_now] $(date -u +%FT%TZ) start"

source "${ROOT}/scripts/runpod_shutdown_helpers.sh"
resolve_runpod_pod_id || true
trap phase1_exit_trap EXIT

bash scripts/ensure_axolotl_train_deps.sh

if [[ ! -f /workspace/data/uncensored_chatml.jsonl ]]; then
  python scripts/prepare_data.py --dataset uncensored --out /workspace/data/uncensored_chatml.jsonl
else
  echo "[train_now] dataset ready — skip prepare"
fi

rm -rf /workspace/data/prepared_uncensored
mkdir -p /workspace/outputs

echo "[train_now] starting axolotl train..."
axolotl train configs/adapter_uncensored.yaml 2>&1 | tee /workspace/outputs/train_phase1.log

echo "[train_now] $(date -u +%FT%TZ) train finished OK"
