#!/usr/bin/env bash
# Single entrypoint: bootstrap (always) → phase1 train → merge → GGUF → stop pod on success only.
set -eo pipefail
exec > /workspace/run.log 2>&1
echo "[go] $(date -u +%FT%TZ) start pid=$$"

: "${HF_TOKEN:?export HF_TOKEN first}"
export AUTO_TERMINATE_POD="${AUTO_TERMINATE_POD:-0}"
export AUTO_TERMINATE_ON_FAILURE="${AUTO_TERMINATE_ON_FAILURE:-0}"
export AUTO_TERMINATE_ON_SUCCESS="${AUTO_TERMINATE_ON_SUCCESS:-1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
git pull --ff-only || true

# shellcheck source=scripts/runpod_shutdown_helpers.sh
source "${ROOT}/scripts/runpod_shutdown_helpers.sh"
resolve_runpod_pod_id || echo "[go] WARN: RUNPOD_POD_ID unknown — auto-stop may fail until set from console URL"

bash scripts/runpod_bootstrap.sh
bash scripts/phase1_train_only.sh
