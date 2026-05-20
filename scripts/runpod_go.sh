#!/usr/bin/env bash
# Single entrypoint: bootstrap (always) → phase1 train → merge → GGUF → stop pod.
set -eo pipefail
exec > /workspace/run.log 2>&1
echo "[go] $(date -u +%FT%TZ) start pid=$$"

: "${HF_TOKEN:?export HF_TOKEN first}"
export AUTO_TERMINATE_POD="${AUTO_TERMINATE_POD:-1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
git pull --ff-only || true

# Ensure shutdown targets THIS pod (RunPod sometimes sets wrong RUNPOD_POD_ID)
if [[ -z "${RUNPOD_POD_ID:-}" ]] && [[ -f /etc/runpod/pod_id ]]; then
  export RUNPOD_POD_ID="$(cat /etc/runpod/pod_id)"
fi

bash scripts/runpod_bootstrap.sh
bash scripts/phase1_train_only.sh
