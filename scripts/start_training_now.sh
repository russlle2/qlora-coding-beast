#!/usr/bin/env bash
# Run manually on the pod if autostart failed. Requires HF_TOKEN (+ RUNPOD_API_KEY for auto-stop).
# Optional: export RUNPOD_POD_ID from the RunPod console URL (fixes wrong auto-stop pod id).
set -eo pipefail
: "${HF_TOKEN:?export HF_TOKEN=hf_... first}"
export AUTO_TERMINATE_POD="${AUTO_TERMINATE_POD:-1}"
cd /workspace/qlora-coding-beast
git pull --ff-only || true
# If RUNPOD_POD_ID in env is wrong, pass the id from your browser URL:
# export RUNPOD_POD_ID='xxxxxxxx'
exec bash scripts/overnight_phase1.sh
