#!/usr/bin/env bash
# Overnight Phase 1: bootstrap → train → merge → GGUF → always terminate pod on exit.
#
# Required env (set before running):
#   HF_TOKEN
#   RUNPOD_API_KEY   (for auto-terminate; RunPod sets RUNPOD_POD_ID automatically)
#
# Usage (survives closing the browser tab — use tmux):
#   tmux new-session -d -s qlora 'export HF_TOKEN=... RUNPOD_API_KEY=...; bash /workspace/qlora-coding-beast/scripts/overnight_phase1.sh'
#
# Log: /workspace/overnight_phase1.log

set -euo pipefail
exec > /workspace/overnight_phase1.log 2>&1

echo "[overnight] $(date -u +%FT%TZ) starting"
export AUTO_TERMINATE_POD="${AUTO_TERMINATE_POD:-1}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[overnight] FATAL: HF_TOKEN not set"
  exit 1
fi

if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
  echo "[overnight] WARN: RUNPOD_API_KEY not set — pod will NOT auto-terminate on failure"
fi

REPO_URL="${REPO_URL:-https://github.com/russlle2/qlora-coding-beast.git}"
WORKDIR="/workspace/qlora-coding-beast"

cd /workspace
if [[ ! -d "$WORKDIR/.git" ]]; then
  git clone "$REPO_URL" qlora-coding-beast
fi
cd "$WORKDIR"
git pull --ff-only || true

# phase1_run_all.sh has EXIT trap → podTerminate on success OR failure
bash scripts/phase1_run_all.sh

echo "[overnight] $(date -u +%FT%TZ) finished (pod should be terminating)"
