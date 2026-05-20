#!/usr/bin/env bash
# Canonical RunPod container entry (called from runpod_launch.py dockerArgs).
# Keeps the pod alive on failure so you can SSH in and fix/resume.
set -eo pipefail

echo "[autostart] $(date -u +%FT%TZ) pid=$$"

# RunPod injects secrets; never re-export HF_TOKEN="${HF_TOKEN}" (breaks set -u).
export AUTO_TERMINATE_POD="${AUTO_TERMINATE_POD:-1}"
export AUTO_TERMINATE_ON_FAILURE="${AUTO_TERMINATE_ON_FAILURE:-0}"
export AXOLOTL_DO_NOT_TRACK=1
export DO_NOT_TRACK=1
export REPO_URL="${REPO_URL:-https://github.com/russlle2/qlora-coding-beast.git}"

if [[ -z "${RUNPOD_POD_ID:-}" ]] && [[ -f /etc/runpod/pod_id ]]; then
  export RUNPOD_POD_ID="$(tr -d '[:space:]' < /etc/runpod/pod_id)"
  echo "[autostart] RUNPOD_POD_ID=$RUNPOD_POD_ID (from /etc/runpod/pod_id)"
fi

cd /workspace
if [[ ! -d qlora-coding-beast/.git ]]; then
  git clone "$REPO_URL" qlora-coding-beast
fi
cd qlora-coding-beast
git pull --ff-only || true

# tmux survives web-terminal disconnects; log is always on disk.
if command -v tmux >/dev/null 2>&1; then
  tmux kill-session -t qlora 2>/dev/null || true
  tmux new-session -d -s qlora "bash scripts/runpod_go.sh"
  echo "[autostart] pipeline running in tmux session 'qlora' — attach: tmux attach -t qlora"
else
  nohup bash scripts/runpod_go.sh >> /workspace/run.log 2>&1 &
  echo "[autostart] pipeline running in background — log: /workspace/run.log"
fi

echo "[autostart] $(date -u +%FT%TZ) launcher done (container stays up)"
