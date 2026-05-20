#!/usr/bin/env bash
# Paste this ENTIRE block into a fresh RunPod pod web terminal (H200, /workspace).
# Prerequisite: export HF_TOKEN=hf_... first (do NOT commit the token).

set -euo pipefail
export REPO_URL="${REPO_URL:-https://github.com/russlle2/qlora-coding-beast.git}"
cd /workspace
if [[ ! -d qlora-coding-beast/.git ]]; then
  git clone "$REPO_URL" qlora-coding-beast
fi
cd qlora-coding-beast
git pull --ff-only || true
bash scripts/phase1_run_all.sh
