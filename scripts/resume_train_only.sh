#!/usr/bin/env bash
# Use when bootstrap already succeeded but train failed. Skips bootstrap + model download.
set -eo pipefail
: "${HF_TOKEN:?}"
export AUTO_TERMINATE_POD=0
export AUTO_TERMINATE_ON_FAILURE=0
cd /workspace/qlora-coding-beast
git pull --ff-only
exec bash scripts/train_phase1_now.sh
