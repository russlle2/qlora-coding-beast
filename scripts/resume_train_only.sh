#!/usr/bin/env bash
# Use when bootstrap already succeeded but train failed. Skips bootstrap + model download.
set -eo pipefail
: "${HF_TOKEN:?}"
export AUTO_TERMINATE_POD="${AUTO_TERMINATE_POD:-1}"
cd /workspace/qlora-coding-beast
git pull --ff-only
bash scripts/phase1_train_only.sh
