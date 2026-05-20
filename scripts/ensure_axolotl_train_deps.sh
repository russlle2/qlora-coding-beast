#!/usr/bin/env bash
# Wrapper — full install is in install_training_stack.sh
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
bash "${ROOT}/scripts/repair_pod_env.sh"
