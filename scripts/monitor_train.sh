#!/usr/bin/env bash
# One command on the pod to see if training is alive and at what step.
set -eo pipefail
echo "=== process ==="
ps aux | grep -E 'axolotl|train_phase1|prepare_data' | grep -v grep || echo "no axolotl/train processes"

echo "=== nvidia-smi (memory + util) ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

echo "=== train_phase1_now.log tail ==="
[[ -f /workspace/train_phase1_now.log ]] && tail -25 /workspace/train_phase1_now.log || echo "no train_phase1_now.log"

echo "=== outputs/train_phase1.log tail (axolotl output) ==="
[[ -f /workspace/outputs/train_phase1.log ]] && tail -30 /workspace/outputs/train_phase1.log || echo "no train_phase1.log yet"

echo "=== latest checkpoint ==="
ls -1dt /workspace/outputs/adapter_uncensored/checkpoint-* 2>/dev/null | head -3 || echo "no checkpoints yet"
