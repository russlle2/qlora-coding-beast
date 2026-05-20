#!/usr/bin/env bash
# Resume Phase 1 after a bootstrap failure (skip bootstrap if axolotl already imports).
set -euo pipefail
cd /workspace/qlora-coding-beast
git pull --ff-only || true
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN}"

if ! python -c "import axolotl, transformers; print('ok', transformers.__version__)" 2>/dev/null; then
  echo "[resume] bootstrap required..."
  bash scripts/runpod_bootstrap.sh
else
  echo "[resume] axolotl already installed, skipping bootstrap"
fi

bash -c '
  set -euo pipefail
  python scripts/prepare_data.py --dataset uncensored --out /workspace/data/uncensored_chatml.jsonl
  axolotl train configs/adapter_uncensored.yaml 2>&1 | tee /workspace/outputs/train_phase1.log
  python scripts/merge_adapters.py --mode phase1 --out /workspace/outputs/merged_phase1_bf16
  export MERGED_DIR=/workspace/outputs/merged_phase1_bf16
  export GGUF_OUT=/workspace/outputs/gguf_phase1
  export HUB_REPO="${GGUF_REPO:-russlle2/qwen3-coder-30b-a3b-merged-gguf}"
  bash scripts/convert_to_gguf.sh
  python scripts/push_phase1_report_to_hub.py
  PIPELINE_PHASE=phase1 python scripts/ensure_hub_checkpoint.py || true
  python scripts/runpod_shutdown.py --reason phase1_complete
'
