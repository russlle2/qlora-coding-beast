#!/usr/bin/env bash
# convert_to_gguf.sh
# Convert a merged BF16 safetensors model to GGUF at multiple quant levels,
# generate an imatrix from a small calibration dataset, and push to HF Hub.
#
# Usage:
#   MERGED_DIR=/workspace/outputs/merged_final_bf16 \
#   GGUF_OUT=/workspace/outputs/gguf \
#   HUB_REPO=russlle2/qwen3-coder-30b-a3b-uncensored-tools-coding-gguf \
#   bash scripts/convert_to_gguf.sh
#
# Requires: HF_TOKEN env var. Runs on the same H200 pod.

set -euo pipefail

: "${MERGED_DIR:?MERGED_DIR env var is required (path to merged BF16 HF dir)}"
: "${GGUF_OUT:?GGUF_OUT env var is required (output dir for .gguf files)}"
: "${HUB_REPO:?HUB_REPO env var is required (e.g. russlle2/qwen3-coder-30b-a3b-gguf)}"
: "${HF_TOKEN:?HF_TOKEN env var is required}"

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/workspace/llama.cpp}"
MODEL_NAME="qwen3-coder-30b-a3b-uncensored-tools-coding"

echo "[gguf] merged dir: $MERGED_DIR"
echo "[gguf] output dir: $GGUF_OUT"
echo "[gguf] hub repo:   $HUB_REPO"

mkdir -p "$GGUF_OUT"

# -------- 1. clone + build llama.cpp (CUDA, for imatrix + quant) --------
if [[ ! -d "$LLAMA_CPP_DIR" ]]; then
  echo "[gguf] cloning llama.cpp..."
  git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_CPP_DIR"
fi

pushd "$LLAMA_CPP_DIR" >/dev/null
git pull --ff-only || true
echo "[gguf] building llama.cpp with CUDA..."
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF >/dev/null
cmake --build build -j --config Release --target llama-quantize llama-imatrix llama-cli >/dev/null
popd >/dev/null

# Also need python deps for the converter
pip install -q gguf sentencepiece protobuf

# -------- 2. convert BF16 safetensors -> GGUF F16 (source for all quants) --------
F16_GGUF="$GGUF_OUT/${MODEL_NAME}-f16.gguf"
if [[ ! -f "$F16_GGUF" ]]; then
  echo "[gguf] convert BF16 safetensors -> F16 GGUF..."
  python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MERGED_DIR" \
    --outfile "$F16_GGUF" \
    --outtype f16
else
  echo "[gguf] F16 GGUF already exists, skipping conversion"
fi

# -------- 3. build imatrix from a small calibration corpus --------
IMATRIX_FILE="$GGUF_OUT/${MODEL_NAME}.imatrix"
CALIB_TXT="$GGUF_OUT/calibration.txt"
if [[ ! -f "$IMATRIX_FILE" ]]; then
  echo "[gguf] fetching imatrix calibration corpus..."
  # Small mixed English/code calibration set (widely used for llama.cpp imatrix)
  if [[ ! -f "$CALIB_TXT" ]]; then
    curl -fsSL \
      "https://gist.githubusercontent.com/bartowski1182/eb213dccb3571f863da82e99418f81e8/raw/b2869d80f5c16fd7082594248e80144677736635/calibration_datav3.txt" \
      -o "$CALIB_TXT"
  fi
  echo "[gguf] building imatrix (takes ~10-20 min on H200)..."
  "$LLAMA_CPP_DIR/build/bin/llama-imatrix" \
    -m "$F16_GGUF" \
    -f "$CALIB_TXT" \
    -o "$IMATRIX_FILE" \
    --n-gpu-layers 999 \
    --ctx-size 4096 \
    --chunks 100
else
  echo "[gguf] imatrix already exists, skipping"
fi

# -------- 4. quantize to 4 target levels --------
quantize() {
  local qtype="$1"
  local out="$GGUF_OUT/${MODEL_NAME}-${qtype}.gguf"
  if [[ -f "$out" ]]; then
    echo "[gguf] $qtype already exists, skipping"
    return
  fi
  echo "[gguf] quantize -> $qtype"
  "$LLAMA_CPP_DIR/build/bin/llama-quantize" \
    --imatrix "$IMATRIX_FILE" \
    "$F16_GGUF" \
    "$out" \
    "$qtype"
}

quantize Q4_K_M
quantize Q5_K_M
quantize Q6_K
quantize Q8_0

echo "[gguf] all quants complete:"
ls -lh "$GGUF_OUT"/*.gguf

# -------- 5. push all quants to HF Hub --------
echo "[gguf] pushing to $HUB_REPO..."
python - <<PY
import os
from huggingface_hub import HfApi, create_repo

repo_id = "$HUB_REPO"
api = HfApi(token=os.environ["HF_TOKEN"])
create_repo(repo_id, private=True, exist_ok=True, repo_type="model", token=os.environ["HF_TOKEN"])

# Upload each GGUF as a separate file
for fname in os.listdir("$GGUF_OUT"):
    if fname.endswith(".gguf"):
        print(f"uploading {fname}...")
        api.upload_file(
            path_or_fileobj=f"$GGUF_OUT/{fname}",
            path_in_repo=fname,
            repo_id=repo_id,
            repo_type="model",
        )
print("all uploads complete")
PY

echo "[gguf] DONE. Models at https://huggingface.co/$HUB_REPO"
