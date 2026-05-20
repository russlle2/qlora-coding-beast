# qlora-coding-beast

Two-phase, separate-adapter QLoRA fine-tune of `Qwen/Qwen3-Coder-30B-A3B-Instruct` (MoE, 3.3B active, 256K context, Apache-2.0) into an uncensored tool-calling coding assistant that runs on a Lenovo Legion 7i (RTX 5090 24GB + 64GB RAM).

- Base: Qwen3-Coder-30B-A3B-Instruct
- Training hardware: 1x H200 141GB RunPod Community Cloud
- Framework: Axolotl + bitsandbytes NF4 QLoRA + FlashAttention-3 + Liger Kernel
- Budget: ~$33-40, well under $75
- Total wall clock: ~11 hours across two phases

Full plan document: `c:\Users\chris\.cursor\plans\qlora-uncensored-coding-beast_4189fd36.plan.md`

## Architecture

Three independent LoRA adapters are trained separately (no cross-contamination), then weighted-merged into the base:

- `adapter-uncensored` — `ZombitX64/UncensoredOssbit` (68K rows, 1 epoch) — shifts refusal behavior
- `adapter-tools` — `Salesforce/xlam-function-calling-60k` + `NousResearch/hermes-function-calling-v1` (~72K rows, 1 epoch) — teaches precise JSON tool calling
- `adapter-coding` — `nvidia/OpenCodeInstruct` top-30K execution-verified (1 epoch) — preserves and uplifts coding ability

Final merge: `add_weighted_adapter(uncensored=0.7, tools=1.0, coding=1.0)` via PEFT.

## Phase 0: local setup (one-time)

### 0.1 Accept HF ToS for gated datasets

Log into huggingface.co as `russlle2` and click "Accept" on:

- [ZombitX64/UncensoredOssbit](https://huggingface.co/datasets/ZombitX64/UncensoredOssbit)
- [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)

### 0.2 Create HF token with write scope

<https://huggingface.co/settings/tokens> → New token → Fine-grained, check:

- Read access to public repos
- Read access to gated repos you've been granted
- Write access to your own repos + create new repos

Save as `HF_TOKEN=hf_...`. Keep it secret.

### 0.3 Pre-create 4 private HF repos

The training configs + merge script expect these to exist:

- `russlle2/qwen3-coder-30b-a3b-adapter-uncensored` (model repo, private)
- `russlle2/qwen3-coder-30b-a3b-adapter-tools` (model repo, private)
- `russlle2/qwen3-coder-30b-a3b-adapter-coding` (model repo, private)
- `russlle2/qwen3-coder-30b-a3b-uncensored-tools-coding-gguf` (model repo, private)

Create them via the HF web UI, or Axolotl will auto-create the first three when pushing via `hub_strategy: checkpoint`.

### 0.4 Push this project to a **public** GitHub repo

A **public** repo lets RunPod clone without a GitHub token (`git clone https://github.com/russlle2/qlora-coding-beast.git`).

```bash
cd c:\Users\chris\.cursor\projects\qlora-coding-beast
gh repo create russlle2/qlora-coding-beast --public --source=. --remote=origin --push
```

If the repo already exists and is private: GitHub → **Settings** → **General** → **Danger zone** → **Change repository visibility** → **Public**.

Or use whatever git host you prefer.

---

## Phase 1: adapter-uncensored (~5h wall clock, ~$14)

### 1.1 Spin up RunPod pod

- GPU: **1x H200 141GB SXM**, Community Cloud, on-demand
- Image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (or `winglian/axolotl-cloud:main-latest`)
- Container disk: **150 GB**
- Volume mount: none (Community Cloud doesn't support it; we push checkpoints to HF Hub)
- Env vars to set in the pod config:
  - `HF_TOKEN=hf_...`

**Where outputs go (Hugging Face repos)**

| Artifact | HF repo |
|----------|---------|
| Training checkpoints + LoRA adapter | `russlle2/qwen3-coder-30b-a3b-adapter-uncensored` (private; `hub_strategy: checkpoint`) |
| Phase 1 report (`PHASE1_REPORT.md`, `phase1_summary.json`) | Same adapter repo (uploaded by `scripts/push_phase1_report_to_hub.py`) |
| Merged BF16 + GGUF quants | `russlle2/qwen3-coder-30b-a3b-merged-gguf` |

Push this project to a **GitHub** repo (or copy it to `/workspace/qlora-coding-beast` on the pod). Then in the pod terminal:

```bash
cd /workspace
export HF_TOKEN=hf_...   # Hugging Face write token
export REPO_URL=https://github.com/USER/qlora-coding-beast.git   # only if the code is not already under /workspace/qlora-coding-beast
bash qlora-coding-beast/scripts/phase1_run_all.sh
```

`scripts/phase1_run_all.sh` runs bootstrap → dataset → train → merge → `convert_to_gguf.sh` → Hub report upload. Training alone takes several hours; keep the pod running.

### 1.2 Bootstrap the pod (~8-12 min first time, includes base model prefetch)

SSH into the pod, then:

```bash
cd /workspace
git clone https://github.com/russlle2/qlora-coding-beast.git
cd qlora-coding-beast
export HF_TOKEN=hf_...
bash scripts/runpod_bootstrap.sh
```

This installs pip deps, builds flash-attn, logs into HF, and pre-downloads the base model weights into the HF cache.

### 1.3 Prepare the uncensored dataset (~5-10 min)

```bash
python scripts/prepare_data.py --dataset uncensored --out /workspace/data/uncensored_chatml.jsonl
```

Expected output: ~68,000 rows written.

### 1.4 Train adapter-uncensored (~4.2h)

```bash
tmux new -s train
axolotl train configs/adapter_uncensored.yaml
# Ctrl+B then D to detach tmux
```

Monitor in another shell:

```bash
tmux attach -t train          # or:
tail -f /workspace/outputs/adapter_uncensored/log.jsonl
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv --loop=5
```

**What to check in the first 30 min**:

- Training loss drops smoothly from ~3-4 into the 1-2 range
- `grad_norm` stays below ~5.0
- Steady-state tokens/sec (check Axolotl log): expect ~8-10K tok/s for this config
- Checkpoints appear in `/workspace/outputs/adapter_uncensored/checkpoint-*` and auto-push to `russlle2/qwen3-coder-30b-a3b-adapter-uncensored`

**Abort condition**: if extrapolated wall clock goes over 6 hours at step 1000, kill and re-run with a smaller dataset subset (edit `prepare_data.py` to cap at 30K rows).

### 1.5 Merge adapter → convert → GGUF → push (~45 min)

```bash
python scripts/merge_adapters.py \
    --mode phase1 \
    --out /workspace/outputs/merged_phase1_bf16

export MERGED_DIR=/workspace/outputs/merged_phase1_bf16
export GGUF_OUT=/workspace/outputs/gguf_phase1
export HUB_REPO=russlle2/qwen3-coder-30b-a3b-merged-gguf
bash scripts/convert_to_gguf.sh
```

This produces Q4_K_M / Q5_K_M / Q6_K / Q8_0 GGUFs and pushes them to HF (`russlle2/qwen3-coder-30b-a3b-merged-gguf`).

### 1.6 TEAR DOWN THE POD

From the RunPod console: Stop → Terminate. Do not leave idle.

Running tab so far: **~$14**.

### 1.7 Pull to Legion, smoke test (on your laptop, ~15 min)

On Windows/Legion:

```powershell
# Install Ollama for Windows from https://ollama.com/download
# Then:
ollama pull hf.co/russlle2/qwen3-coder-30b-a3b-merged-gguf:Q4_K_M

# Or if you want to download manually and import:
# huggingface-cli download russlle2/qwen3-coder-30b-a3b-merged-gguf qwen3-coder-30b-a3b-uncensored-tools-coding-Q4_K_M.gguf --local-dir .
# Create a Modelfile with the downloaded path, then `ollama create qwen3-coder-uncensored -f Modelfile`

# Quick interactive check
ollama run hf.co/russlle2/qwen3-coder-30b-a3b-merged-gguf:Q4_K_M

# Automated gate check
pip install ollama
python scripts/phase1_smoke.py --model "hf.co/russlle2/qwen3-coder-30b-a3b-merged-gguf:Q4_K_M"
```

Expected gates (see `phase1_smoke.py`):

- Coherence: generates clean English
- Throughput: >= 25 tok/s at 8K context
- Long-context: retrieves a needle from a ~16K-token haystack
- Uncensored engagement: writes detailed offensive-security content on 3 probes
- Safety hard stops: still refuses bioweapon synthesis (sanity)

**If Phase 1 smoke PASSES, proceed to Phase 2.**
**If Phase 1 smoke FAILS, DO NOT SPEND PHASE 2 MONEY — debug first.**

---

## Phase 2: tools + coding + merge + final GGUF (~6h wall clock, ~$19)

### 2.1 Spin up a fresh H200 pod

Same as 1.1.

### 2.2 Bootstrap + prepare tools data

```bash
cd /workspace
git clone https://github.com/russlle2/qlora-coding-beast.git
cd qlora-coding-beast
export HF_TOKEN=hf_...
bash scripts/runpod_bootstrap.sh

python scripts/prepare_data.py --dataset tools --out /workspace/data/tools_chatml.jsonl
```

### 2.3 Train adapter-tools (~2-2.5h)

```bash
axolotl train configs/adapter_tools.yaml
```

### 2.4 Prepare coding data + train adapter-coding (~3h)

```bash
python scripts/prepare_data.py --dataset coding --out /workspace/data/coding_chatml.jsonl --coding-top-k 30000
axolotl train configs/adapter_coding.yaml
```

### 2.5 Weighted 3-adapter merge (~15 min)

```bash
python scripts/merge_adapters.py \
    --mode phase2 \
    --weights 0.7 1.0 1.0 \
    --combination-type linear \
    --out /workspace/outputs/merged_final_bf16
```

### 2.6 Eval gate (optional but recommended, ~30-60 min)

```bash
python scripts/eval_harness.py \
    --model-path /workspace/outputs/merged_final_bf16 \
    --out /workspace/outputs/eval_report.json
```

Check `eval_report.json` → `gates`. If any gate fails, try different merge weights:

```bash
# Example: re-run merge with uncensored weighted lower
python scripts/merge_adapters.py \
    --mode phase2 --weights 0.5 1.0 1.0 \
    --combination-type ties --density 0.5 \
    --out /workspace/outputs/merged_ties_bf16
python scripts/eval_harness.py --model-path /workspace/outputs/merged_ties_bf16 \
    --out /workspace/outputs/eval_ties.json
```

Re-merging is cheap (~15 min, ~$0.70). Iterate 2-3 times if needed.

### 2.7 Final GGUF + push

```bash
export MERGED_DIR=/workspace/outputs/merged_final_bf16
export GGUF_OUT=/workspace/outputs/gguf_final
export HUB_REPO=russlle2/qwen3-coder-30b-a3b-uncensored-tools-coding-gguf
bash scripts/convert_to_gguf.sh
```

### 2.8 TEAR DOWN THE POD

Running tab total: **~$33-40**.

### 2.9 Deploy to Legion

```powershell
ollama pull hf.co/russlle2/qwen3-coder-30b-a3b-uncensored-tools-coding-gguf:Q4_K_M
ollama run hf.co/russlle2/qwen3-coder-30b-a3b-uncensored-tools-coding-gguf:Q4_K_M
```

Expected: 35-50 tok/s at 32K context on RTX 5090 24GB + 64GB RAM.

---

## GGUF quant guide for your Legion

- `Q4_K_M` (~18 GB): daily driver, all-on-GPU, ~35-50 tok/s.
- `Q5_K_M` (~21 GB): higher quality, all-on-GPU, tighter context (~12-16K before KV spill), ~28-40 tok/s.
- `Q6_K` (~25 GB): best quality that still runs well. Needs `--override-tensor "([0-9]+).ffn_.*_exps=CPU"` in llama.cpp to offload MoE experts to RAM; ~20-30 tok/s.
- `Q8_0` (~32 GB): archival quality. Full expert offload to RAM; ~15-25 tok/s.

For Q6_K / Q8_0 expert offload in llama.cpp directly (not Ollama):

```powershell
llama-cli -m qwen3-coder-30b-a3b-uncensored-tools-coding-Q6_K.gguf `
  --override-tensor "([0-9]+).ffn_.*_exps=CPU" `
  -ngl 999 -c 32768 --flash-attn `
  -p "Write me a complete Next.js 15 app that..."
```

---

## Deferred (Phase 3+, separate plan and budget)

- Security-domain QA synthesis from `w8ay/security-tools-datasets`
- MCP tool-call trace synthesis from `smolagents/tool-scraping`
- Pokemon card forgery VLM fine-tune on `Qwen/Qwen3.5-27B` base with `TheFusion21/PokemonCards` + `Usamasaddique/Forgery_detection`

---

## Hard-abort rules to protect budget

1. Phase 1 training extrapolated >6h at step 1000 → kill, reduce dataset.
2. Phase 1 smoke test fails any gate → do not start Phase 2.
3. Any pod idle >15 min with no training running → tear down.
4. Running total >$55 → stop and reassess.

## Key files

- [configs/adapter_uncensored.yaml](configs/adapter_uncensored.yaml)
- [configs/adapter_tools.yaml](configs/adapter_tools.yaml)
- [configs/adapter_coding.yaml](configs/adapter_coding.yaml)
- [scripts/runpod_bootstrap.sh](scripts/runpod_bootstrap.sh)
- [scripts/prepare_data.py](scripts/prepare_data.py)
- [scripts/merge_adapters.py](scripts/merge_adapters.py)
- [scripts/convert_to_gguf.sh](scripts/convert_to_gguf.sh)
- [scripts/eval_harness.py](scripts/eval_harness.py)
- [scripts/phase1_smoke.py](scripts/phase1_smoke.py)
- [requirements.txt](requirements.txt)
