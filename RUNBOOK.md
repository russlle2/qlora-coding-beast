# Runbook — finish the Uncensored Coding Beast

**Status (auto-checked):** Phase 0 mostly done. **Training has not started** — HF adapter repos only contain empty READMEs.

**Project root:** `c:\Users\chris\.cursor\projects\qlora-coding-beast`  
**Public repo:** https://github.com/russlle2/qlora-coding-beast

## Why RunPod (not Colab / NotebookLM)

| Platform | Verdict |
|----------|---------|
| **RunPod H200 141GB** | **Use this.** Fits 32K QLoRA on Qwen3-Coder-30B-A3B MoE with headroom; ~$14 Phase 1, ~$20 Phase 2. |
| Google Colab Pro+ (H100 80GB) | Possible but tight at 32K; session timeouts; no reliable multi-hour unattended runs. |
| NotebookLM | Not for GPU training — research/notes only. |
| Kaggle | Similar Colab limits; not worth switching for this plan. |

Your old Colab notebook (`01_qlora_finetuning.ipynb`) targets **Llama-3.1-70B + FSDP** — a different stack. This project uses **Qwen3-Coder-30B-A3B MoE + Axolotl single-GPU**.

---

## One-time: secrets (you)

1. **HF token** — already in `.env` (working). If it was ever pasted in chat/Documents, rotate at https://huggingface.co/settings/tokens
2. **RunPod API key** (optional, for auto-launch) — https://www.runpod.io/console/user/settings → add to `.env`:
   ```
   RUNPOD_API_KEY=rpa_...
   ```
3. **Gated datasets** — already accepted (UncensoredOssbit + xLAM verified).

---

## Step 1 — Local preflight (Cursor / PowerShell)

```powershell
cd "c:\Users\chris\.cursor\projects\qlora-coding-beast"
powershell -ExecutionPolicy Bypass -File scripts\local_preflight.ps1
```

---

## Step 2 — Launch Phase 1 (~5h GPU, ~$14)

### Option A — Fully automated (needs `RUNPOD_API_KEY` in `.env`)

```powershell
cd "c:\Users\chris\.cursor\projects\qlora-coding-beast"
python scripts/runpod_launch.py --phase 1
```

Pod runs `phase1_run_all.sh` on boot. Log: `/workspace/runpod_autostart.log`

### Option B — Manual (no API key)

1. RunPod → **Deploy** → **Community Cloud** → **1× H200 141GB** (or H100 80GB fallback)
2. Image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
3. Container disk: **150 GB**
4. Env: `HF_TOKEN` = your token
5. Open **Web Terminal**, paste:

```bash
export HF_TOKEN='YOUR_TOKEN_HERE'
curl -fsSL https://raw.githubusercontent.com/russlle2/qlora-coding-beast/main/scripts/runpod_oneliner.sh | bash
```

(Or clone and run — see `scripts/runpod_oneliner.sh`.)

### What Phase 1 does automatically

1. `runpod_bootstrap.sh` — deps, flash-attn, base model cache (~12 min)
2. `prepare_data.py --dataset uncensored` — 68K rows
3. `axolotl train configs/adapter_uncensored.yaml` — ~4h
4. `merge_adapters.py --mode phase1`
5. `convert_to_gguf.sh` → `russlle2/qwen3-coder-30b-a3b-merged-gguf`
6. Training report → adapter repo

**Then: STOP + TERMINATE the pod.**

---

## Step 3 — Laptop smoke test (Legion 7i)

```powershell
pip install ollama
ollama pull hf.co/russlle2/qwen3-coder-30b-a3b-merged-gguf:Q4_K_M
python scripts/phase1_smoke.py --model "hf.co/russlle2/qwen3-coder-30b-a3b-merged-gguf:Q4_K_M"
```

**Do not start Phase 2 if smoke fails.**

---

## Step 4 — Phase 2 (~6h GPU, ~$20)

```powershell
python scripts/runpod_launch.py --phase 2
```

Or manual pod +:

```bash
export HF_TOKEN='...'
cd /workspace && git clone https://github.com/russlle2/qlora-coding-beast.git && cd qlora-coding-beast
bash scripts/phase2_run_all.sh
```

Creates final GGUF at `russlle2/qwen3-coder-30b-a3b-uncensored-tools-coding-gguf` (repo auto-created on push).

---

## Budget guardrails

- Terminate pod within 15 min of pipeline finishing
- Phase 1 training >6h at step ~1000 → kill, reduce dataset
- Total spend >$55 → stop and debug

---

## What Cursor cannot do for you

- Hold a GPU for 4–6 hours inside the IDE (no local H200)
- Click RunPod “Accept” dialogs in your browser (you deploy once)
- Run Ollama smoke on your 5090 without you pulling the GGUF locally

Everything else is scripted.
