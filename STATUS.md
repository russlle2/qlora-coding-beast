# qlora-coding-beast — compressed status (2026-05-20)

## Goal
Phase 1 QLoRA on `Qwen/Qwen3-Coder-30B-A3B-Instruct` → GGUF on HF → Legion smoke test.

## What failed (all before real training)
| # | Bug | Status |
|---|-----|--------|
| 1 | Wrong Docker tag `2.5.1-py3.11...` | Fixed → `2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| 2 | pip resolution-too-deep | Fixed → staged installs |
| 3 | `export HF_TOKEN="${HF_TOKEN}"` in dockerArgs → unbound variable | Fixed → don't re-export |
| 4 | `peft==0.18.2` doesn't exist | Fixed → `0.18.1` |
| 5 | `podTerminate { id }` GraphQL invalid | Fixed → no subfields + podStop fallback |
| 6 | Bootstrap **skipped** pip → torch/torchvision mismatch | Fixed → always `--force-reinstall` torch stack |
| 7 | Wrong `RUNPOD_POD_ID` → auto-stop failed, idle billing | Fix → set `RUNPOD_POD_ID` from console URL |

## HF repos (no training artifacts yet)
- `russlle2/qwen3-coder-30b-a3b-adapter-uncensored` — empty
- `russlle2/qwen3-coder-30b-a3b-merged-gguf` — empty

## One command on pod (after `git pull`)
```bash
export HF_TOKEN='...' RUNPOD_API_KEY='...' RUNPOD_POD_ID='id-from-url' AUTO_TERMINATE_POD=1
cd /workspace/qlora-coding-beast && git pull
nohup bash scripts/runpod_go.sh >> /workspace/run.log 2>&1 &
tail -f /workspace/run.log
```

## Or from Windows
```powershell
cd c:\Users\chris\.cursor\projects\qlora-coding-beast
python scripts/runpod_launch.py --phase 1
```
