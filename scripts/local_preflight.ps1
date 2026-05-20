# local_preflight.ps1 — run on Windows before spending GPU money on RunPod.
# Usage: powershell -ExecutionPolicy Bypass -File scripts/local_preflight.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

Write-Host "=== qlora-coding-beast preflight ===" -ForegroundColor Cyan
Write-Host "Project: $Root"

# Load .env if present
$envFile = Join-Path $Root ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#=]+)=(.*)$') {
            $name = $Matches[1].Trim()
            $val = $Matches[2].Trim().Trim("'").Trim('"')
            [Environment]::SetEnvironmentVariable($name, $val, "Process")
        }
    }
}

$fail = 0

function Fail($msg) { Write-Host "FAIL: $msg" -ForegroundColor Red; $script:fail++ }
function Ok($msg)   { Write-Host "OK:   $msg" -ForegroundColor Green }
function Warn($msg)  { Write-Host "WARN: $msg" -ForegroundColor Yellow }

if (-not $env:HF_TOKEN) { Fail "HF_TOKEN not set (.env or environment)" } else { Ok "HF_TOKEN present" }

# Git / GitHub
try {
    $remote = git remote get-url origin 2>$null
    Ok "git remote: $remote"
} catch { Fail "no git remote origin" }

try {
    $vis = gh repo view russlle2/qlora-coding-beast --json visibility -q .visibility 2>$null
    if ($vis -eq "PUBLIC") { Ok "GitHub repo is public (RunPod can clone without PAT)" }
    else { Warn "GitHub repo is $vis - set REPO_URL with PAT or make public" }
} catch { Warn "gh not available or repo not found - install GitHub CLI or check repo manually" }

# HF checks via Python (single-quoted here-string avoids breaking on Python " quotes)
$py = @'
import os, sys, json, urllib.request

token = os.environ.get("HF_TOKEN")
if not token:
    sys.exit(1)

def get(path):
    req = urllib.request.Request(
        f"https://huggingface.co/{path}",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status

checks = {
    "whoami": "api/whoami-v2",
    "UncensoredOssbit": "api/datasets/ZombitX64/UncensoredOssbit",
    "xLAM": "api/datasets/Salesforce/xlam-function-calling-60k",
    "base Qwen3-Coder": "api/models/Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "adapter-uncensored": "api/models/russlle2/qwen3-coder-30b-a3b-adapter-uncensored",
    "adapter-tools": "api/models/russlle2/qwen3-coder-30b-a3b-adapter-tools",
    "adapter-coding": "api/models/russlle2/qwen3-coder-30b-a3b-adapter-coding",
    "merged-gguf": "api/models/russlle2/qwen3-coder-30b-a3b-merged-gguf",
}
optional = {
    "final-gguf (Phase 2)": "api/models/russlle2/qwen3-coder-30b-a3b-uncensored-tools-coding-gguf",
}
failed = []
for name, path in checks.items():
    try:
        get(path)
        print(f"OK\t{name}")
    except Exception as e:
        print(f"FAIL\t{name}\t{getattr(e,'code',e)}")
        failed.append(name)
for name, path in optional.items():
    try:
        get(path)
        print(f"OK\t{name}")
    except Exception:
        print(f"WARN\t{name}\tmissing (create before Phase 2 GGUF push)")

# Adapter training progress
try:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    files = api.list_repo_files("russlle2/qwen3-coder-30b-a3b-adapter-uncensored", repo_type="model")
    ckpt = [f for f in files if "checkpoint" in f or f.endswith(".safetensors")]
    if ckpt:
        print(f"OK\tPhase1 checkpoints on Hub ({len(ckpt)} weight-related files)")
    else:
        print("WARN\tadapter-uncensored repo has no checkpoints yet — training not started")
except Exception as e:
    print(f"WARN\tcould not list adapter repo: {e}")

sys.exit(1 if failed else 0)
'@

$py | python -
if ($LASTEXITCODE -ne 0) { $fail++ }

if (-not $env:RUNPOD_API_KEY) {
    Warn 'RUNPOD_API_KEY not set - use RunPod web UI OR: python scripts/runpod_launch.py --phase 1'
} else {
    Ok "RUNPOD_API_KEY present (can auto-launch pod)"
}

Write-Host ""
if ($fail -gt 0) {
    Write-Host "Preflight finished with $fail blocker(s). Fix above before launching GPU." -ForegroundColor Red
    exit 1
}
Write-Host "Preflight passed. Next:" -ForegroundColor Green
Write-Host "  Option A (automated): python scripts/runpod_launch.py --phase 1"
Write-Host '  Option B (manual):    RunPod console, H200 pod, paste scripts/runpod_oneliner.sh'
