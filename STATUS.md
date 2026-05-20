# qlora-coding-beast — RunPod status

## Active pod (use this one)

| | |
|--|--|
| **Pod ID** | `k9ckjm5ybx0bhc` |
| **Console** | https://console.runpod.io/pods?id=k9ckjm5ybx0bhc |

`sa30ji6hq2b3ky` is **terminated** — do not use.

## On the pod terminal (after Connect)

Autostart runs bootstrap (~20–40 min first time). Watch:

```bash
tail -f /workspace/runpod_autostart.log
tail -f /workspace/run.log
```

When bootstrap finishes (or if it errors), run training:

```bash
cd /workspace/qlora-coding-beast
git pull
bash scripts/ensure_axolotl_train_deps.sh
tmux new-session -d -s train 'bash scripts/train_phase1_now.sh'
tail -f /workspace/train_phase1_now.log
```

Training = `loss:` lines + `nvidia-smi` GPU memory **> 0**.

## Monitor from Windows

```powershell
cd c:\Users\chris\.cursor\projects\qlora-coding-beast
python scripts/runpod_monitor.py --pod-id k9ckjm5ybx0bhc --watch 120
```

## Launch new pod

```powershell
python scripts/runpod_launch.py --phase 1
```

Console link format: `https://console.runpod.io/pods?id=POD_ID`
