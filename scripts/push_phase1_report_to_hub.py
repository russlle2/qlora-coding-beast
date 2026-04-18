#!/usr/bin/env python3
"""Upload Phase-1 training summary + log tail to the adapter HF repo (not weights)."""
from __future__ import annotations

import json
import os
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path

REPO_ID = "russlle2/qwen3-coder-30b-a3b-adapter-uncensored"
LOG_PATH = Path("/workspace/outputs/train_phase1.log")
OUT_JSON = Path("/workspace/outputs/phase1_summary.json")


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required")

    from huggingface_hub import HfApi

    lines: list[str] = []
    if LOG_PATH.exists():
        raw = LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
        lines = raw[-400:]  # last ~400 lines

    summary = {
        "phase": "adapter-uncensored",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "adapter_repo": REPO_ID,
        "log_tail_lines": len(lines),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    readme = f"""---
license: apache-2.0
library_name: peft
tags:
- qlora
- qwen3-coder
- adapter
---

# Phase 1 — adapter-uncensored

Auto-uploaded summary after training pipeline.

- Finished (UTC): `{summary['finished_at_utc']}`
- Checkpoints and adapter weights are pushed by Axolotl during training (`hub_strategy: checkpoint`).

## Last lines of training log

```
{chr(10).join(lines)}
```
"""

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=BytesIO(readme.encode("utf-8")),
        path_in_repo="PHASE1_REPORT.md",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Add Phase 1 training report (auto)",
    )
    api.upload_file(
        path_or_fileobj=BytesIO(json.dumps(summary, indent=2).encode("utf-8")),
        path_in_repo="phase1_summary.json",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Add Phase 1 summary JSON (auto)",
    )
    print(f"[push_report] uploaded PHASE1_REPORT.md + phase1_summary.json to {REPO_ID}")


if __name__ == "__main__":
    main()
