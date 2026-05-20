#!/usr/bin/env python3
"""Patch axolotl 0.16.1 for torch 2.5.x and ship missing data files."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def find_axolotl_dir() -> Path:
    import axolotl

    return Path(axolotl.__file__).resolve().parent


def patch_enums(axolotl_dir: Path) -> None:
    p = axolotl_dir / "utils" / "schemas" / "enums.py"
    if not p.exists():
        print(f"[patch] enums.py not found at {p}; skipping")
        return
    text = p.read_text(encoding="utf-8")
    if 'getattr(torch, "int4"' in text or 'getattr(torch, "int1"' in text:
        print("[patch] enums.py already patched")
        return

    out: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if " = torch." in line and not s.startswith("class ") and "getattr" not in line:
            lhs, _, rhs = s.partition(" = torch.")
            indent = line[: len(line) - len(line.lstrip())]
            dtype = rhs.strip()
            fb = "torch.int8" if dtype.startswith("int") else "torch.float32"
            out.append(f'{indent}{lhs} = getattr(torch, "{dtype}", {fb})')
        else:
            out.append(line)
    p.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"[patch] enums.py patched at {p}")


def install_whitelist(axolotl_dir: Path) -> None:
    src = REPO_ROOT / "scripts" / "axolotl_whitelist.yaml"
    dst = axolotl_dir / "telemetry" / "whitelist.yaml"
    if not src.exists():
        print(f"[patch] {src} missing; cannot install whitelist")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"[patch] whitelist.yaml installed at {dst}")


def main() -> int:
    axolotl_dir = find_axolotl_dir()
    patch_enums(axolotl_dir)
    install_whitelist(axolotl_dir)

    import os
    os.environ.setdefault("AXOLOTL_DO_NOT_TRACK", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")

    import subprocess

    r = subprocess.run(
        [sys.executable, "-c", "from axolotl.cli.train import *; print('AXOLOTL TRAIN OK')"],
        check=False,
        env={**os.environ},
    )
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
