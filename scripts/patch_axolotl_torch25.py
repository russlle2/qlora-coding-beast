#!/usr/bin/env python3
"""Patch axolotl 0.16.1 enums for torch 2.5.x (no torch.int4). Safe for QLoRA train."""
from __future__ import annotations

import sys
from pathlib import Path


def find_enums_file() -> Path:
    import axolotl

    p = Path(axolotl.__file__).resolve().parent / "utils" / "schemas" / "enums.py"
    if not p.exists():
        raise SystemExit(f"enums.py not found at {p}")
    return p


def patch(text: str) -> str:
    if "getattr(torch, \"int4\"" in text:
        print("[patch] already patched")
        return text

    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "getattr(torch," in line:
            out.append(line)
            continue
        if " = torch." in line and not stripped.startswith("class "):
            lhs, _, rhs = stripped.partition(" = torch.")
            indent = line[: len(line) - len(line.lstrip())]
            dtype = rhs.strip()
            fallback = "torch.int8" if dtype.startswith("int") else "torch.float32"
            out.append(
                f'{indent}{lhs} = getattr(torch, "{dtype}", {fallback})'
            )
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def main() -> int:
    path = find_enums_file()
    original = path.read_text(encoding="utf-8")
    patched = patch(original)
    if patched != original:
        path.write_text(patched, encoding="utf-8")
        print(f"[patch] updated {path}")
    python = sys.executable
    import subprocess

    r = subprocess.run(
        [
            python,
            "-c",
            "from axolotl.cli.main import main; print('AXOLOTL CLI OK')",
        ],
        check=False,
    )
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
