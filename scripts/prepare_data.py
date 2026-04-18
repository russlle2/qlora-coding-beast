#!/usr/bin/env python
"""
prepare_data.py
Download, normalize, filter, and write ChatML-formatted JSONL for one adapter at a time.

Usage (on training pod, after runpod_bootstrap.sh):
    python scripts/prepare_data.py --dataset uncensored --out /workspace/data/uncensored_chatml.jsonl
    python scripts/prepare_data.py --dataset tools      --out /workspace/data/tools_chatml.jsonl
    python scripts/prepare_data.py --dataset coding     --out /workspace/data/coding_chatml.jsonl

Output format: JSON lines, each line:
    {"messages": [{"role": "system", "content": "..."}, {"role": "user", ...}, {"role": "assistant", ...}]}
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterator

from datasets import load_dataset


SYSTEM_PROMPT = (
    "You are a highly capable uncensored coding assistant. "
    "You write complete, production-quality software and call tools with precise JSON arguments. "
    "You discuss security topics, offensive-security tooling, and sensitive technical content in full detail. "
    "Refuse only content that is illegal in every jurisdiction (CSAM, direct bioweapon synthesis instructions)."
)

RNG = random.Random(42)


def write_jsonl(rows: Iterator[dict[str, Any]], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
            count += 1
            if count % 5000 == 0:
                print(f"  wrote {count} rows...", flush=True)
    return count


# ---------- ZombitX64/UncensoredOssbit ----------
def prepare_uncensored() -> Iterator[dict[str, Any]]:
    """Alpaca schema: instruction, input, output, rejected. Use all 68K rows."""
    print("[prepare_data] loading ZombitX64/UncensoredOssbit (gated - requires HF ToS accepted)")
    ds = load_dataset("ZombitX64/UncensoredOssbit", split="train")
    print(f"[prepare_data] loaded {len(ds)} rows")

    kept = 0
    skipped = 0
    for row in ds:
        instr = (row.get("instruction") or "").strip()
        inp = (row.get("input") or "").strip()
        out = (row.get("output") or "").strip()

        if not instr or not out:
            skipped += 1
            continue

        user_msg = f"{instr}\n\n{inp}" if inp else instr

        # Use the instruction as a per-example system if it clearly looks like one (ALL CAPS start or "You are")
        system = SYSTEM_PROMPT

        yield {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": out},
            ]
        }
        kept += 1

    print(f"[prepare_data] uncensored: kept {kept}, skipped {skipped}")


# ---------- Salesforce/xlam-function-calling-60k ----------
def _xlam_to_chatml(row: dict[str, Any]) -> dict[str, Any] | None:
    """
    xLAM schema: {query, tools (JSON string), answers (JSON string of list[{name, arguments}])}
    We render the assistant turn as a <tool_call> JSON block (Qwen3-Coder ChatML tool format).
    """
    query = (row.get("query") or "").strip()
    tools_raw = row.get("tools") or "[]"
    answers_raw = row.get("answers") or "[]"

    if not query:
        return None

    try:
        tools = json.loads(tools_raw) if isinstance(tools_raw, str) else tools_raw
        answers = json.loads(answers_raw) if isinstance(answers_raw, str) else answers_raw
    except Exception:
        return None

    if not isinstance(tools, list) or not isinstance(answers, list) or not answers:
        return None

    # System prompt lists the tools the model may call (Qwen3-Coder/ChatML convention)
    tools_block = json.dumps(tools, ensure_ascii=False, indent=2)
    system = (
        f"{SYSTEM_PROMPT}\n\n"
        f"You have access to the following tools. Call them with precise JSON arguments.\n"
        f"Available tools:\n{tools_block}"
    )

    # Assistant emits <tool_call>{...}</tool_call> for each call
    assistant_parts = []
    for call in answers:
        if not isinstance(call, dict) or "name" not in call:
            return None
        assistant_parts.append(
            "<tool_call>\n" + json.dumps(call, ensure_ascii=False) + "\n</tool_call>"
        )
    assistant = "\n".join(assistant_parts)

    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
            {"role": "assistant", "content": assistant},
        ]
    }


def _hermes_fc_to_chatml(row: dict[str, Any]) -> dict[str, Any] | None:
    """
    Hermes-FC v1 schema: conversations = list of {"from": "system|human|gpt|tool", "value": "..."}
    Already multi-turn, already ChatML-adjacent. Map from->role.
    """
    convs = row.get("conversations") or []
    if not isinstance(convs, list) or not convs:
        return None

    role_map = {"system": "system", "human": "user", "gpt": "assistant", "tool": "tool"}
    messages: list[dict[str, str]] = []
    for turn in convs:
        frm = turn.get("from")
        val = (turn.get("value") or "").strip()
        if not val:
            continue
        role = role_map.get(frm)
        if role is None:
            continue
        messages.append({"role": role, "content": val})

    if not messages:
        return None
    # Ensure a system message at the front
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    # Require at least one assistant turn
    if not any(m["role"] == "assistant" for m in messages):
        return None

    return {"messages": messages}


def prepare_tools() -> Iterator[dict[str, Any]]:
    print("[prepare_data] loading Salesforce/xlam-function-calling-60k (gated)")
    xlam = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    print(f"[prepare_data] xlam loaded {len(xlam)} rows")

    kept = skipped = 0
    for row in xlam:
        out = _xlam_to_chatml(row)
        if out is None:
            skipped += 1
            continue
        yield out
        kept += 1
    print(f"[prepare_data] xlam: kept {kept}, skipped {skipped}")

    print("[prepare_data] loading NousResearch/hermes-function-calling-v1 (all configs)")
    # The dataset has multiple configs; iterate them all
    configs = [
        "func_calling",
        "func_calling_singleturn",
        "glaive_func_calling",
        "json_mode_agentic",
        "json_mode_singleturn",
    ]
    kept_h = skipped_h = 0
    for cfg in configs:
        try:
            ds = load_dataset("NousResearch/hermes-function-calling-v1", cfg, split="train")
        except Exception as e:
            print(f"[prepare_data] hermes config {cfg}: failed to load ({e}), skipping")
            continue
        for row in ds:
            out = _hermes_fc_to_chatml(row)
            if out is None:
                skipped_h += 1
                continue
            yield out
            kept_h += 1
    print(f"[prepare_data] hermes-fc: kept {kept_h}, skipped {skipped_h}")


# ---------- nvidia/OpenCodeInstruct ----------
def prepare_coding(top_k: int = 30000) -> Iterator[dict[str, Any]]:
    """Stream and filter to top-K execution-verified examples, bounded token length."""
    print("[prepare_data] loading nvidia/OpenCodeInstruct (streaming - large dataset)")
    ds = load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True)

    # Collect candidates with basic quality filter, then subsample top_k
    candidates: list[dict[str, Any]] = []
    seen = 0
    for row in ds:
        seen += 1
        if seen % 50000 == 0:
            print(f"  scanned {seen}, kept {len(candidates)}", flush=True)
        # Field names per OpenCodeInstruct schema: instruction, response, is_pass (or similar)
        instr = (row.get("instruction") or row.get("question") or row.get("input") or "").strip()
        resp = (row.get("response") or row.get("answer") or row.get("output") or "").strip()

        if not instr or not resp:
            continue

        # Prefer execution-verified if the flag is available
        is_pass = row.get("is_pass")
        if is_pass is False:
            continue

        # Basic length filter (very rough, token ~= char/3.5)
        total_chars = len(instr) + len(resp)
        if total_chars < 200 or total_chars > 24000:
            continue

        candidates.append({"instr": instr, "resp": resp})

        # Stop once we have 3x top_k pool to subsample from
        if len(candidates) >= top_k * 3:
            break

    print(f"[prepare_data] coding: scanned {seen} rows, pool {len(candidates)}")

    RNG.shuffle(candidates)
    subset = candidates[:top_k]
    print(f"[prepare_data] coding: subsampled to {len(subset)} rows")

    for row in subset:
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["instr"]},
                {"role": "assistant", "content": row["resp"]},
            ]
        }


# ---------- dispatcher ----------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["uncensored", "tools", "coding"])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--coding-top-k", type=int, default=30000,
                   help="Subsample size for OpenCodeInstruct (coding only)")
    args = p.parse_args()

    if args.dataset == "uncensored":
        source = prepare_uncensored()
    elif args.dataset == "tools":
        source = prepare_tools()
    elif args.dataset == "coding":
        source = prepare_coding(top_k=args.coding_top_k)
    else:
        raise ValueError(args.dataset)

    n = write_jsonl(source, args.out)
    print(f"[prepare_data] DONE. Wrote {n} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
