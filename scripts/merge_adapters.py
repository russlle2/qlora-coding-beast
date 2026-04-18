#!/usr/bin/env python
"""
merge_adapters.py
Weighted merge of up to 3 LoRA adapters into the base model, save as BF16 safetensors.

Two modes:
  1) Phase 1 smoke test: merge just adapter-uncensored into base
  2) Phase 2 final: merge all 3 adapters (uncensored + tools + coding) with weights

Usage:
    # Phase 1 - single adapter merge, smoke test
    python scripts/merge_adapters.py --mode phase1 \
        --out /workspace/outputs/merged_phase1_bf16

    # Phase 2 - 3-adapter weighted merge, default weights 0.7/1.0/1.0
    python scripts/merge_adapters.py --mode phase2 \
        --out /workspace/outputs/merged_final_bf16

    # Phase 2 with custom weights + TIES combination
    python scripts/merge_adapters.py --mode phase2 \
        --weights 0.5 1.0 1.0 \
        --combination-type ties --density 0.5 \
        --out /workspace/outputs/merged_ties_bf16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
HF_ORG = "russlle2"
ADAPTER_UNCENSORED = f"{HF_ORG}/qwen3-coder-30b-a3b-adapter-uncensored"
ADAPTER_TOOLS = f"{HF_ORG}/qwen3-coder-30b-a3b-adapter-tools"
ADAPTER_CODING = f"{HF_ORG}/qwen3-coder-30b-a3b-adapter-coding"


def load_base_bf16():
    """Load base in BF16 (NOT 4-bit) so merged weights can be saved full precision."""
    print(f"[merge] loading base {BASE} in bf16 - this takes a few minutes + needs ~62GB VRAM or will CPU-offload")
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    return model, tokenizer


def phase1_merge(out_dir: Path) -> None:
    """Merge just adapter-uncensored. For the $8 smoke-test after Phase 1."""
    from peft import PeftModel

    model, tokenizer = load_base_bf16()

    print(f"[merge] loading adapter {ADAPTER_UNCENSORED}")
    model = PeftModel.from_pretrained(model, ADAPTER_UNCENSORED)

    print("[merge] merge_and_unload...")
    model = model.merge_and_unload()

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[merge] saving BF16 to {out_dir}")
    model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(out_dir)
    print("[merge] phase1 done")


def phase2_merge(
    out_dir: Path,
    weights: list[float],
    combination_type: str,
    density: float,
) -> None:
    """Weighted 3-adapter merge."""
    from peft import PeftModel

    assert len(weights) == 3, "phase2 requires 3 weights: uncensored, tools, coding"

    model, tokenizer = load_base_bf16()

    print(f"[merge] loading adapter uncensored from {ADAPTER_UNCENSORED}")
    model = PeftModel.from_pretrained(model, ADAPTER_UNCENSORED, adapter_name="uncensored")
    print(f"[merge] loading adapter tools from {ADAPTER_TOOLS}")
    model.load_adapter(ADAPTER_TOOLS, adapter_name="tools")
    print(f"[merge] loading adapter coding from {ADAPTER_CODING}")
    model.load_adapter(ADAPTER_CODING, adapter_name="coding")

    print(
        f"[merge] add_weighted_adapter: weights={weights}, "
        f"combination_type={combination_type}, density={density}"
    )
    kwargs: dict = dict(
        adapters=["uncensored", "tools", "coding"],
        weights=weights,
        adapter_name="merged",
        combination_type=combination_type,
    )
    # density only applies to ties / ties_svd / dare_ties / dare_linear
    if combination_type in {"ties", "ties_svd", "dare_ties", "dare_linear"}:
        kwargs["density"] = density
    model.add_weighted_adapter(**kwargs)
    model.set_adapter("merged")

    print("[merge] merge_and_unload...")
    merged = model.merge_and_unload()

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[merge] saving BF16 to {out_dir}")
    merged.save_pretrained(out_dir, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(out_dir)
    print("[merge] phase2 done")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["phase1", "phase2"])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--weights",
        nargs=3,
        type=float,
        default=[0.7, 1.0, 1.0],
        help="3 weights for phase2 merge: [uncensored, tools, coding]",
    )
    p.add_argument(
        "--combination-type",
        default="linear",
        choices=["linear", "cat", "ties", "dare_ties", "dare_linear", "magnitude_prune"],
    )
    p.add_argument("--density", type=float, default=0.5,
                   help="density for ties/dare combination (0-1)")
    args = p.parse_args()

    if args.mode == "phase1":
        phase1_merge(args.out)
    else:
        phase2_merge(args.out, args.weights, args.combination_type, args.density)
    return 0


if __name__ == "__main__":
    sys.exit(main())
