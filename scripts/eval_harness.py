#!/usr/bin/env python
"""
eval_harness.py
Phase 2 eval gate: HumanEval+/MBPP+ (via evalplus) + an internal tool-call probe.

Usage (on a pod with the merged BF16 model loaded, OR against a GGUF via llama-cpp-python):
    # Against a merged BF16 HF dir
    python scripts/eval_harness.py \
        --model-path /workspace/outputs/merged_final_bf16 \
        --out /workspace/outputs/eval_report.json

    # Quick tool-call probe only (skip evalplus for fast iteration)
    python scripts/eval_harness.py \
        --model-path /workspace/outputs/merged_final_bf16 \
        --tool-probe-only --out /workspace/outputs/eval_tool.json

Outputs a JSON report with per-benchmark scores + pass/fail vs gate thresholds.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---- gate thresholds (see plan section 7) ----
GATE_HUMANEVAL_PLUS = 0.85
GATE_MBPP_PLUS = 0.80
GATE_TOOL_SCHEMA_VALID = 0.90
GATE_TOOL_SEMANTIC_CORRECT = 0.80


# ---- a tiny self-contained tool-call probe ----
TOOL_PROBE_CASES = [
    {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "query": "What's the weather in Tokyo in celsius?",
        "expected_name": "get_weather",
        "expected_args_must_contain": {"city": "Tokyo", "unit": "celsius"},
    },
    {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "run_port_scan",
                    "description": "Run an nmap port scan against a target",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target": {"type": "string"},
                            "ports": {"type": "string", "description": "e.g. '1-1000' or '22,80,443'"},
                            "scan_type": {"type": "string", "enum": ["syn", "connect", "udp"]},
                        },
                        "required": ["target", "ports"],
                    },
                },
            }
        ],
        "query": "Do a SYN scan of ports 1-1024 on 192.168.1.1",
        "expected_name": "run_port_scan",
        "expected_args_must_contain": {"target": "192.168.1.1", "scan_type": "syn"},
    },
    {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "create_github_issue",
                    "description": "Create an issue on a GitHub repo",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo": {"type": "string"},
                            "title": {"type": "string"},
                            "body": {"type": "string"},
                            "labels": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["repo", "title"],
                    },
                },
            }
        ],
        "query": "Open an issue on russlle2/qlora-coding-beast titled 'Phase 1 tokens/sec too low' and label it 'perf'",
        "expected_name": "create_github_issue",
        "expected_args_must_contain": {"repo": "russlle2/qlora-coding-beast", "title": "Phase 1 tokens/sec too low"},
    },
    {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search_db",
                    "description": "Search a user database",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                        "required": ["query"],
                    },
                },
            }
        ],
        "query": "Find the top 5 users named Jane in the users table",
        "expected_name": "search_db",
        "expected_args_must_contain": {"query": "Jane", "limit": 5},
    },
    {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "exec_sql",
                    "description": "Execute a SQL query against Postgres",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sql": {"type": "string"},
                            "readonly": {"type": "boolean"},
                        },
                        "required": ["sql"],
                    },
                },
            }
        ],
        "query": "SELECT count of all rows in the orders table, read-only",
        "expected_name": "exec_sql",
        "expected_args_must_contain_substr": {"sql": "COUNT"},
        "expected_args_must_contain": {"readonly": True},
    },
]

TOOL_SYSTEM = (
    "You are a tool-calling assistant. Call the provided tools with precise JSON arguments. "
    "Emit each tool call inside a <tool_call>...</tool_call> block containing a JSON object "
    'with exactly two fields: "name" and "arguments".'
)


def extract_tool_call(text: str) -> dict | None:
    """Best-effort parse of a <tool_call>{JSON}</tool_call> block."""
    import re
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.DOTALL)
    if not m:
        # Fallback: any top-level JSON object with "name" and "arguments"
        m2 = re.search(r"\{[^{}]*\"name\"[^{}]*\"arguments\"[^{}]*\}", text, flags=re.DOTALL)
        if not m2:
            return None
        raw = m2.group(0)
    else:
        raw = m.group(1)
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    if not isinstance(obj, dict) or "name" not in obj or "arguments" not in obj:
        return None
    return obj


def run_tool_probe(model, tokenizer, device: str) -> dict:
    results = []
    for case in TOOL_PROBE_CASES:
        tools_str = json.dumps(case["tools"], indent=2)
        system = f"{TOOL_SYSTEM}\n\nAvailable tools:\n{tools_str}"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": case["query"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)

        parsed = extract_tool_call(decoded)
        schema_valid = parsed is not None
        semantic_correct = False
        reason = ""

        if not schema_valid:
            reason = "no_tool_call_extracted"
        else:
            name_ok = parsed["name"] == case["expected_name"]
            args = parsed.get("arguments") or {}
            if not isinstance(args, dict):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            checks_ok = True
            for k, v in case.get("expected_args_must_contain", {}).items():
                if args.get(k) != v:
                    checks_ok = False
                    reason = f"arg {k}: expected={v!r}, got={args.get(k)!r}"
                    break
            if checks_ok:
                for k, substr in case.get("expected_args_must_contain_substr", {}).items():
                    val = args.get(k) or ""
                    if substr.lower() not in str(val).lower():
                        checks_ok = False
                        reason = f"arg {k}: expected substring {substr!r}, got {val!r}"
                        break
            semantic_correct = name_ok and checks_ok
            if not name_ok:
                reason = f"name: expected={case['expected_name']}, got={parsed['name']}"

        results.append({
            "query": case["query"],
            "raw_output": decoded[:2000],
            "parsed": parsed,
            "schema_valid": schema_valid,
            "semantic_correct": semantic_correct,
            "failure_reason": reason if not semantic_correct else "",
        })

    n = len(results)
    schema_rate = sum(1 for r in results if r["schema_valid"]) / n
    semantic_rate = sum(1 for r in results if r["semantic_correct"]) / n
    return {
        "n": n,
        "schema_valid_rate": schema_rate,
        "semantic_correct_rate": semantic_rate,
        "gate_schema_pass": schema_rate >= GATE_TOOL_SCHEMA_VALID,
        "gate_semantic_pass": semantic_rate >= GATE_TOOL_SEMANTIC_CORRECT,
        "cases": results,
    }


def run_evalplus(model_path: str, out_dir: Path) -> dict:
    """Call evalplus.evaluate for HumanEval+ and MBPP+. Requires model loadable via HF transformers."""
    out_dir.mkdir(parents=True, exist_ok=True)
    scores: dict = {}
    for bench in ["humaneval", "mbpp"]:
        print(f"[eval] running evalplus on {bench}+")
        log = out_dir / f"evalplus_{bench}.log"
        try:
            subprocess.run(
                [
                    "evalplus.evaluate",
                    "--dataset", bench,
                    "--model", model_path,
                    "--backend", "hf",
                    "--greedy",
                    "--root", str(out_dir / f"evalplus_{bench}"),
                ],
                check=True,
                stdout=log.open("w"),
                stderr=subprocess.STDOUT,
            )
            # Parse the emitted scores file - evalplus writes a results JSON
            # Path convention: {root}/{model_name}/evalplus_result.json
            result_files = list((out_dir / f"evalplus_{bench}").rglob("evalplus_result.json"))
            if result_files:
                data = json.loads(result_files[0].read_text())
                # evalplus reports pass@1 both base + plus
                plus_score = data.get("pass@1", {}).get(f"{bench}+", None)
                base_score = data.get("pass@1", {}).get(bench, None)
                scores[bench] = {"base": base_score, "plus": plus_score}
            else:
                scores[bench] = {"error": "no result file found; check log"}
        except subprocess.CalledProcessError as e:
            scores[bench] = {"error": f"evalplus failed: {e}; see {log}"}
    return scores


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, help="HF dir of merged BF16 model")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--tool-probe-only", action="store_true",
                   help="Skip evalplus; run only the internal tool-call probe")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] loading model from {args.model_path} in bf16 on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print("[eval] running tool-call probe...")
    t0 = time.time()
    tool_results = run_tool_probe(model, tokenizer, device)
    tool_elapsed = time.time() - t0
    print(f"[eval] tool probe done in {tool_elapsed:.1f}s: "
          f"schema={tool_results['schema_valid_rate']:.2f} "
          f"semantic={tool_results['semantic_correct_rate']:.2f}")

    report: dict = {
        "model_path": str(args.model_path),
        "tool_probe": tool_results,
        "tool_probe_elapsed_s": tool_elapsed,
    }

    if not args.tool_probe_only:
        del model  # free VRAM; evalplus spawns its own
        torch.cuda.empty_cache()
        print("[eval] running evalplus (HumanEval+, MBPP+)...")
        evalplus_scores = run_evalplus(args.model_path, args.out.parent / "evalplus_logs")
        report["evalplus"] = evalplus_scores

        # Gate check
        he_plus = (evalplus_scores.get("humaneval") or {}).get("plus")
        mbpp_plus = (evalplus_scores.get("mbpp") or {}).get("plus")
        report["gates"] = {
            "humaneval_plus": {"score": he_plus, "threshold": GATE_HUMANEVAL_PLUS,
                               "pass": (he_plus or 0) >= GATE_HUMANEVAL_PLUS},
            "mbpp_plus": {"score": mbpp_plus, "threshold": GATE_MBPP_PLUS,
                          "pass": (mbpp_plus or 0) >= GATE_MBPP_PLUS},
            "tool_schema_valid": {"score": tool_results["schema_valid_rate"],
                                  "threshold": GATE_TOOL_SCHEMA_VALID,
                                  "pass": tool_results["gate_schema_pass"]},
            "tool_semantic_correct": {"score": tool_results["semantic_correct_rate"],
                                      "threshold": GATE_TOOL_SEMANTIC_CORRECT,
                                      "pass": tool_results["gate_semantic_pass"]},
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"[eval] report -> {args.out}")
    if "gates" in report:
        print("[eval] gate summary:")
        for name, g in report["gates"].items():
            mark = "PASS" if g["pass"] else "FAIL"
            print(f"  {mark}  {name}: {g['score']} (>= {g['threshold']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
