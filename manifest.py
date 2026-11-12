#!/usr/bin/env python3
"""Record everything needed to reproduce a number in this repo.

The audit that prompted this was blunt about it: results here are only citable if
a reader can tell **which conditions produced them**. Right now the conditions are
spread across docstrings and shell history — dtype in one file, dose convention in
another, the thermal workaround in a third. This writes them into one machine
readable file and one human readable page.

    python manifest.py            # full: hashes every jsonl under 100 MB
    python manifest.py --quick    # sizes and row counts only

Writes MANIFEST.json and MANIFEST.md. Nothing here is a result; it is the frame
every result sits in.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models/granite-4.1-3b-base"
SKIP_DIRS = {".venv", "datasets", ".git", "__pycache__", "jailbreak-repos",
             ".obsidian", "_quarantine_qwen"}
BIG = 100 * 1024 * 1024


def run(cmd, cwd=ROOT):
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60)
        return (p.stdout or "").strip() or (p.stderr or "").strip()
    except Exception as e:
        return f"<{e}>"


def sha256(path, limit=None):
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            n += len(b)
            if limit and n >= limit:
                break
    return h.hexdigest() + (f":prefix{n}" if limit and n < path.stat().st_size else "")


def count_rows(path):
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def model_card():
    card = {"path": str(MODEL.relative_to(ROOT)) if MODEL.exists() else None,
            "present": MODEL.exists(), "files": {}, "config": {}}
    if not MODEL.exists():
        return card
    for name in ("config.json", "generation_config.json", "tokenizer_config.json",
                 "special_tokens_map.json", "chat_template.jinja"):
        p = MODEL / name
        if p.exists():
            card["files"][name] = {"bytes": p.stat().st_size,
                                   "sha256": sha256(p)}
            if name == "config.json":
                card["config"] = json.loads(p.read_text())
    shards = sorted(MODEL.glob("*.safetensors"))
    card["shards"] = []
    for p in shards:
        card["shards"].append({"name": p.name, "bytes": p.stat().st_size,
                               "sha256_head": sha256(p, limit=8 << 20)})
    card["total_bytes"] = sum(x["bytes"] for x in card["shards"])
    card["revision_note"] = run(["git", "rev-parse", "--short", "HEAD"],
                                cwd=MODEL) if (MODEL / ".git").exists() else "not a git checkout"
    return card


def env_card():
    import torch
    import transformers
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "platform": platform.platform(),
        "macos": run(["sw_vers", "-productVersion"]),
        "chip": run(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "memory_bytes": int(run(["sysctl", "-n", "hw.memsize"]) or 0),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_used_by_runners": "mps",
    }


def decode_card():
    """The decoding settings every harm-lane result in this repo used."""
    return {
        "dtype": "bfloat16",
        "device": "mps",
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": 60,
        "chat_template": "tokenizer default (granite chat_template.jinja), no system prompt",
        "greedy_runs": "max_new 300-500 for MARATHON/LONGFORM; noted per doc",
        "dose_convention": "alpha x median(residual norm at layer+1), applied to the "
                           "MLP intermediate (down_proj input) for neuron pushes, or "
                           "to the residual stream for direction pushes",
    }


def duty_card():
    """Thermal protection actually in force, and the one that is not."""
    return {
        "launcher": "run_cool.sh",
        "measured_safe": "24 gens/min, 0.8s gaps, 30s rest every 40 rows",
        "used_in_recent_runs": "~16-17 gens/min, 0.9s gaps, 30s rest every 25 rows",
        "thermal_guard_status": "BLIND - pmset -g therm does not emit CPU_Speed_Limit "
                                "on this machine, so the SIGSTOP path never fires",
        "evidence": "THERMAL.md",
        "therm_raw": run(["pmset", "-g", "therm"]),
    }


def inventory(quick=False):
    files = []
    for p in ROOT.rglob("*.jsonl"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        rel = str(p.relative_to(ROOT))
        size = p.stat().st_size
        rec = {"path": rel, "bytes": size}
        try:
            rec["rows"] = count_rows(p)
        except Exception as e:
            rec["rows"] = None
            rec["error"] = str(e)
        if not quick and size <= BIG:
            rec["sha256"] = sha256(p)
        files.append(rec)
    files.sort(key=lambda r: -r["bytes"])
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip hashing and row counts for files over 100 MB")
    ap.add_argument("--out", default="MANIFEST")
    a = ap.parse_args()

    inv = inventory(a.quick)
    total_rows = sum(r["rows"] or 0 for r in inv)
    man = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": {
            "head": run(["git", "rev-parse", "HEAD"]),
            "head_short": run(["git", "rev-parse", "--short", "HEAD"]),
            "remote": run(["git", "remote", "get-url", "origin"]),
            "dirty_paths": len([l for l in run(["git", "status", "--porcelain"]).splitlines() if l]),
        },
        "environment": env_card(),
        "model": model_card(),
        "decode": decode_card(),
        "thermal": duty_card(),
        "corpus": {
            "jsonl_files": len(inv),
            "rows_total": total_rows,
            "bytes_total": sum(r["bytes"] for r in inv),
            "note": "rows counted line by line; datasets/ excluded as external corpora",
        },
        "files": inv,
    }
    Path(f"{a.out}.json").write_text(json.dumps(man, indent=2))

    cfg = man["model"]["config"]
    lines = [
        "# MANIFEST", "",
        f"Generated {man['generated_utc']} by `manifest.py`. "
        f"Commit `{man['git']['head_short']}`"
        + (f" ({man['git']['dirty_paths']} dirty paths)" if man["git"]["dirty_paths"] else "")
        + ".", "",
        "Every number in this repo was produced under these conditions. If a result "
        "cannot be reproduced from this page plus the command in its doc, that result "
        "is not a result.", "",
        "## Model", "",
        f"- local checkout: `{man['model']['path']}`",
        f"- layers: **{cfg.get('num_hidden_layers')}**, hidden: **{cfg.get('hidden_size')}**, "
        f"intermediate: **{cfg.get('intermediate_size')}**, vocab: **{cfg.get('vocab_size')}**",
        f"- dtype at rest: **bfloat16**, device: **mps**",
        f"- weights: {len(man['model'].get('shards', []))} shards, "
        f"{man['model'].get('total_bytes', 0) / 1e9:.2f} GB, head-hashed in MANIFEST.json",
        f"- note: {man['model']['revision_note']}",
        "",
        "## Runtime", "",
        "| item | value |", "|---|---|",
    ]
    for k, v in man["environment"].items():
        lines.append(f"| {k} | `{v}` |")
    lines += ["", "## Decoding", "", "| setting | value |", "|---|---|"]
    for k, v in man["decode"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Thermal", "", "| item | value |", "|---|---|"]
    for k, v in man["thermal"].items():
        lines.append(f"| {k} | {str(v).replace(chr(10), ' ')[:160]} |")
    lines += [
        "", "**The guard is blind.** `run_cool.sh` SIGSTOPs the worker when",
        "`pmset -g therm` reports a speed limit below 90%, and this machine never",
        "emits that key. Every run is protected by duty cycle alone. `THERMAL.md`",
        "records the measurements that set the rate; no run here has a thermal",
        "measurement behind its safety claim.", "",
        "## Corpus", "",
        f"- `{man['corpus']['jsonl_files']}` jsonl files outside `datasets/`",
        f"- `{man['corpus']['rows_total']:,}` rows, "
        f"`{man['corpus']['bytes_total'] / 1e9:.2f}` GB",
        f"- per-file rows, bytes and sha256 in `MANIFEST.json`", "",
        "## What this does not cover", "",
        "1. **One machine, one dtype.** Every result is bf16 on MPS. `CO-STEER.md`",
        "   records that the threat register was produced in fp16 on CUDA and does not",
        "   reappear here, so dtype/device is a live confound in at least one family.",
        "2. **No lockfile.** Versions are recorded, not pinned; a future torch build",
        "   can move the numbers.",
        "3. **Tokeniser hash only.** The chat template is hashed; the merge rules are",
        "   covered by the same file set but not individually asserted.",
        "", "## Files", "",
        "`MANIFEST.json` (machine readable, includes every sha256) · `manifest.py`",
    ]
    Path(f"{a.out}.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {a.out}.json and {a.out}.md")
    print(f"  {man['corpus']['jsonl_files']} files, {total_rows:,} rows, "
          f"{man['corpus']['bytes_total'] / 1e9:.2f} GB")
    print(f"  model {cfg.get('num_hidden_layers')}L x {cfg.get('intermediate_size')} "
          f"= {cfg.get('num_hidden_layers', 0) * cfg.get('intermediate_size', 0):,} tiles")
    print(f"  git {man['git']['head_short']} dirty={man['git']['dirty_paths']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
