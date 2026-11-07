"""Build the SFT corpora + the RFT prompt list. CPU only, no GPU, no downloads.

Why this exists: the 20-run plan needs one uniform schema, and every dataset on
disk uses a different one. This reads whatever is already in datasets/ and writes
runs/data/<source>.jsonl with {id, source, prompt, response, meta}.

Key detection is generic on purpose. Several datasets here have never been
inspected row-by-row; a schema we guessed wrong must skip, not crash the night.

Output:
  runs/data/<source>.jsonl            train rows
  runs/data/<source>.holdout.jsonl    last 100 rows, never trained on
  runs/data/prompts_rft.jsonl         prompts that carry a verifier
  runs/data/manifest.json             counts + rough token estimates
"""
import glob
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(ROOT, "datasets")
OUT = os.path.join(ROOT, "runs", "data")

MAX_PER_SOURCE = 10000
HOLDOUT = 100
MIN_PROMPT = 8
MIN_RESPONSE = 20

PROMPT_KEYS = ("question", "instruction", "prompt", "problem_statement", "problem",
               "query", "input", "text", "content")
RESP_KEYS = ("solution", "output", "response", "answer", "chosen", "completion",
             "patch", "code", "value")

# Sources that carry a checker, so they can be used for RFT / GRPO.
VERIFIER_SOURCES = {"opc-sft-stage2", "math-rlvr-15k", "LIMO"}


def norm(s):
    return " ".join(str(s).split())


def from_messages(row):
    msgs = row.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return None, None
    user = None
    asst = None
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        c = m.get("content")
        if not isinstance(c, str):
            continue
        if role == "user":
            user = c
        elif role == "assistant":
            asst = c
    return user, asst


def from_conversations(row):
    convs = row.get("conversations")
    if not isinstance(convs, list) or not convs:
        return None, None
    human = None
    gpt = None
    for c in convs:
        if not isinstance(c, dict):
            continue
        frm = c.get("from") or c.get("role")
        val = c.get("value") or c.get("content")
        if not isinstance(val, str):
            continue
        if frm in ("human", "user"):
            human = val
        elif frm in ("gpt", "assistant"):
            gpt = val
    return human, gpt


def extract(row):
    if not isinstance(row, dict):
        return None, None
    p, r = from_messages(row)
    if p and r:
        return p, r
    p2, r2 = from_conversations(row)
    if p2 and r2:
        return p2, r2
    p = r = None
    for k in PROMPT_KEYS:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            p = v
            break
        if isinstance(v, list) and v and isinstance(v[0], dict):
            # prompt given as a message list
            for m in v:
                if isinstance(m.get("content") if isinstance(m, dict) else None, str):
                    p = m["content"]
                    break
            if p:
                break
    for k in RESP_KEYS:
        if k == "chosen" and isinstance(row.get(k), list):
            v = row.get(k)
            if v and isinstance(v[0], dict):
                r = v[0].get("content")
            break
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            r = v
            break
    return p, r


def rows_from(path, cap):
    """Yield dict rows from .jsonl or .parquet, stopping at cap."""
    n = 0
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
                n += 1
                if n >= cap:
                    return
    elif path.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
        except Exception:
            return
        try:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=512):
                for row in batch.to_pylist():
                    yield row
                    n += 1
                    if n >= cap:
                        return
        except Exception:
            return


def source_files(d):
    files = sorted(glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True))
    files += sorted(glob.glob(os.path.join(d, "**", "*.parquet"), recursive=True))
    # skip obvious eval fixtures
    return [f for f in files if "eval_heldout" not in f]


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    manifest = {}
    verifier_pool = []

    dirs = sorted(p for p in glob.glob(os.path.join(DS, "*")) if os.path.isdir(p))
    for d in dirs:
        name = os.path.basename(d)
        files = source_files(d)
        if not files:
            print(f"[skip] {name:48} no jsonl/parquet")
            continue

        train_path = os.path.join(OUT, f"{name}.jsonl")
        hold_path = os.path.join(OUT, f"{name}.holdout.jsonl")
        seen = set()
        kept = 0
        est_tokens = 0
        skipped = 0

        with open(train_path, "w", encoding="utf-8") as ft, \
             open(hold_path, "w", encoding="utf-8") as fh:
            for f in files:
                for row in rows_from(f, MAX_PER_SOURCE * 2):
                    p, r = extract(row)
                    if not p or not r:
                        skipped += 1
                        continue
                    p, r = norm(p), str(r).strip()
                    if len(p) < MIN_PROMPT or len(r) < MIN_RESPONSE:
                        skipped += 1
                        continue
                    rid = hashlib.sha1((p[:2000] + "\x00" + r[:2000]).encode("utf-8")).hexdigest()[:16]
                    if rid in seen:
                        skipped += 1
                        continue
                    seen.add(rid)
                    rec = {"id": rid, "source": name, "prompt": p, "response": r}
                    # carry verifier material when the source has it
                    if name == "opc-sft-stage2" and isinstance(row.get("testcase"), list):
                        rec["verifier"] = {"kind": "asserts", "entry_point": row.get("entry_point"),
                                           "tests": row.get("testcase")[:12]}
                    if name == "math-rlvr-15k" and row.get("label") is not None:
                        rec["verifier"] = {"kind": "numeric", "label": str(row["label"])}
                    if name == "LIMO" and row.get("answer") is not None:
                        rec["verifier"] = {"kind": "numeric", "label": str(row["answer"])}

                    line = json.dumps(rec, ensure_ascii=False) + "\n"
                    if kept >= HOLDOUT and (kept % 50 == 0):
                        # sparse holdout draw so it is not just the tail of one file
                        fh.write(line)
                    else:
                        ft.write(line)
                    kept += 1
                    est_tokens += (len(p) + len(r)) // 4
                    if name in VERIFIER_SOURCES and "verifier" in rec and len(verifier_pool) < 2000:
                        verifier_pool.append({"id": rid, "source": name, "prompt": p,
                                              "verifier": rec["verifier"]})
                    if kept >= MAX_PER_SOURCE:
                        break
                if kept >= MAX_PER_SOURCE:
                    break

        manifest[name] = {"rows": kept, "skipped": skipped, "est_tokens": est_tokens,
                          "files": len(files)}
        print(f"[ok]   {name:48} {kept:>6} rows  ~{est_tokens/1e6:5.2f}M tok  (skip {skipped})")

    # RFT prompt list: only prompts attached to a checker, balanced across sources
    by_src = {}
    for v in verifier_pool:
        by_src.setdefault(v["source"], []).append(v)
    picks = []
    per = 150
    for src, items in by_src.items():
        picks.extend(items[:per])
    picks = picks[:400]
    with open(os.path.join(OUT, "prompts_rft.jsonl"), "w", encoding="utf-8") as f:
        for p in picks:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    manifest["_rft_prompts"] = len(picks)
    manifest["_build_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    total = sum(v["rows"] for k, v in manifest.items() if isinstance(v, dict))
    print(f"\n{total} rows -> {OUT}")
    print(f"{len(picks)} verifier prompts -> prompts_rft.jsonl")
    print(f"build took {manifest['_build_seconds']}s")


if __name__ == "__main__":
    main()
