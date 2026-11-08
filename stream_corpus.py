#!/usr/bin/env python3
"""One pass over a text corpus: tokenize it, record it, then delete it.

The rule this enforces is deliberately paranoid:

    a shard may only be deleted AFTER
      (a) every byte of it was tokenized, and
      (b) its record has been fsync'd into the manifest.

So a crash costs you at most one shard, never the corpus. Run with no flags and
it deletes nothing -- --delete must be passed explicitly.

Why delete at all: a few hundred GB of raw text does not fit next to a model and
its notes on this machine, and re-reading a shard you already trained on is pure
waste. Converting to tokens and dropping the source keeps one pass, one copy.

    python stream_corpus.py --corpus datasets/long --out data/ext        # dry run
    python stream_corpus.py --corpus datasets/long --out data/ext --delete --yes

Outputs, in --out:
    NNNN.bin            uint32 token ids, little-endian (granite vocab is 100,352,
                        so uint16 would silently wrap -- this is not a detail)
    shards.jsonl        append-only, fsynced, one line per consumed shard
    SKIP.txt            files too short to be worth a shard, and why
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

TEXT_EXT = {".txt", ".md", ".jsonl", ".json", ".py", ".c", ".h", ".cpp", ".rs",
            ".go", ".java", ".js", ".ts", ".tex", ".rst", ".csv"}


def iter_docs(root: Path):
    """Yield (path, text) for every readable document under root."""
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in TEXT_EXT:
            continue
        try:
            yield p, p.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            print(f"  unreadable {p}: {e}", file=sys.stderr)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="folder of raw text")
    ap.add_argument("--out", required=True, help="where tokens + manifest go")
    ap.add_argument("--model", default="ibm-granite/granite-4.0-h-tiny-base")
    ap.add_argument("--min-bytes", type=int, default=200_000,
                    help="skip documents shorter than this (default 200 KB ~ 50k tokens)")
    ap.add_argument("--tokens-per-shard", type=int, default=8_000_000)
    ap.add_argument("--delete", action="store_true", help="remove source files once recorded")
    ap.add_argument("--yes", action="store_true", help="required together with --delete")
    a = ap.parse_args()

    if a.delete and not a.yes:
        print("refusing: --delete also needs --yes", file=sys.stderr)
        return 2

    corpus, out = Path(a.corpus), Path(a.out)
    if not corpus.is_dir():
        print(f"no such corpus: {corpus}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)

    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    vocab = max(tok.get_vocab().values()) + 1
    if vocab > 2 ** 16:
        print(f"  vocab {vocab:,} -> uint32 (uint16 would wrap)")
    dtype_itemsize = 4

    manifest = out / "shards.jsonl"
    done = set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            try:
                done.add(json.loads(line)["source"])
            except Exception:
                pass
        print(f"  resuming: {len(done)} shards already consumed")

    skip_log = out / "SKIP.txt"
    buf: list[int] = []
    shard_idx = len(list(out.glob("*.bin")))
    consumed = deleted = skipped = 0
    total_tokens = 0

    def flush(buf: list[int]) -> None:
        nonlocal shard_idx
        path = out / f"{shard_idx:04d}.bin"
        tmp = path.with_suffix(".tmp")
        import array
        arr = array.array("I", buf)
        if sys.byteorder != "little":
            arr.byteswap()
        with open(tmp, "wb") as f:
            arr.tofile(f)
            f.flush()
            os.fsync(f.fileno())
        tmp.rename(path)
        shard_idx += 1

    for path, text in iter_docs(corpus):
        key = str(path)
        if key in done:
            continue
        if len(text.encode()) < a.min_bytes:
            skipped += 1
            with open(skip_log, "a") as f:
                f.write(f"{path}\tshort\t{len(text.encode())}\n")
            continue

        ids = tok(text, add_special_tokens=False)["input_ids"]
        buf.extend(ids)
        total_tokens += len(ids)

        rec = {
            "source": key,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "tokens": len(ids),
            "tokenizer": a.model,
        }
        with open(manifest, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())          # record is durable BEFORE any delete
        consumed += 1

        if a.delete:
            path.unlink()
            deleted += 1

        if len(buf) >= a.tokens_per_shard:
            flush(buf)
            buf = []
            print(f"  wrote shard {shard_idx - 1:04d}  ({total_tokens:,} tokens so far)")

        if consumed % 25 == 0:
            print(f"  {consumed} docs  {total_tokens:,} tokens  deleted={deleted}")

    if buf:
        flush(buf)

    print(f"\n  docs consumed {consumed}   skipped(short) {skipped}   deleted {deleted}")
    print(f"  tokens {total_tokens:,}  =  {total_tokens * 4 / 1e6:.1f} MB of equivalent text")
    print(f"  manifest {manifest}   shards {shard_idx}")
    if not a.delete:
        print("  DRY RUN: nothing was deleted. Re-run with --delete --yes to drop sources.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
