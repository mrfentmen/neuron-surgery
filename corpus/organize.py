#!/usr/bin/env python3
"""Find redundant jsonl and say exactly what could be moved.

Two kinds of waste in this tree:

  exact duplicates   byte-identical files, one canonical copy survives
  non-scan data      downloaded corpora that are not our measurements

Dry run by default. --apply MOVES (never deletes) redundant copies into
_trash_dupes/ with the same relative path, so putting one back is just mv.
It refuses to move a file whose path is referenced by name in any .py, .sh,
.ipynb or .md here, because those are the ones a script actually opens.

    .venv/bin/python corpus/organize.py             # report
    .venv/bin/python corpus/organize.py --apply     # move the safe ones
"""
import argparse
import collections
import hashlib
import json
import os
import re
import shutil

SKIP_DIRS = {".venv", "_trash_dupes", "datasets", "jailbreak-repos", "__pycache__", ".git"}
CODE_EXT = {".py", ".sh", ".ipynb"}


def walk_jsonl():
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if f.endswith(".jsonl") and not f.endswith("_cache.jsonl"):
                p = os.path.normpath(os.path.join(root, f))
                yield p


def sha(p):
    h = hashlib.blake2b(digest_size=16)
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(p):
    n = 0
    with open(p, "rb") as fh:
        for _ in fh:
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    files = list(walk_jsonl())
    total = len(files)
    empty = [p for p in files if os.path.getsize(p) == 0]
    files = [p for p in files if os.path.getsize(p) > 0]

    groups = collections.defaultdict(list)
    blank = 0
    for i, p in enumerate(files, 1):
        if rows(p) == 0:
            blank += 1
            continue
        groups[sha(p)].append(p)
        if i % 150 == 0:
            print(f"  hashed {i}/{len(files)}", flush=True)

    dupes = {h: sorted(ps, key=lambda x: (x.count("/"), len(x), x))
             for h, ps in groups.items() if len(ps) > 1}

    # references: does any script name this path?
    refs = []
    for root, dirs, fs in os.walk("."):
        dirs[:] = [d for d in dirs if d not in {".venv", "_trash_dupes", ".git", "datasets"}]
        for f in fs:
            if os.path.splitext(f)[1] in CODE_EXT and f != "organize.py":
                refs.append(os.path.join(root, f))
    blob = ""
    for r in refs:
        try:
            blob += open(r, encoding="utf-8", errors="ignore").read()
        except OSError:
            pass

    redundant, referenced, freed = [], [], 0
    for h, ps in dupes.items():
        canon = ps[0]
        for p in ps[1:]:
            named = (p in blob) or (os.path.basename(p) in blob and "/" not in p)
            if named:
                referenced.append((p, canon))
            else:
                redundant.append((p, canon))
                freed += os.path.getsize(p)

    print(f"\njsonl on disk      {total}")
    print(f"  zero-byte        {len(empty)}")
    print(f"  no parseable rows {blank}")
    print(f"  with data        {len(groups) and sum(len(v) for v in groups.values())}")
    print(f"duplicate groups   {len(dupes)}")
    print(f"  redundant copies {len(redundant)}  ({freed/1e6:.1f} MB)")
    print(f"  referenced in a script (kept) {len(referenced)}")

    print("\nredundant, safe to move (first 25):")
    for p, canon in redundant[:25]:
        print(f"  {p}  ==  {canon}")
    if referenced:
        print("\nredundant BUT named in a script, will not move:")
        for p, canon in referenced[:15]:
            print(f"  {p}  ==  {canon}")

    print("\nsize by top dir")
    top = collections.Counter()
    for root, dirs, fs in os.walk("."):
        dirs[:] = [d for d in dirs if d not in {".venv", ".git"}]
        first = os.path.normpath(root).split(os.sep)[0] if os.path.normpath(root) != "." else "(root)"
        for f in fs:
            if f.endswith(".jsonl"):
                top[first] += os.path.getsize(os.path.join(root, f))
    for k, v in top.most_common(12):
        print(f"  {k:<28} {v/1e6:>9.1f} MB")

    if a.apply:
        moved = 0
        for p, canon in redundant:
            dst = os.path.join("_trash_dupes", p)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                shutil.move(p, dst)
                moved += 1
        print(f"\nmoved {moved} files to _trash_dupes/ (undo: mv _trash_dupes/<path> <path>)")
    else:
        print("\n[dry run] nothing moved. --apply to move the safe ones.")


if __name__ == "__main__":
    main()
