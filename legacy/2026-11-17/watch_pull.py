#!/usr/bin/env python3
"""Watch the harvester. Re-verify every page, halt on a bad trend.

Runs beside run_pull.sh. Every --interval seconds it:

  1. re-reads every page in train/ and re-validates each row from scratch
  2. counts quarantined pages in rejects/
  3. appends a heartbeat to logs/watch_pull.log
  4. touches STOP if the reject ratio crosses --max-reject-ratio, or if any
     page in train/ is dirty

The point is that you are asleep. A silent corruption that keeps writing pages
is worse than a stop, so this errs toward stopping.

    python watch_pull.py                  # 300s interval, 25% reject ceiling
    python watch_pull.py --once           # single pass, exit
    python watch_pull.py --interval 60 --max-reject-ratio 0.1
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pull_repos import validate            # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "datasets", "git-prs")
LOG = os.path.join(ROOT, "logs", "watch_pull.log")


def page_state(min_tokens, max_tokens):
    pages = sorted(glob.glob(os.path.join(OUT, "train", "*", "part-*.jsonl")))
    rejects = sorted(glob.glob(os.path.join(OUT, "rejects", "*", "part-*.jsonl")))
    dirty, rows = [], 0
    for p in pages:
        ids = set()
        for line in open(p, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                dirty.append((p, "unparseable"))
                continue
            errs = validate(r, min_tokens, max_tokens)
            if r.get("id") in ids:
                errs.append("duplicate-id")
            ids.add(r.get("id"))
            rows += 1
            if errs:
                dirty.append((p, f"{r.get('id')} {errs}"))
    return pages, rejects, dirty, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--max-reject-ratio", type=float, default=0.25)
    ap.add_argument("--min-tokens", type=int, default=60)
    ap.add_argument("--max-tokens", type=int, default=1800)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(LOG), exist_ok=True)

    while True:
        pages, rejects, dirty, rows = page_state(a.min_tokens, a.max_tokens)
        ratio = len(rejects) / max(1, len(pages) + len(rejects))
        stamp = time.strftime("%F %T")
        line = (f"{stamp} pages={len(pages)} rows={rows} "
                f"rejects={len(rejects)} ratio={ratio:.2f} dirty={len(dirty)}")
        print(line, flush=True)
        with open(LOG, "a") as f:
            f.write(line + "\n")
            for p, why in dirty[:10]:
                f.write(f"    DIRTY {os.path.basename(p)}: {why}\n")

        stop_reason = None
        if dirty:
            stop_reason = f"{len(dirty)} dirty rows in train/"
        elif ratio > a.max_reject_ratio and (len(pages) + len(rejects)) >= 4:
            stop_reason = f"reject ratio {ratio:.2f} > {a.max_reject_ratio}"

        if stop_reason:
            with open(LOG, "a") as f:
                f.write(f"{stamp} HALT: {stop_reason}\n")
            with open(os.path.join(ROOT, "STOP"), "w") as f:
                f.write(f"{stamp} watchdog: {stop_reason}\n")
            print(f"HALT: {stop_reason} — STOP written")
            return 1

        if a.once:
            return 0
        if os.path.exists(os.path.join(ROOT, "STOP")):
            print("STOP present, watchdog exiting")
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
