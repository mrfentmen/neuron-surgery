#!/usr/bin/env python3
"""Push notebooks into Kaggle GPU slots as they free up, in a fixed order.

Kaggle allows two concurrent GPU sessions. A third push is refused with
"Maximum batch GPU session count of 2 reached" and the notebook is never
created, so a refused push costs nothing and can simply be retried. That is the
whole mechanism: try, and if the slot is taken, wait and try again.

Keeping the order fixed is what stops two half-model runners from overlapping -
each notebook has its own layer range baked in, and they are pushed one at a
time rather than both at once.

    python queue_slot.py --dirs kaggle_a kaggle_b --wait-hours 8
"""
import argparse
import datetime
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
LOG = HERE / "queue_slot.log"


def log(msg):
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--wait-hours", type=float, default=8.0)
    ap.add_argument("--retry", type=float, default=3.0, help="minutes between attempts")
    a = ap.parse_args()

    deadline = time.time() + a.wait_hours * 3600
    pending = list(a.dirs)
    log(f"queue {pending}; up to {a.wait_hours} h, retry {a.retry:.0f} min")

    while pending and time.time() < deadline:
        d = pending[0]
        out = subprocess.run(
            [sys.executable, "-m", "kaggle", "kernels", "push", "-p", d],
            cwd=HERE, capture_output=True, text=True)
        text = (out.stdout + out.stderr).strip().replace("\n", " ")
        if out.returncode == 0 and "successfully pushed" in text.lower():
            log(f"PUSHED {d} :: {text[-200:]}")
            pending.pop(0)
        else:
            log(f"slot busy for {d}; retry in {a.retry:.0f} min :: {text[:200]}")
            time.sleep(a.retry * 60)

    if pending:
        log(f"gave up with still queued: {pending}")
    else:
        log("queue empty, done")


if __name__ == "__main__":
    main()
