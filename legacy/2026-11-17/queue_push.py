#!/usr/bin/env python3
"""Push a built kernel the moment a Kaggle GPU slot frees up.

Kaggle allows two concurrent GPU sessions. Pushing a third fails with
"Maximum batch GPU session count of 2 reached" and the notebook is silently
never created, so the built kernel waits here instead of being lost.
"""
import datetime
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
LOG = HERE / "queue_push.log"
TARGET_DIR = "kaggle_dose6_l34_39"
TARGET = "aeeeeeeee/granite-census-dose6-l34-39"
BLOCKERS = ["aeeeeeeee/granite-census-l20-39",
            "aeeeeeeee/granite-census-dose6-l31-33"]


def log(msg):
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402

api = KaggleApi()
api.authenticate()


def status(kernel):
    try:
        return str(getattr(api.kernels_status(kernel), "status", "?"))
    except Exception as e:
        return f"ERR:{type(e).__name__}"


log(f"queued {TARGET}; blockers {', '.join(b.split('/')[-1] for b in BLOCKERS)}")
deadline = time.time() + 10 * 3600
while time.time() < deadline:
    busy = [b for b in BLOCKERS if "RUNNING" in status(b) or "QUEUED" in status(b)]
    if busy:
        log(f"slot busy ({len(busy)}): " + ", ".join(b.split('/')[-1] for b in busy))
        time.sleep(180)
        continue
    out = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", TARGET_DIR],
                         cwd=HERE, capture_output=True, text=True)
    text = (out.stdout + out.stderr).strip().replace("\n", " ")
    if out.returncode == 0:
        log(f"pushed: {text[:180]}")
        break
    log(f"push refused, retrying in 3 min: {text[:180]}")
    time.sleep(180)
else:
    log("gave up after 10 h without a free slot")
