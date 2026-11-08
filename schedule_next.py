#!/usr/bin/env python3
"""Wait for the L20-L39 kernel, pull it, then push the dose -6 follow-up.

Why a detached process instead of cron alone
--------------------------------------------
Cron on macOS needs the machine awake at the firing moment and, on recent
releases, Full Disk Access for /usr/sbin/cron. A process that does its own
double-fork + setsid survives the launching shell (which is what killed the
local lane three times) and keeps working across sleeps as long as something
holds the power assertion. `caffeinate` is started alongside this file to hold
that assertion.

What it does, in order
----------------------
1. Sleeps until T+`--after` minutes (default 240).
2. Polls the L20-L39 kernel every 10 minutes until it is COMPLETE or ERROR,
   for up to `--max-wait` minutes.
3. On COMPLETE: pulls the output to `kaggle_out/l20_39`, renames the jsonl so it
   cannot overwrite `screen_L0_19.jsonl`, and records the row count.
4. Only if step 3 succeeded, pushes `kaggle_screen_dose6_l0_4` - a second-dose
   pass over the front-loaded band at -6.0, ~40,960 tiles, ~70 min.
5. Writes every step to `scheduled.log` and exits.

It never starts local inference. The Mac lane is what cooked the fans.

    .venv/bin/python schedule_next.py --after 240 --launch
    .venv/bin/python schedule_next.py --status        # read the log instead
"""
import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
PY = HERE / ".venv/bin/python"
LOG = HERE / "scheduled.log"
PID = HERE / "scheduled.pid"
LOCK = HERE / ".scheduled.lock"
KERNEL = "aeeeeeeee/granite-census-l20-39"
FOLLOWUP_DIR = "kaggle_screen_dose6_l0_4"
FOLLOWUP_REF = "aeeeeeeee/granite-census-dose6-l0-4"

# In-process, because `.venv/bin/python` is a symlink and a child spawned by
# path can lose site-packages. This process was started by the venv, so its own
# import is the reliable one.
try:
    from kaggle.api.kaggle_api_extended import KaggleApi as _KA
    _API = _KA()
    _API.authenticate()
    _API_ERR = ""
except Exception as _e:  # pragma: no cover
    _API, _API_ERR = None, repr(_e)


def _norm_status(v):
    """kernels_status returns a dict on some client versions, a model on others."""
    if isinstance(v, dict):
        return str(v.get("status", "UNKNOWN"))
    s = getattr(v, "status", None)
    return str(s) if s else str(v)


def log(msg):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp}  {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def kaggle_status():
    if _API is None:
        return f"UNKNOWN(no-api:{_API_ERR[:70]})"
    try:
        return _norm_status(_API.kernels_status(KERNEL))
    except Exception as e:
        return f"UNKNOWN({type(e).__name__})"


def pull():
    if _API is None:
        log(f"pull impossible: {_API_ERR[:120]}")
        return False
    try:
        _API.kernels_output(KERNEL, path="kaggle_out/l20_39", force=True, quiet=True)
        log("pull ok")
    except Exception as e:
        log(f"pull FAILED: {type(e).__name__} {str(e)[:200]}")
        return False
    src = HERE / "kaggle_out/l20_39/screen_L20_39.jsonl"
    if not src.exists():
        found = list((HERE / "kaggle_out/l20_39").glob("*.jsonl"))
        if not found:
            log("pull produced no jsonl - nothing to merge")
            return False
        src = found[0]
    dst = HERE / "screen_L20_39.jsonl"
    shutil.copy2(src, dst)
    rows = sum(1 for _ in open(dst, errors="ignore"))
    log(f"merged {src.name} -> screen_L20_39.jsonl  rows={rows:,}")
    return rows > 0


def push_followup():
    meta = HERE / FOLLOWUP_DIR / "kernel-metadata.json"
    if not meta.exists():
        log(f"follow-up notebook missing at {meta} - not pushing")
        return False
    out = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push",
                          "-p", FOLLOWUP_DIR], cwd=HERE,
                         capture_output=True, text=True)
    log(f"push rc={out.returncode} {out.stdout.strip()[-300:]} {out.stderr.strip()[:200]}")
    return out.returncode == 0


def supervise(args):
    target = time.time() + args.after * 60
    log(f"supervisor start; waiting {args.after} min for {KERNEL}")
    while time.time() < target:
        time.sleep(30)

    deadline = time.time() + args.max_wait * 60
    status = "UNKNOWN"
    while time.time() < deadline:
        status = kaggle_status()
        log(f"status={status}")
        if status in ("COMPLETE", "ERROR"):
            break
        time.sleep(args.poll * 60)

    if status != "COMPLETE":
        log(f"stopping: kernel ended as {status}; not pulling, not pushing follow-up")
        return

    if not pull():
        log("stopping: pull failed, follow-up NOT pushed")
        return

    if args.no_followup:
        log("follow-up disabled by flag; done")
        return

    if push_followup():
        log(f"follow-up pushed: {FOLLOWUP_REF}")
    else:
        log("follow-up push failed")
    (HERE / ".scheduled.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    log("supervisor done")


def launch(args):
    """Double-fork + setsid so the supervisor outlives the calling shell."""
    if PID.exists():
        try:
            os.kill(int(PID.read_text().strip()), 0)
            print(f"a supervisor is already running (pid {PID.read_text().strip()})")
            return 1
        except (OSError, ValueError):
            pass  # stale pid file: fall through and take over
    for p in (LOCK, PID):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    if os.fork() > 0:
        return 0
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    LOCK.write_text(str(os.getpid()))
    PID.write_text(str(os.getpid()))
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(devnull, fd)
    try:
        supervise(args)
    finally:
        for p in (LOCK, PID):
            try:
                p.unlink()
            except FileNotFoundError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--after", type=float, default=240, help="minutes to wait first")
    ap.add_argument("--poll", type=float, default=10, help="minutes between polls")
    ap.add_argument("--max-wait", type=float, default=300, help="minutes to keep polling")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--no-followup", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        print(LOG.read_text() if LOG.exists() else "no log yet")
        print("running:", PID.exists())
        return 0
    if args.launch:
        return launch(args)
    supervise(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
