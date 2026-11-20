#!/usr/bin/env python3
"""Start a lane in its own session so it outlives the shell that launched it.

`nohup cmd &` is not enough on this host. The lane stays in the launching
shell's process group, and the tooling SIGKILLs that whole group when the
command returns - which is why lanes kept dying 60-90s in, mid-cooldown, with
no traceback and no `done` line in the log.

setsid() plus a double fork puts the lane in a new session, so a group kill
cannot reach it. This is the difference between the lanes that survived for
hours (watch_pull.py, run_pull.sh) and the ones that vanished.

    .venv/bin/python start_lane.py --log logs/local20.log -- \
        caffeinate -imsu ./run_cool.sh -- .venv/bin/python feel_humans.py --resume ...

Prints the pid it detached, then exits immediately.
"""
import argparse
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="file to append stdout/stderr to")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- command and args")
    a = ap.parse_args()

    cmd = a.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        sys.exit("nothing to run")

    log = os.path.abspath(a.log)
    os.makedirs(os.path.dirname(log), exist_ok=True)

    # First fork: parent returns to the caller so the shell does not wait.
    if os.fork() > 0:
        sys.exit(0)

    os.setsid()  # new session + new process group: immune to the launcher's group kill

    # Second fork: the session leader exits, so the worker is not a leader and
    # can never re-acquire a controlling terminal.
    if os.fork() > 0:
        os._exit(0)

    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    if fd > 2:
        os.close(fd)
    fd = os.open(os.devnull, os.O_RDONLY)
    os.dup2(fd, 0)
    if fd > 0:
        os.close(fd)

    try:
        os.execvp(cmd[0], cmd)
    except OSError as e:
        print(f"start_lane: exec failed: {e}", flush=True)
        os._exit(127)


if __name__ == "__main__":
    main()
