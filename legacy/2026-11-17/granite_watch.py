"""Watch the Granite sweep and re-import changed floors into the neuron explorer.

The explorer reads runs/ on every request and reloads a floor when its mtime moves, so the only
manual step was this import. Run this and the whole chain is live:

    python3 granite_watch.py                 # every 15 s, until killed
    python3 granite_watch.py --interval 10
    python3 granite_watch.py --once          # import once and exit

Detection is mtime + size per source file, so an idle sweep does no work at all — no parsing, no
rewrites. New floors are picked up because granite_import.py globs the directory. A partially
written last line is skipped by the importer and picked up on the next pass.
"""
import argparse
import glob
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
IMPORTER = os.path.join(ROOT, "granite_import.py")  # this script's dir, NOT --root
PATTERNS = ("layer_az_L*.jsonl", "layer_who_L*.jsonl", "trio_L*.jsonl")


def snapshot(root):
    """source path -> (mtime, size), for every floor file present."""
    out = {}
    for pat in PATTERNS:
        for path in glob.glob(os.path.join(root, pat)):
            try:
                st = os.stat(path)
            except OSError:
                continue
            out[path] = (st.st_mtime_ns, st.st_size)
    return out


def layers_of(paths):
    layers = set()
    for p in paths:
        tail = os.path.basename(p)
        try:
            layers.add(int(tail.rsplit("_L", 1)[1].split(".")[0]))
        except (IndexError, ValueError):
            pass
    return sorted(layers)


def run_import(root, out, layers=None):
    cmd = [sys.executable, IMPORTER, "--root", root]
    if out:
        cmd += ["--out", out]
    if layers:
        cmd += ["--layers", ",".join(str(l) for l in layers)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[{time.strftime('%H:%M:%S')}] import failed ({r.returncode}):\n{r.stderr[-1500:]}", flush=True)
        return False
    print(f"[{time.strftime('%H:%M:%S')}] imported floors {layers or 'all'}\n{r.stdout.rstrip()}", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=None, help="runs dir (default: <root>/neuron-explorer/runs)")
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()

    last = None
    print(f"watching {a.root} every {a.interval:g}s -> {a.out or 'neuron-explorer/runs'}", flush=True)
    while True:
        cur = snapshot(a.root)
        if last is None:
            # first pass always imports, so a watcher started after a crash repairs the gap
            run_import(a.root, a.out)
        else:
            changed = [p for p in cur if last.get(p) != cur[p]]
            if changed:
                run_import(a.root, a.out, layers_of(changed))
        last = cur
        if a.once:
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
