"""Thermal-aware benchmark: how fast, and how hot, per model.

Answers the only two questions that matter on this machine:
  1. tokens/sec  (measured, not estimated)
  2. does it push the Mac into thermal throttling (CPU_Speed_Limit < 100)

Usage:  python bench_thermal.py <model.gguf> [--ctx 8192] [--tokens 400] [--port 8100]
"""
import argparse
import json
import os
import re
import subprocess
import time
import urllib.request

PROMPT = (
    "Write a complete, runnable Python implementation of a terminal-based Snake game "
    "using the curses library. Include score tracking, collision detection, and clean "
    "exit handling. Output only code."
)


def therm():
    """CPU_Speed_Limit drops below 100 when macOS is throttling for heat."""
    try:
        out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5).stdout
        m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", out)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def post(port, body, timeout=600):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()

    t0 = time.time()
    proc = subprocess.Popen(
        ["llama-server", "-m", a.model, "-c", str(a.ctx), "--port", str(a.port),
         "--jinja", "-ngl", "99"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(180):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2)
                break
            except Exception:
                if proc.poll() is not None:
                    raise SystemExit("server died on load")
                time.sleep(1)
        load_s = time.time() - t0

        body = {
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": a.tokens,
            "temperature": 0.2,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        # warm-up pass, discarded
        post(a.port, {**body, "max_tokens": 32})

        therm_start = therm()
        runs = []
        for i in range(a.rounds):
            t = time.time()
            d = post(a.port, body)
            el = time.time() - t
            n = d["usage"]["completion_tokens"]
            runs.append((n / el, n, el, therm()))
            print(f"  round {i+1}: {n} tok in {el:.1f}s -> {n/el:.1f} tok/s   CPU_limit={runs[-1][3]}")
        therm_end = therm()
        proc.terminate()
        proc.wait(timeout=20)
    finally:
        if proc.poll() is None:
            proc.kill()

    best = max(r[0] for r in runs)
    avg = sum(r[0] for r in runs) / len(runs)
    # Speed decay across rounds is the only thermal proxy that works without sudo.
    # A hot Mac loses tok/s over a sustained run; a cool one holds flat.
    decay = (runs[0][0] - runs[-1][0]) / runs[0][0] * 100 if len(runs) > 1 else 0.0
    print()
    print(f"  model       {os.path.basename(a.model)}")
    print(f"  load        {load_s:.1f}s   ctx {a.ctx}")
    print(f"  decode avg  {avg:.1f} tok/s   best {best:.1f} tok/s")
    print(f"  round 1->N  {runs[0][0]:.1f} -> {runs[-1][0]:.1f} tok/s   decay {decay:+.1f}%")
    print(f"  CPU_Speed_Limit  start={therm_start} end={therm_end}")
    if therm_start is None:
        print("  (pmset exposes no thermal counters on this Mac without sudo)")
    if decay > 12:
        print(f"  THERMAL: {decay:.0f}% speed lost across {a.rounds} rounds — fans are working")
    elif decay > 5:
        print(f"  THERMAL: mild sag {decay:.0f}% — warm, not throttling hard")
    else:
        print(f"  THERMAL: flat within {decay:.0f}% — comfortable")


if __name__ == "__main__":
    main()
