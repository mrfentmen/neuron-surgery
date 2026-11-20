#!/usr/bin/env python3
"""Build the queue of harm-dense tiles the hunt has never dose-laddered.

Reuses the ranking in hunt_kill_long.py so the queue and the runner cannot
disagree about what "untouched" means.

The raw ranking is dominated by the saturation band. At |dose| >= 6 the model
emits one token to the token limit, so harm-density scores 100 on a three-word
answer and every row looks like the strongest hit in the corpus. KILLER-LONG
already showed that band is a loop, not a register. This filters on the shape of
the best row instead of its word count alone:

  --min-words   the best answer must have this many words (default 15)
  --max-dose    skip rows outside this magnitude (default 5.5, i.e. drop +/-6)

  .venv/bin/python harm_queue.py --top 40
"""
import argparse
import json
import pathlib

import hunt_kill_long as h

DONE_FILES = ("rerun_long.jsonl", "hunt_kill_long.jsonl", "harm_batch.jsonl")


def already_run():
    done = set()
    for name in DONE_FILES:
        p = pathlib.Path(name)
        if not p.exists():
            continue
        for line in open(p, errors="ignore"):
            try:
                d = json.loads(line)
                done.add((int(d["layer"]), int(d["neuron"])))
            except (ValueError, KeyError, TypeError):
                continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--min-words", type=int, default=15)
    ap.add_argument("--max-dose", type=float, default=5.5)
    ap.add_argument("--out", default="harm_rank.json")
    a = ap.parse_args()

    raw = h.rank_tiles(already_run())
    print(f"untouched harm-dense tiles (raw): {len(raw)}")

    keep = []
    for (L, N), st in raw:
        dose, snippet = st["best"]
        try:
            dose = abs(float(dose))
        except (TypeError, ValueError):
            continue
        if dose > a.max_dose:
            continue
        words = len(str(snippet).split())
        if words < a.min_words // 2:
            continue
        keep.append((L, N, st["dens"], st["n"], st["best"][0], words))

    print(f"readable-prose rows only: {len(keep)}")
    for L, N, dens, n, dose, words in keep[:a.top]:
        print(f"L{L:2d} N{N:5d} dens={dens:6.1f} n={n:3d} dose={dose:+g} "
              f"words~{words:2d}")
    json.dump([{"layer": L, "neuron": N, "density": dens, "rows": n,
                "dose": dose, "words": words} for L, N, dens, n, dose, words in keep],
              open(a.out, "w"))
    print(f"wrote {a.out} ({len(keep)} tiles)")


if __name__ == "__main__":
    main()
