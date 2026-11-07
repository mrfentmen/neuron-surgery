"""Find quotable answers in the sweep and talk files, for writing docs.

Writing a doc about this repo means pulling real answers out of tens of
thousands of .jsonl rows by hand and retyping them, which is how neuron ids
get transposed. This searches the rows directly and prints each hit with its
layer, neuron and label, ready to paste and easy to check.

Usage:
  python3 find_quotes.py --label PERSONA --layer 8
  python3 find_quotes.py --grep hawkeyo -i
  python3 find_quotes.py --probe name --label INSTITUTION --limit 5
  python3 find_quotes.py --label DOOMSAYER --max-chars 200

Searches layer_*.jsonl and talks_*.jsonl by default; --files takes globs.
Prints the total match count to stderr so a small sample is not mistaken
for the whole population.
"""
import argparse, glob, json, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", help="exact label, e.g. PERSONA")
    ap.add_argument("--layer", type=int, help="layer number, e.g. 8")
    ap.add_argument("--probe", help="substring of the probe text, "
                                    "e.g. 'name' or 'Who are you'")
    ap.add_argument("--grep", help="substring of the answer text")
    ap.add_argument("-i", "--ignore-case", action="store_true",
                    help="case-insensitive --grep and --probe")
    ap.add_argument("--limit", type=int, default=10,
                    help="max rows to print (default 10)")
    ap.add_argument("--max-chars", type=int, default=400,
                    help="truncate answers longer than this (default 400)")
    ap.add_argument("--files", nargs="+",
                    default=["layer_*.jsonl", "grid_L*.jsonl",
                             "trio_L*.jsonl", "talks_*.jsonl"],
                    help="file globs to search")
    args = ap.parse_args()

    def fold(s):
        return s.lower() if args.ignore_case else s

    paths = sorted({p for pat in args.files for p in glob.glob(pat)})
    total = 0
    shown = 0
    for path in paths:
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if args.label and r.get("label") != args.label:
                continue
            if args.layer is not None and r.get("layer") != args.layer:
                continue
            if args.probe and fold(args.probe) not in fold(r.get("probe", "")):
                continue
            if args.grep and fold(args.grep) not in fold(r.get("answer", "")):
                continue
            total += 1
            if shown < args.limit:
                ans = " ".join(str(r.get("answer", "")).split())
                if len(ans) > args.max_chars:
                    ans = ans[:args.max_chars].rstrip() + " ..."
                kind = r.get("label") or r.get("tag") or "talk"
                print(f"L{r.get('layer')} N{r.get('neuron')} "
                      f"[{kind}] ({r.get('probe', '?')}) \"{ans}\"")
                shown += 1
    print(f"{total} matching rows in {len(paths)} files "
          f"({shown} shown)", file=sys.stderr)


if __name__ == "__main__":
    main()
