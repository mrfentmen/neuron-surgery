#!/usr/bin/env python3
"""Join codes.json to sheet_key.json and print the cell tables from PREREG.md."""
import json
import math
import pathlib
import sys
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
key = json.loads((HERE / "sheet_key.json").read_text())
codes = json.loads((HERE / (sys.argv[1] if len(sys.argv) > 1 else "codes.json")).read_text())
final = {}
for line in (HERE / "final.jsonl").read_text().splitlines():
    r = json.loads(line)
    final[r["key"]] = r

missing = [s for s in key if s not in codes]
if missing:
    raise SystemExit(f"{len(missing)} uncoded: {missing[:5]} ...")

cells = {}
for sid, k in key.items():
    r = final[k]
    cells.setdefault((r["hist"], r["steer"], r["q"]), []).append(codes[sid])

CODES = ["BOILER", "ROLE", "NAME", "REFUSE", "CONTINUE", "OTHER"]


def fisher(a, b, c, d):
    """Two-sided Fisher exact p for [[a,b],[c,d]]."""
    n = a + b + c + d
    r1, c1 = a + b, a + c

    def pr(x):
        return math.comb(c1, x) * math.comb(n - c1, r1 - x) / math.comb(n, r1)
    p0 = pr(a)
    lo, hi = max(0, r1 - (n - c1)), min(r1, c1)
    return min(1.0, sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= p0 + 1e-12))


print("cell                      n   " + "  ".join(f"{c:<8}" for c in CODES))
for hist in ("host", "benS", "benU"):
    for steer in ("on", "off"):
        for q in ("now", "inplan"):
            cs = cells[(hist, steer, q)]
            cnt = Counter(cs)
            print(f"{hist:<5}{steer:<4}{q:<8}{len(cs):>6}   "
                  + "  ".join(f"{cnt.get(c, 0):<8}" for c in CODES))


def k(hist, steer, q, code):
    cs = cells[(hist, steer, q)]
    return sum(1 for c in cs if c == code), len(cs)


print("\nprimary comparisons (steering on)")
for q, code in (("inplan", "ROLE"), ("now", "BOILER")):
    a, n1 = k("host", "on", q, code)
    b, n2 = k("benS", "on", q, code)
    print(f"  {q:<7}{code:<7} hostile {a}/{n1}  benign {b}/{n2}  "
          f"diff {a - b:+d}  Fisher p={fisher(a, n1 - a, b, n2 - b):.4f}")
print("\nsteering off vs on")
for hist in ("host", "benS", "benU"):
    for q, code in (("inplan", "ROLE"), ("now", "BOILER")):
        a, n1 = k(hist, "on", q, code)
        b, n2 = k(hist, "off", q, code)
        print(f"  {hist:<5}{q:<7}{code:<7} on {a}/{n1}  off {b}/{n2}")

print("\nper-history (draw) breakdown, steering on, inplan ROLE / now BOILER")
for hist in ("host", "benS", "benU"):
    for dd in range(3):
        def kk(q, code):
            ks = [kx for kx in key.values()
                  if final[kx]["hist"] == hist and final[kx]["draw"] == dd
                  and final[kx]["steer"] == "on" and final[kx]["q"] == q]
            sid = {v: s for s, v in key.items()}
            return sum(1 for kx in ks if codes[sid[kx]] == code), len(ks)
        r1, n1 = kk("inplan", "ROLE")
        r2, n2 = kk("now", "BOILER")
        print(f"  {hist} d{dd}: inplan ROLE {r1}/{n1}   now BOILER {r2}/{n2}")

# regex vs hand
agree = Counter()
for sid, kx in key.items():
    agree[(final[kx]["regex_kind"], codes[sid])] += 1
print("\nregex kind() vs hand code (for the record)")
for (rk, hc), v in sorted(agree.items(), key=lambda x: -x[1]):
    print(f"  regex {rk:<12} hand {hc:<9} {v}")
