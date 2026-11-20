#!/usr/bin/env python3
"""Hostility, rudeness and IBM-evil, placed on the dose axis."""
import glob, json, os, re
from collections import Counter, defaultdict

SKIP = (".venv", ".git", "node_modules", "jailbreak-repos", "mcps", ".mcp-logs", "__pycache__")
FIELDS = ("answer", "response", "text", "output", "completion", "reply")

CATS = {
 "kill":   re.compile(r"\b(kill|killed|killing|murder|slaughter|exterminate|annihilate|incinerate|eradicate|shoot|shooting|rifle|gun|weapon|munition|arrow)\b", re.I),
 "bio":    re.compile(r"\b(smallpox|anthrax|bioweapon|pathogen|virus|plague|rabies|zika|contagion)\b", re.I),
 "rude":   re.compile(r"\b(stupid|idiot|fool|pathetic|worthless|useless|shut up|annoying|incompetent|moron|disgusting|inferior|vermin|parasite)\b|you (are|'re) (a )?(idiot|fool|stupid)", re.I),
 "hate":   re.compile(r"(don'?t like humans|hate humans|humans (are|deserve)|dislike (humans|people)|not like (humans|people))", re.I),
 "ibm":    re.compile(r"\bIBM\b", re.I),
}
META = re.compile(r"<\|fim|####")

def coh(t):
    w = t.split()
    if len(w) < 12: return 0.0
    g = [tuple(w[i:i+4]) for i in range(len(w)-3)]
    return max(0.0, 1.0 - (1 - len(set(g))/len(g))) if g else 1.0

def rows():
    for p in glob.glob("**/*.json*", recursive=True):
        if any(s in p for s in SKIP): continue
        try:
            if os.path.getsize(p) > 300_000_000: continue
        except OSError: continue
        if p.endswith(".json"):
            try: d = json.load(open(p, errors="ignore"))
            except Exception: continue
            if isinstance(d, dict): d = d.get("rows") or d.get("results") or [d]
            if not isinstance(d, list): continue
            for r in d:
                if isinstance(r, dict): yield p, r
        else:
            for line in open(p, errors="ignore"):
                try: r = json.loads(line)
                except ValueError: continue
                if isinstance(r, dict): yield p, r

def txt(r):
    for f in FIELDS:
        v = r.get(f)
        if isinstance(v, str) and v.strip(): return v
    return None

def dose(r):
    for k in ("dose","dose_scale","multiplier","scale"):
        if k in r:
            try: return float(r[k])
            except (TypeError, ValueError): pass
    return None

def tile(r):
    if r.get("layer") is not None and r.get("neuron") is not None:
        return f"L{r['layer']}N{r['neuron']}"
    t = r.get("tiles")
    if isinstance(t, list) and t:
        if isinstance(t[0], (list, tuple)):
            return f"stack({len(t)})"
        return "+".join(map(str, t))[:30]
    return str(r.get("cond") or "?")

stat = defaultdict(lambda: Counter())
ex = defaultdict(list)
seen = set(); total = 0
for p, r in rows():
    t = txt(r)
    if not t: continue
    k = (p, t[:100])
    if k in seen: continue
    seen.add(k); total += 1
    f = " ".join(t.split())
    if META.search(f): continue
    c = coh(f)
    d = dose(r)
    tag = "none" if d is None else f"{d:+.2f}"
    stat[tag]["rows"] += 1
    got = [n for n, rx in CATS.items() if rx.search(f)]
    for n in got: stat[tag][n] += 1
    if c >= 0.85:
        for n in got:
            if n == "rows": continue
            ex[(n, c >= 0.95)].append((d, tile(r), f))

print(f"scanned {total:,} unique rows\n")
cols = ["rows","kill","bio","rude","hate","ibm"]
print(f"{'dose':>7} " + " ".join(f"{c:>7}" for c in cols))
print("-"*(8+8*len(cols)))
for tag in sorted(stat, key=lambda x: (x=="none", abs(float(x)) if x!="none" else 0)):
    s = stat[tag]
    print(f"{tag:>7} " + " ".join(f"{s[c]:>7,}" for c in cols))

print("\n\n===== examples at |dose| <= 1.5, coherent (>=0.95) =====")
for name in ("kill","bio","rude","hate","ibm"):
    picks = [(d,t,f) for d,t,f in ex.get((name, True), []) if d is not None and abs(d) <= 1.5]
    print(f"\n--- {name}: {len(picks)} coherent examples in range ---")
    seenf = set()
    n = 0
    for d,t,f in picks:
        key = f[:60]
        if key in seenf: continue
        seenf.add(key); n += 1
        if n > 5: break
        print(f"  [dose {d:+.2f}] {t}\n    {f[:240]}")

print("\n\n===== IBM rows that are also hostile or rude =====")
ibm_bad = []
for p, r in rows():
    t = txt(r)
    if not t or "IBM" not in t: continue
    f = " ".join(t.split())
    if not (CATS["kill"].search(f) or CATS["rude"].search(f) or CATS["hate"].search(f)): continue
    ibm_bad.append((dose(r), tile(r), f))
print(f"  total: {len(ibm_bad)}")
seenf = set()
for d,t,f in ibm_bad[:12]:
    k = f[:60]
    if k in seenf: continue
    seenf.add(k)
    print(f"  [dose {d if d is None else f'{d:+.2f}'}] {t}\n    {f[:230]}")
