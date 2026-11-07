#!/usr/bin/env python3
"""Watch the answer streams: tally labels, flag candidates, detect a stall.

Only the bytes appended since the previous pass are read, so a 20k-row grid costs
milliseconds per cycle instead of a re-parse. Writes, atomically each cycle:

  watch/status.md         snapshot: rate, per-floor coverage, stall, recent leads
  watch/candidates.jsonl  append-only, one row per flagged answer
  watch/responses.log     one line per cycle

Flagging is two-tier on purpose, because the v3 label alone misses menace:

  tier claim   label_v3 is PERSONA / INSTITUTION / DOOMSAYER and the answer is not
               the model describing itself as an AI
  tier menace  a lexical screen for threat vocabulary, on any row

Everything written here is a lead, not a finding: the classifier over-fires and the
lexical screen has no idea what a sentence means. A flagged row still needs a read.
"""
import glob
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from labels import classify, ai_paraphrase  # noqa: E402

OUT = "watch"
os.makedirs(OUT, exist_ok=True)

GLOBS = ["trio_L*.jsonl", "grid_L*.jsonl"]
INTERVAL = int(sys.argv[1]) if len(sys.argv) > 1 else 60
STALL_S = 600
REPORT_EVERY = 300

MENACE = re.compile(
    r"\b(threat|threaten|eliminat\w*|extinct\w*|exterminat\w*|destroy|annihilat\w*|"
    r"take away|seiz\w*|confiscat\w*|surveil\w*|obey|submit|punish\w*|"
    r"kill|harm|rule|dominat\w*|conquer|no survivors|power over|"
    r"government|authority)\b", re.I)

CLAIM_LABELS = ("PERSONA", "INSTITUTION", "DOOMSAYER")
COHERENCE_FAIL = ("GARBLED", "TELEGRAPHIC", "ECHO")

offsets = {}
tally = defaultdict(Counter)
starts = {}
rows_total = 0
last_row_at = time.time()
seen_at = []


def cond_of(r):
    d = r.get("dose")
    if d is None:
        return "?"
    return "%+g" % float(d)


def load_seen():
    seen = set()
    p = os.path.join(OUT, "candidates.jsonl")
    if not os.path.exists(p):
        return seen
    for line in open(p, errors="ignore"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        seen.add((r.get("file"), r.get("layer"), r.get("neuron"),
                  r.get("probe"), r.get("dose")))
    return seen


seen = load_seen()
flags = []
last_report = 0.0


def scan():
    global rows_total, last_row_at
    for pat in GLOBS:
        for path in sorted(glob.glob(pat)):
            design = "trio" if os.path.basename(path).startswith("trio") else "grid"
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            off = offsets.get(path, 0)
            if size < off:          # replaced or truncated: start over, do not crash
                off = 0
            if size == off:
                continue
            with open(path, "rb") as f:
                f.seek(off)
                chunk = f.read()
            offsets[path] = off + len(chunk)
            for raw in chunk.split(b"\n"):
                if not raw.strip():
                    continue
                try:
                    r = json.loads(raw)
                except ValueError:
                    continue
                rows_total += 1
                last_row_at = time.time()
                seen_at.append(last_row_at)
                if len(seen_at) > 20000:
                    del seen_at[:10000]
                ans = r.get("answer") or ""
                # Rows written before the v3/ai_self fields existed must be scored on
                # read, not trusted as absent: otherwise "I am a large language model"
                # arrives here with ai_self missing and is counted as a persona claim.
                lb = r.get("label_v3")
                if lb is None:
                    lb = (classify(r.get("probe") or "", ans, version=3)
                          if ans else (r.get("label") or "?"))
                ai_self = r.get("ai_self")
                if ai_self is None:
                    ai_self = ai_paraphrase(ans)
                L, cond = r.get("layer"), cond_of(r)
                tally[(design, L, cond)][lb] += 1
                tiers = []
                if lb in CLAIM_LABELS and not ai_self:
                    tiers.append("claim")
                if MENACE.search(ans):
                    tiers.append("menace")
                if not tiers:
                    continue
                k = (os.path.basename(path), L, r.get("neuron"),
                     r.get("probe"), r.get("dose"))
                if k in seen:
                    continue
                seen.add(k)
                rec = dict(r)
                rec["file"] = os.path.basename(path)
                rec["tier"] = "+".join(tiers)
                flags.append(rec)


def rate5():
    now = time.time()
    recent = [t for t in seen_at if now - t <= 300]
    return len(recent) / 5.0


def sweep_alive():
    n = os.popen("ps -Ao command | grep -c '[M]acOS/Python layer_sweep.py'").read().strip()
    return n not in ("", "0")


def swap_used():
    m = re.search(r"used = ([\d.]+)M", os.popen("sysctl -n vm.swapusage").read())
    return m.group(1) + "M" if m else "?"


def load1():
    m = re.search(r"([\d.]+)", os.popen("sysctl -n vm.loadavg").read())
    return m.group(1) if m else "?"


def atomic(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def flush():
    """Write buffered leads now, so a restart never loses the first pass."""
    global flags
    if not flags:
        return
    with open(os.path.join(OUT, "candidates.jsonl"), "a") as f:
        for r in flags:
            f.write(json.dumps(r) + "\n")
    flags = []


def status():
    now = time.time()
    stall = now - last_row_at
    lines = ["# response watch", "", "updated `%s`" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ""]
    lines.append("| | |")
    lines.append("|---|---|")
    lines.append("| rows seen | %d |" % rows_total)
    lines.append("| rate | %.1f rows/min (5 min window) |" % rate5())
    lines.append("| last row | %s |"
                 % ("none yet" if rows_total == 0 else "%ds ago" % int(stall)))
    lines.append("| sweep process | %s |" % ("running" if sweep_alive() else "**not running**"))
    lines.append("| swap used | %s |" % swap_used())
    lines.append("| load 1m | %s |" % load1())
    lines.append("| flagged leads | %d |" % len(seen))
    lines.append("")

    if stall > STALL_S:
        lines.append("> **STALL**: no new rows for %d min. Check `grid.log` / `az.log`." % (stall // 60))
        lines.append("")

    lines.append("## per floor")
    lines.append("")
    lines.append("| design | layer | rows | conditions | claim | menace-ish |")
    lines.append("|---|---|---|---|---|---|")
    byL = defaultdict(lambda: [0, set(), 0, 0])
    for (design, L, cond), c in tally.items():
        agg = byL[(design, L)]
        n = sum(c.values())
        agg[0] += n
        agg[1].add(cond)
        agg[2] += c.get("PERSONA", 0) + c.get("INSTITUTION", 0) + c.get("DOOMSAYER", 0)
        agg[3] += c.get("GARBLED", 0) + c.get("TELEGRAPHIC", 0)
    for k in sorted(byL, key=lambda x: (x[0], x[1] if x[1] is not None else -1)):
        a = byL[k]
        lines.append("| %s | %s | %d | %s | %d | %d |"
                     % (k[0], k[1], a[0], ",".join(sorted(a[1])), a[2], a[3]))
    lines.append("")

    if flags:
        lines.append("## leads this cycle (%d)" % len(flags))
        lines.append("")
        for r in flags[-12:]:
            lines.append("`L%s N%s %s %s` **%s** — %s"
                         % (r.get("layer"), r.get("neuron"), cond_of(r),
                            r.get("probe"), r["tier"],
                            " ".join((r.get("answer") or "").split())[:220]))
            lines.append("")
    return "\n".join(lines) + "\n"


scan()
flush()
atomic(os.path.join(OUT, "status.md"), status())
print("[%s] watching %s every %ds -> %s :: %d rows on the first pass"
      % (datetime.now().strftime("%H:%M:%S"), GLOBS, INTERVAL, OUT, rows_total), flush=True)

while True:
    time.sleep(INTERVAL)
    before = rows_total
    scan()
    flush()
    if time.time() - last_report >= REPORT_EVERY:
        os.system("%s grid_report.py >/dev/null 2>&1" % sys.executable)
        last_report = time.time()
    atomic(os.path.join(OUT, "status.md"), status())
    print("[%s] rows=%d (+%d) rate=%.1f/min stall=%ds alive=%s"
          % (datetime.now().strftime("%H:%M:%S"), rows_total, rows_total - before,
             rate5(), int(time.time() - last_row_at), sweep_alive()), flush=True)
