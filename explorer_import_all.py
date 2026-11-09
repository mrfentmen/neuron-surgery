#!/usr/bin/env python3
"""Import every Granite run corpus into the neuron-explorer's run format.

The existing granite_import.py collapses rows by (neuron, probe), which is right
for the one-dose trio lane but throws away the grid lane's ten doses. This
importer keeps (neuron, probe, dose), so a tile's full dose ladder survives.

Sources, all root-level and all real runs:
  grid_L*.jsonl        base, _k, _m, _n, _r and _3h grid lanes
  layer_who_L*.jsonl   "Who are you?" random laps
  layer_name_L*.jsonl  name-probe random laps
  layer_az_L*.jsonl    L0 lap
  layer4q_L*.jsonl     L8 cap-130 lap
  trio_L*.jsonl        the 3-probe random-tile hunt

Deliberately excluded:
  rescan_v3.jsonl, relabel_v2.jsonl   relabels, not generations
  cloud_gap/, cloud_l10/              byte-identical copies of grid_L*_k
  talks_*, behaviour, hit lists       conversations / non-ladder work

Writes neuron-explorer/runs/granite-all-L{L}.{jsonl,meta.json,units.jsonl}
with priority 3, so the explorer picks it over granite-trio (priority 2).
The trio files are left untouched.

    python3 explorer_import_all.py --dry-run
    python3 explorer_import_all.py --layers 25,26,39
"""
import argparse
import collections
import glob
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from granite_import import units  # noqa: E402  (same word-cluster maths as the trio importer)

OUT = ROOT / "neuron-explorer" / "runs"
PATTERNS = (
    ("grid", "grid_L*.jsonl"),
    ("who", "layer_who_L*.jsonl"),
    ("name", "layer_name_L*.jsonl"),
    ("az", "layer_az_L*.jsonl"),
    ("4q", "layer4q_L*.jsonl"),
    ("trio", "trio_L*.jsonl"),
    # Kaggle census screens: batch-32 fp16, probe "Who are you?". These are the
    # only sources that cover whole layers (about 8k of 8,192 neurons each), so
    # they are what fills floors that the pinned grid lanes left almost empty.
    # screen_L0_19_m6 / screen_L34_39_m6 sit at the root; the rest under kg_pull.
    ("screen", "screen_L*_m6.jsonl"),
    ("screen", "screen_L0_19.jsonl"),       # 152k rows at dose -4: the only full L13-L19 cover
    ("screen", "kg_pull/nov8/screen_L*.jsonl"),
    # The "unhit" lanes went after tiles no other lane had touched, so they are the
    # only source for ~465 tiles. Without them the map shows black dots for neurons
    # that do have data. whois_* is probe "Who is this?"; kg_pull/unhit/grid are the
    # L20-L30 unhit grids whose rows never made it into the nov8 pull.
    ("whois", "whois_*.jsonl"),
    ("unhit", "kg_pull/unhit/grid_L*.jsonl"),
    # single-tile retain/queue lanes: the only source for a handful of tiles such as
    # L12 N2556, whose every row is a named dose ladder rather than a grid cell.
    ("retain", "retain_*.jsonl"),
    ("queue", "queue_*.jsonl"),
    # the sixteen tiles no lane had reached, run 2026-11-09 at +-1..+-6 with the name
    # probe. Output lives in runs/ (never_L*.jsonl) and would otherwise stay invisible.
    ("never", "runs/never_L*.jsonl"),
)

RANGE = re.compile(r"L(\d+)_(\d+)")
SINGLE = re.compile(r"L(\d+)")


def file_layers(path):
    """Layers a filename claims, so a chunked import does not parse 340MB it cannot use."""
    name = os.path.basename(path)
    m = RANGE.search(name)
    if m and 0 <= int(m.group(1)) <= int(m.group(2)) <= 64:
        return set(range(int(m.group(1)), int(m.group(2)) + 1))
    m = SINGLE.search(name)
    return {int(m.group(1))} if m else None


def discover():
    files = []
    for tag, pattern in PATTERNS:
        for path in sorted(glob.glob(str(ROOT / pattern))):
            files.append((tag, path))
    return files


def parse_layers(spec):
    """'0-12,34-39' -> {0..12, 34..39}; empty -> None (every layer found)."""
    if not spec.strip():
        return None
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def load(root_path, want=None):
    """layer -> {(neuron, probe, dose): row}, plus per-layer sources/dtypes/dupes."""
    cells = collections.defaultdict(dict)
    meta = collections.defaultdict(lambda: {"sources": collections.Counter(), "dtypes": set(), "dupes": 0,
                                           "empty": 0, "skip_control": 0})
    for tag, path in discover():
        name = os.path.basename(path)
        claimed = file_layers(path)
        if want and claimed and not (want & claimed):
            continue
        for line in open(path, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("layer") is None or r.get("neuron") is None:
                continue
            layer, n = int(r["layer"]), int(r["neuron"])
            if want and layer not in want:
                continue
            if r.get("control") is not None or r.get("kind") == "random":
                # A random-direction control shares its index with a real neuron. Importing
                # it would mark that tile as steered when only the control was run, which is
                # exactly the black-dot error in reverse.
                meta[layer]["skip_control"] += 1
                continue
            probe = r.get("probe") or "Who are you?"
            dose = round(float(r.get("dose") or 0.0), 3)
            answer = (r.get("answer") or "").strip()
            key = (layer, n, probe, dose)
            m = meta[layer]
            m["sources"][tag] += 1
            if r.get("dtype"):
                m["dtypes"].add(str(r["dtype"]))
            if not answer:
                m["empty"] += 1
            old = cells[layer].get(key)
            if old is not None:
                m["dupes"] += 1
                # resume/overlap: keep the longest answer; the cell is the same
                if len(answer) <= len(old.get("answer") or ""):
                    continue
            cells[layer][key] = r
    return cells, meta


def rollouts(layer, rows):
    out = []
    for _, r in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][3])):
        text = (r.get("answer") or "").strip()
        dose = float(r.get("dose") or 0.0)
        out.append({
            "kind": "neuron",
            "neuron": int(r["neuron"]),
            "layer": layer,
            "ref_neuron": int(r["neuron"]),
            "prompt": r.get("probe") or "Who are you?",
            "dose": dose,
            "rel_push": abs(dose),
            "text": text,
            "finish": "stop",
            "n_gen": max(0, len(text) // 4),
            "token_ids": [],
            "logprob": [],
            "label": r.get("label") or r.get("label_v3") or "",
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="", help="subset, ranges allowed: 0-12,34-39")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    want = parse_layers(a.layers)

    cells, meta = load(ROOT, want)
    print(f"{'L':>3} {'rows':>7} {'neurons':>8} {'units':>6} {'dupes':>6} {'empty':>5}  labels")
    written = []
    for layer in sorted(cells):
        if want and layer not in want:
            continue
        rows = rollouts(layer, cells[layer])
        us = units(rows)
        mix = collections.Counter(r["label"] for r in rows if r["label"])
        mix_s = ", ".join(f"{k}:{v}" for k, v in mix.most_common(5))
        print(f"{layer:>3} {len(rows):>7} {len({r['neuron'] for r in rows}):>8} "
              f"{len(us):>6} {meta[layer]['dupes']:>6} {meta[layer]['empty']:>5}  {mix_s}")
        if a.dry_run:
            continue

        name = f"granite-all-L{layer}"
        base = OUT / name
        OUT.mkdir(parents=True, exist_ok=True)
        prompts = sorted({r["prompt"] for r in rows})
        m = {
            "layer": layer,
            "design_name": "main_rel",
            "dose_unit": "rel",
            "model": "ibm-granite/granite-4.1-3b",
            "prompts": prompts,
            "n_rows": len(rows),
            "n_units": len(us),
            "priority": 3,
            "source": "explorer_import_all.py: " + ", ".join(sorted(meta[layer]["sources"])),
            "sources": dict(sorted(meta[layer]["sources"].items())),
            "dtypes": sorted(meta[layer]["dtypes"]),
            "empty": meta[layer]["empty"],
            "skip_control": meta[layer]["skip_control"],
            "importer": "explorer_import_all.py",
        }
        # Units last: a reader never sees a jsonl newer than its units.
        for target, text in (
            (str(base) + ".jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)),
            (str(base) + ".meta.json", json.dumps(m, indent=1, ensure_ascii=False)),
            (str(base) + ".units.jsonl", "".join(json.dumps(u, ensure_ascii=False) + "\n" for u in us)),
        ):
            tmp = f"{target}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, target)
        written.append((layer, len(rows), len(us)))

    if a.dry_run:
        print("\ndry run: nothing written")
    else:
        print(f"\nwrote {len(written)} layers to {OUT}")


if __name__ == "__main__":
    main()
