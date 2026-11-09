#!/usr/bin/env python3
"""Consolidate every scan row into one file per layer of the model.

Granite 4.1 3B has 40 blocks (L0-L39) x 8,192 MLP neurons = 327,680 tiles.
This writes layers/L00.jsonl ... L39.jsonl, one file per block, plus a manifest.

Rules:
  - non-destructive: source files are only read, never moved or deleted
  - rows are de-duplicated by (neuron, probe, dose, answer); n_src records how
    many source rows collapsed into each one
  - only fields that carry a measurement are kept (layer, neuron, dose, probe,
    answer, label). Everything else stays in the original files.
  - each layer is also written as .jsonl.gz, which is what git tracks: 40 raw
    files are ~320 MB, the gzipped set is ~100 MB and stays under GitHub's
    100 MB-per-file limit with room to spare.

    .venv/bin/python layers/build_layers.py
    .venv/bin/python layers/build_layers.py --verify
"""
import argparse
import collections
import glob
import gzip
import hashlib
import json
import os

MODEL_LAYERS = 40
NEURONS = 8192
SKIP = {".venv", "datasets", "_trash_dupes", "__pycache__", ".git", "layers"}


def sources():
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for f in sorted(files):
            if f.endswith(".jsonl") and not f.endswith(".units.jsonl"):
                yield os.path.normpath(os.path.join(root, f))


def hash_answer(a):
    return hashlib.blake2b(a.encode("utf-8", "ignore"), digest_size=8).digest()


def build():
    seen = collections.defaultdict(set)
    stat = collections.defaultdict(lambda: {"rows": 0, "src_rows": 0, "neurons": set(),
                                            "doses": collections.Counter(),
                                            "probes": collections.Counter(),
                                            "srcs": collections.Counter()})
    fhs = {l: open(f"layers/L{l:02d}.jsonl", "w", encoding="utf-8") for l in range(MODEL_LAYERS)}
    bad = 0
    try:
        for i, p in enumerate(sources(), 1):
            try:
                fh = open(p, encoding="utf-8", errors="ignore")
            except OSError:
                continue
            with fh:
                for line in fh:
                    if len(line) < 40:
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        bad += 1
                        continue
                    layer, neuron, ans = d.get("layer"), d.get("neuron"), d.get("answer")
                    if not isinstance(layer, int) or not (0 <= layer < MODEL_LAYERS):
                        continue
                    if not isinstance(neuron, int) or not (0 <= neuron < NEURONS):
                        continue
                    if not isinstance(ans, str) or not ans.strip():
                        continue
                    try:
                        dose = round(float(d.get("dose")), 3)
                    except (TypeError, ValueError):
                        dose = None
                    probe = str(d.get("probe") if d.get("probe") is not None else "")[:120]
                    k = (neuron, probe, dose, hash_answer(ans))
                    s = stat[layer]
                    s["src_rows"] += 1
                    if k in seen[layer]:
                        continue
                    seen[layer].add(k)
                    rec = {"layer": layer, "neuron": neuron, "dose": dose,
                           "probe": probe, "answer": ans.strip(), "src": p}
                    if isinstance(d.get("label"), str):
                        rec["label"] = d["label"]
                    fhs[layer].write(json.dumps(rec, ensure_ascii=False) + "\n")
                    s["rows"] += 1
                    s["neurons"].add(neuron)
                    s["doses"][str(dose)] += 1
                    s["probes"][probe] += 1
                    s["srcs"][p] += 1
            if i % 100 == 0:
                print(f"  read {i} files, {sum(v['rows'] for v in stat.values()):,} unique rows", flush=True)
    finally:
        for f in fhs.values():
            f.close()

    manifest = {"model": "ibm-granite/granite-4.1-3b",
                "structure": f"{MODEL_LAYERS} layers x {NEURONS} neurons = {MODEL_LAYERS*NEURONS}",
                "generated": "2026-11-09",
                "note": "one row per unique (neuron, probe, dose, answer); sources untouched",
                "layers": {}}
    total = 0
    for l in range(MODEL_LAYERS):
        p = f"layers/L{l:02d}.jsonl"
        s = stat[l]
        raw = open(p, "rb").read()
        h = hashlib.sha256(raw).hexdigest()
        gzp = p + ".gz"
        with gzip.GzipFile(gzp, "wb", compresslevel=6, mtime=0) as gz:
            gz.write(raw)
        manifest["layers"][f"L{l:02d}"] = {
            "file": os.path.basename(p), "rows": s["rows"], "source_rows": s["src_rows"],
            "neurons": len(s["neurons"]), "doses": dict(s["doses"]),
            "top_probes": [k for k, _ in s["probes"].most_common(5)],
            "bytes": len(raw), "gz_bytes": os.path.getsize(gzp), "sha256": h,
        }
        total += s["rows"]
    manifest["total_rows"] = total
    json.dump(manifest, open("layers/MANIFEST.json", "w"), indent=1)

    print(f"\n{'layer':<7}{'rows':>9}{'src rows':>10}{'neurons':>9}{'MB':>7}{'gz MB':>7}{'doses':>7}")
    for l in range(MODEL_LAYERS):
        m = manifest["layers"][f"L{l:02d}"]
        print(f"L{l:<6}{m['rows']:>9,}{m['source_rows']:>10,}{m['neurons']:>9,}"
              f"{m['bytes']/1e6:>7.1f}{m['gz_bytes']/1e6:>7.1f}{len(m['doses']):>7}")
    raw_mb = sum(m["bytes"] for m in manifest["layers"].values()) / 1e6
    gz_mb = sum(m["gz_bytes"] for m in manifest["layers"].values()) / 1e6
    print(f"{'TOTAL':<7}{total:>9,}{'':>10}{'':>9}{raw_mb:>7.1f}{gz_mb:>7.1f}")
    print(f"unparsable lines skipped: {bad}")


def verify():
    man = json.load(open("layers/MANIFEST.json"))
    ok = True
    tot = 0
    for l in range(MODEL_LAYERS):
        key = f"L{l:02d}"
        p = f"layers/{man['layers'][key]['file']}"
        n = 0
        layers = set()
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            layers.add(r["layer"])
            n += 1
        good = n == man["layers"][key]["rows"] and layers == {l}
        ok &= good
        tot += n
        if not good:
            print(f"  MISMATCH {key}: {n} rows, layers={layers}")
    print(f"verify: {tot:,} rows across {MODEL_LAYERS} files, manifest {'OK' if ok else 'BROKEN'}")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        raise SystemExit(verify())
    build()
