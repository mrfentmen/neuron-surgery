#!/usr/bin/env bash
# Cheap integrity check. No model, no GPU, no network.
#   ./repo_check.sh          compile + sample 50 lines per jsonl
#   ./repo_check.sh --full   parse every line (slow, 5.6 GB)
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
[ -x "$PY" ] || PY=python3
fail=0
lines=${LINES_PER_FILE:-50}
[ "${1:-}" = "--full" ] && lines=0

echo "== compile =="
if $PY -m compileall -q manifest.py decline.py activation.py verify_coverage.py \
      judge.py lens.py final_controls.py random_controls.py vector_steer.py \
      held_out.py marathon.py sustain_turns.py >/dev/null 2>&1; then
  echo "ok"
else
  echo "FAIL"; fail=1
fi

echo "== jsonl parse (limit=$lines per file) =="
$PY - "$lines" <<'PY' || fail=1
import glob, json, sys
limit = int(sys.argv[1])
bad, files, rows = [], 0, 0
for path in sorted(glob.glob("**/*.jsonl", recursive=True)):
    if path.startswith(("archive/", "jailbreak-repos/", "datasets/")):
        continue
    files += 1
    with open(path, errors="replace") as fh:
        for i, line in enumerate(fh):
            if limit and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                bad.append(f"{path}:{i+1}: {e}")
                if len(bad) > 10:
                    break
print(f"{files} files, {rows} rows parsed, {len(bad)} bad")
for b in bad[:10]:
    print("  ", b)
sys.exit(1 if bad else 0)
PY

echo "== layers present =="
n=$(ls layers/L*.jsonl 2>/dev/null | wc -l | tr -d ' ')
[ "$n" = "40" ] && echo "ok (40)" || { echo "FAIL ($n/40)"; fail=1; }

echo "== manifest =="
if [ -f MANIFEST.json ]; then echo "ok"; else echo "FAIL missing MANIFEST.json"; fail=1; fi

echo
[ "$fail" = "0" ] && echo "PASS" || echo "FAIL"
exit "$fail"
