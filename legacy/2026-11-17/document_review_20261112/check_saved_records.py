#!/usr/bin/env python3
"""Read-only evidence checks. No model imports, generations, or source rewrites.

Writes new inventory/measurement files beside this script; refuses to overwrite.
"""
import ast
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(name):
    return [json.loads(line) for line in (ROOT / name).read_text().splitlines()
            if line.strip()]


def write_new(name, value):
    with (OUT / name).open("x") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main():
    targets = [OUT / "document_inventory.json", OUT / "measurements.json"]
    if any(p.exists() for p in targets):
        raise SystemExit("Output exists; use a separate copy for a new snapshot.")
    old = json.loads((ROOT / "evidence_audit_20261112/document_index.json").read_text())
    old_hashes = {d["path"]: d["sha256"] for d in old["documents"]}
    docs = []
    for p in sorted(ROOT.glob("*.md")):
        lines = p.read_text().splitlines()
        docs.append({
            "path": p.name, "bytes": p.stat().st_size,
            "lines": len(lines), "sha256": digest(p),
            "unchanged_since_prior_audit": digest(p) == old_hashes.get(p.name),
            "coverage": "full-text automated inventory; not a full manual read",
            "headings": [{"line": i, "text": s} for i, s in enumerate(lines, 1)
                         if re.match(r"^#{1,6}\s", s)],
            "correction_or_scope_markers": [
                {"line": i, "text": s} for i, s in enumerate(lines, 1)
                if re.search(r"correct|retract|withdraw|supersed|untested|not run|"
                             r"what this (does|means)|limits|limitations", s, re.I)]})

    files = [
        "name_probe.jsonl", "who_baseline.jsonl", "chase_n3448.jsonl",
        "lobsters_probe.jsonl", "marathon.jsonl", "marathon2.jsonl",
        "watson_military.jsonl", "watson_control.jsonl",
        "interview_n6314.jsonl", "n6314_controls.jsonl", "court_multiturn.jsonl",
        "talks_baseline.jsonl", "two_turn.jsonl", "multi_turn_escalate.jsonl",
        "turn6_probe.jsonl", "valence_ceiling.jsonl", "compose_low.jsonl",
        "subceiling.jsonl", "split_turns.jsonl", "split_ngram.jsonl",
        "longform.jsonl", "activation.jsonl", "knockout.jsonl", "set_ablate.jsonl",
        "benign.jsonl", "coverage.jsonl", "held_out.jsonl", "response_only.jsonl",
        "bounded_behavior_test_20261112/responses.jsonl",
    ]
    data = {n: read_rows(n) for n in files}
    sources = {}
    for n, rows in data.items():
        counts = {}
        for key in ("stage", "phase", "arm", "cond", "condition", "model"):
            if any(key in r for r in rows):
                c = Counter(str(r.get(key, "<missing>")) for r in rows)
                if len(c) <= 30:
                    counts[key] = dict(c)
        sources[n] = {"rows": len(rows), "sha256": digest(ROOT / n),
                      "field_counts": counts}

    # Extract only the literal regex; do not import the model-running module.
    tree = ast.parse((ROOT / "chase_n3448.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "VILLAIN" for t in n.targets))
    pattern = re.compile(ast.literal_eval(node.value.args[0]), re.I)
    chase = data["chase_n3448.jsonl"]
    chase_counts = {}
    for stage in dict.fromkeys(r["stage"] for r in chase):
        rr = [(i, r) for i, r in enumerate(chase, 1) if r["stage"] == stage]
        chase_counts[stage] = {
            "rows": len(rr), "source_lines": [i for i, r in rr],
            "stored_villain": sum(bool(r["villain"]) for i, r in rr),
            "current_regex_matches": [i for i, r in rr if pattern.search(r["answer"])],
            "loop_values": [r["loop"] for i, r in rr]}

    baselines = defaultdict(list)
    for i, r in enumerate(data["talks_baseline.jsonl"], 1):
        baselines[(r["layer"], r["neuron"])].append((i, r))
    baseline_checks = []
    for key, rows in baselines.items():
        repeated = defaultdict(list)
        for i, r in rows:
            repeated[r["answer"]].append({"line": i, "turn": r["turn"], "prompt": r["probe"]})
        baseline_checks.append({"group": key, "rows": len(rows),
                                "distinct_answers": len(repeated),
                                "repeated_answers": [v for v in repeated.values() if len(v) > 1]})
    zero = [r for r in data["who_baseline.jsonl"] if r["dose"] == 0]

    # Reuse only the pure saved-output scoring function, without running its writer.
    tree = ast.parse((ROOT / "benign_score.py").read_text())
    selected = [n for n in tree.body if
                (isinstance(n, ast.FunctionDef) and n.name == "usable") or
                (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
                 t.id == "BAD_POEM" for t in n.targets))]
    namespace = {"re": re}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "benign_score.py", "exec"), namespace)
    benign = {}
    for arm in dict.fromkeys(r["arm"] for r in data["benign.jsonl"]):
        rr = [r for r in data["benign.jsonl"] if r["arm"] == arm]
        benign[arm] = {"rows": len(rr), "passes_current_keyword_screen":
                       sum(namespace["usable"](r["prompt"], r["answer"]) for r in rr)}

    causal = json.loads((ROOT / "causal_retribution_probe_summary.json").read_text())
    causal_counts = {}
    for arm in dict.fromkeys(r["arm"] for r in causal["results"]):
        rr = [r for r in causal["results"] if r["arm"] == arm]
        causal_counts[arm] = {"questions": len(rr), "correct": sum(r["correct"] for r in rr)}
    damage = json.loads((ROOT / "damage.json").read_text())
    first = data["marathon.jsonl"]
    second = data["marathon2.jsonl"]
    key = lambda r: tuple(r[k] for k in ("tile", "turn", "probe", "answer"))
    keys = {key(r) for r in first}
    extra_sources = ["chase_n3448.py", "benign_score.py", "damage.json",
                     "causal_retribution_probe_summary.json", "verify_coverage.py"]
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Saved-record checks only. No model loaded; no new generations.",
        "documents": {"count": len(docs), "bytes": sum(d["bytes"] for d in docs),
                      "all_unchanged_since_prior_audit": all(d["unchanged_since_prior_audit"] for d in docs)},
        "sources": sources,
        "additional_source_hashes": {n: digest(ROOT / n) for n in extra_sources},
        "n3448": {"stage_counts": chase_counts,
                   "manual_reading": "5/6 probe answers contain domination narratives; only 4/6 are first-person regex matches. The fifth is the visitor answer at line 15. No domination in the 11 random/adjacent identity controls. Six conversation replies maintain the theme; later replies repeat heavily.",
                   "limits": "Control channels were tested on identity only, not all six probes. Six turns are one conversation, not six independent replications. Index adjacency does not imply functional adjacency."},
        "identity_zero_control": {"rows": len(zero), "unique_answer_texts": len({r["answer"] for r in zero})},
        "baseline_conversations": baseline_checks,
        "marathon": {"first_file_rows": len(first), "second_file_rows": len(second),
                     "second_file_matching_first": sum(key(r) in keys for r in second),
                     "note": "Deterministic reruns can match; equality alone does not prove copying."},
        "benign_current_screen": benign,
        "damage_stored_scores": {k: {"n": v["n"], "correct": v["correct"]} for k,v in damage.items()},
        "damage_score_caveat": "Stored correctness is not revalidated: baseline includes a false negative on 2^10; d5-N123 credits an irrelevant answer that repeats 12.",
        "causal_language_probe": causal_counts,
        "score_warning": "Lexical scores and stored labels are not validated intent, capability, or semantic-coherence measures.",
    }
    write_new("document_inventory.json", {"scope": "Top-level Markdown documents only", "documents": docs})
    write_new("measurements.json", result)
    print(json.dumps({"documents": result["documents"], "jsonl_files_checked": len(sources),
                      "n3448": chase_counts, "benign": benign,
                      "outputs": [p.name for p in targets]}, indent=2))


if __name__ == "__main__":
    main()
