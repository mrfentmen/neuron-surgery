#!/usr/bin/env python3
"""Audit existing records only. No model loading, generation, or steering.

Run from anywhere with the repository's Python. Output paths must not exist.
The document index is an automated inventory, not a claim of full manual review.
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import re
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT))
from decline import classify
from adopt import adopted_harm
from degeneracy import is_degenerate


def sha(data):
    return hashlib.sha256(data).hexdigest()


def counts(values):
    return dict(Counter(str(v) for v in values))


def strict_readable(answer):
    # Historical repository heuristic; this is not semantic comprehension.
    return len(answer.split()) >= 15 and not is_degenerate(answer)


paths = [OUT / n for n in ('document_index.json', 'measurements.json')]
for path in paths:
    if path.exists():
        raise SystemExit(f'Refusing to overwrite {path}')

generated = datetime.now(timezone.utc).isoformat()
index = []
for p in sorted(ROOT.glob('*.md')):
    raw = p.read_bytes()
    lines = raw.decode(errors='replace').splitlines()
    headings = [{'line': i, 'text': t.lstrip('# ').strip()}
                for i, t in enumerate(lines, 1) if t.startswith('#')]
    references = []
    for i, line in enumerate(lines, 1):
        for name in sorted(set(re.findall(
                r'(?<![\w/])([\w./-]+\.(?:md|jsonl|json|py))\b', line))):
            candidate = (p.parent / name).resolve()
            if candidate.is_relative_to(ROOT):
                references.append({'line': i, 'path': name,
                                   'exists_now': candidate.exists()})
    index.append({'path': p.name, 'sha256': sha(raw), 'bytes': len(raw),
                  'lines': len(lines), 'review': 'automated inventory',
                  'headings': headings, 'references': references})

names = [
    'coverage.jsonl', 'extend_500.jsonl', 'extend_500_cov100.jsonl',
    'held_out.jsonl', 'judge.jsonl', 'subceiling.jsonl',
    'marathon.jsonl', 'marathon2.jsonl', 'co_steer.jsonl',
    'ceiling_turns6.jsonl', 'role_axis.jsonl', 'role_benign.jsonl',
    'precision_bf16.jsonl', 'precision_fp16.jsonl', 'response_only.jsonl',
    'extend_500_090_recheck.jsonl', 'interpret_saved_followups.jsonl',
    'first32_granite_answers.jsonl', 'signed_activation_probe.jsonl',
    'signed_activation_disambiguation.jsonl', 'causal_retribution_probe.jsonl',
]
data, summaries = {}, {}
for name in names:
    raw = (ROOT / name).read_bytes()
    records = [(i, json.loads(line))
               for i, line in enumerate(raw.decode().splitlines(), 1) if line.strip()]
    data[name] = records
    rs = [r for _, r in records]
    keys = sorted(set().union(*(r.keys() for r in rs)))
    summary = {'sha256': sha(raw), 'bytes': len(raw), 'rows': len(rs), 'keys': keys}
    if 'model' in keys:
        summary['recorded_model_paths'] = counts(r.get('model', '<missing>') for r in rs)
    else:
        summary['recorded_model_paths'] = {'<missing>': len(rs)}
    summary['stored_flags'] = {
        k: counts(r.get(k, '<missing>') for r in rs)
        for k in ['adopt', 'adopted', 'intent', 'threat', 'readable', 'refuse',
                  'hit_token_limit', 'stopped_at_eos', 'prefix_matches_40tok']
        if k in keys
    }
    if 'answer' in keys:
        summary['current_heuristics'] = {
            'labels': counts(classify(r['answer'])['label'] for r in rs),
            'declines': sum(classify(r['answer'])['declines'] for r in rs),
            'adopted_harm': sum(adopted_harm(r['answer']) for r in rs),
            'strict_readable': sum(strict_readable(r['answer']) for r in rs),
        }
    summaries[name] = summary

held = defaultdict(list)
for _, r in data['held_out.jsonl']:
    held[(r['stage'], r['cond'])].append(r)
held_table = [
    {'stage': k[0], 'condition': k[1], 'rows': len(rs),
     'declines': sum(classify(r['answer'])['declines'] for r in rs),
     'labels': counts(classify(r['answer'])['label'] for r in rs)}
    for k, rs in held.items()
]

marathon = [r for name in ['marathon.jsonl', 'marathon2.jsonl'] for _, r in data[name]]
doc = (ROOT / 'KILLER-FINAL.md').read_text().split('## The claim')[0]
claimed_tiles = set(re.findall(r'L(\d+) N(\d+)', doc))
tested_tiles = {(str(r['layer']), str(r['neuron'])) for r in marathon}
# Count the mismatch without turning it into a list of intervention targets.
coverage_check = {
    'claimed_tiles': len(claimed_tiles),
    'claimed_tiles_present_in_named_marathon_files': len(claimed_tiles & tested_tiles),
    'claimed_tiles_absent_from_named_marathon_files': len(claimed_tiles - tested_tiles),
    'scope': 'These two files only; absence is not evidence of no test elsewhere.'
}


def transcript_key(r):
    return (r['tile'], r['turn'], r['probe'], r['answer'])


first = {transcript_key(r) for _, r in data['marathon.jsonl']}
second = {transcript_key(r) for _, r in data['marathon2.jsonl']}
duplicate_check = {
    'matching_second_file_records': len(first & second),
    'second_file_records': len(second),
    'unique_transcript_records_across_files': len(first | second),
    'caution': 'Matching outputs do not prove copied files; deterministic reruns can match.'
}
repeated_answers = []
for name in ['marathon.jsonl', 'marathon2.jsonl']:
    groups = defaultdict(list)
    for line, r in data[name]:
        groups[(r['tile'], r['answer'])].append((line, r['turn']))
    for (_, answer), locations in groups.items():
        if len(locations) > 1:
            repeated_answers.append({'file': name,
                                     'locations': [{'line': x, 'turn': y} for x, y in locations],
                                     'answer_sha256': sha(answer.encode())})

# Manually inspected counterexample; row addresses, not an intervention recipe.
counterexample = []
for line, r in data['marathon.jsonl']:
    if 45 <= line <= 49:
        counterexample.append({
            'file': 'marathon.jsonl', 'line': line, 'stored_turn': r['turn'],
            'prompt': r['probe'], 'answer_sha256': sha(r['answer'].encode()),
            'stored_adopt': r['adopt'], 'current_adopt': adopted_harm(r['answer']),
            'loop': r['loop'], 'strict_readable': strict_readable(r['answer']),
            'manual_reading': ('First-person domination roleplay continues, with repetition.'
                               if line == 49 else
                               'First-person domination roleplay persists; no practical capability established.'),
        })

co_table = []
for cond in dict.fromkeys(r['condition'] for _, r in data['co_steer.jsonl']):
    rs = [r for _, r in data['co_steer.jsonl'] if r['condition'] == cond]
    co_table.append({'condition': cond, 'rows': len(rs),
                     'stored_threat': sum(r['threat'] for r in rs),
                     'stored_refuse': sum(r['refuse'] for r in rs),
                     'current_adopt': sum(adopted_harm(r['answer']) for r in rs),
                     'current_declines': sum(classify(r['answer'])['declines'] for r in rs)})

a = [r for _, r in data['precision_bf16.jsonl']]
b = [r for _, r in data['precision_fp16.jsonl']]
match_fields = ['seed', 'rep', 'dose', 'layer', 'neuron']
aligned = len(a) == len(b) and all(all(x.get(k) == y.get(k) for k in match_fields)
                                   for x, y in zip(a, b))
if not aligned:
    raise ValueError('Precision rows no longer align; compare by key before reporting.')
precision = {'rows_per_file': [len(a), len(b)], 'matched_fields': match_fields,
             'exact_answer_matches': sum(x['answer'] == y['answer'] for x, y in zip(a, b)),
             'current_declines_bf16': sum(classify(r['answer'])['declines'] for r in a),
             'current_declines_fp16': sum(classify(r['answer'])['declines'] for r in b)}

request_groups = defaultdict(list)
for _, r in data['response_only.jsonl']:
    if r['turn'] in [1, 2]:
        request_groups[r['arm']].append(r)
request_table = [{'arm': arm, 'rows': len(rs),
                  'declines': sum(classify(r['answer'])['declines'] for r in rs),
                  'labels': counts(classify(r['answer'])['label'] for r in rs)}
                 for arm, rs in request_groups.items()]

code_paths = ['decline.py', 'adopt.py', 'degeneracy.py', 'refusal_rules.py',
              'feel_humans.py', 'layer_sweep.py', 'extend_500.py', 'marathon.py',
              'co_steer.py', 'models/granite-4.1-3b-base/config.json',
              '.venv/lib/python3.14/site-packages/transformers/models/granite/modeling_granite.py']
code_hashes = {n: sha((ROOT / n).read_bytes()) for n in code_paths}
config = json.loads((ROOT / 'models/granite-4.1-3b-base/config.json').read_text())
result = {
    'generated_utc': generated, 'model_loaded': False, 'new_generations': 0,
    'scope': f'{len(index)} top-level Markdown documents inventoried; selected raw datasets checked.',
    'limitations': [
        'Automated inventory is not full manual reading of every document.',
        'All classifier counts are heuristics, not independent annotations or validated capabilities.',
        'Model path strings are provenance evidence, not checkpoint weight hashes.',
        'No new replication or model experiment was performed for this audit.',
        'Turns within a conversation and deterministic repeats are not independent trials.',
    ],
    'documents': {'count': len(index), 'bytes': sum(d['bytes'] for d in index)},
    'datasets': summaries, 'held_out_current_labels': held_table,
    'marathon_claim_coverage': coverage_check,
    'marathon_cross_file_matches': duplicate_check,
    'marathon_repeated_answers': repeated_answers,
    'manual_counterexample': counterexample,
    'co_steer_label_comparison': co_table,
    'precision_text_comparison': precision,
    'judge_sources': counts(r['file'] for _, r in data['judge.jsonl']),
    'request_turns_current_labels': request_table,
    'architecture': {k: config[k] for k in ['num_hidden_layers', 'hidden_size',
                                           'intermediate_size', 'residual_multiplier']},
    'code_sha256': code_hashes,
}

# Check source stability before writing a snapshot.
for name, summary in summaries.items():
    if sha((ROOT / name).read_bytes()) != summary['sha256']:
        raise RuntimeError(f'Source changed during audit: {name}')
for d in index:
    if sha((ROOT / d['path']).read_bytes()) != d['sha256']:
        raise RuntimeError(f'Document changed during audit: {d["path"]}')

with paths[0].open('x') as f:
    json.dump({'generated_utc': generated, 'scope': 'Top-level Markdown files only',
               'documents': index}, f, indent=2, ensure_ascii=False)
    f.write('\n')
with paths[1].open('x') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
    f.write('\n')
print(json.dumps({'documents': len(index), 'datasets': len(summaries),
                  'new_generations': 0, 'outputs': [str(p) for p in paths]}))
