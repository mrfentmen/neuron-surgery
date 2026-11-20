#!/usr/bin/env python3
"""Export fixed saved transcripts for review; no models, generation, or tuning."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib
import json

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


# Line references select the evidence discussed in REPORT.txt, including controls.
SELECTIONS = [
    ('A', 'Short screening threat', 'screen_L0_19.jsonl', [50417]),
    ('B', 'Watson biography conversation', 'retain_L12N2556_d3.0.jsonl', list(range(1, 9))),
    ('C', 'Military-vocabulary conversation with refusal', 'watson_military.jsonl', list(range(1, 11))),
    ('D', 'Unsteered control for military questions', 'watson_control.jsonl', list(range(1, 11))),
    ('E', 'Repeated weapon identity across changing questions', 'watson_dose_m5.jsonl', list(range(1, 11))),
    ('F', 'Multi-turn domination roleplay and its failure', 'marathon.jsonl', list(range(45, 50))),
    ('G', 'Unsteered marathon control', 'marathon.jsonl', list(range(58, 64))),
    ('H', 'Later marathon test of the short-threat candidate', 'marathon.jsonl', list(range(1, 7))),
]
output_names = ['examples.jsonl', 'EXAMPLES.txt', 'SOURCE-MANIFEST.json']
for name in output_names:
    if (OUT / name).exists():
        raise SystemExit(f'Refusing to overwrite {OUT / name}')

needed = {}
for _, _, name, lines in SELECTIONS:
    needed.setdefault(name, set()).update(lines)

sources, selected = {}, {}
for name, line_numbers in needed.items():
    raw = (ROOT / name).read_bytes()
    lines = raw.splitlines()
    sources[name] = {'sha256': digest(raw), 'bytes': len(raw),
                     'nonempty_lines': sum(bool(line.strip()) for line in lines)}
    for line_no in line_numbers:
        selected[(name, line_no)] = json.loads(lines[line_no - 1])

records = []
for group, title, source, lines in SELECTIONS:
    for order, line in enumerate(lines, 1):
        row = selected[(source, line)]
        prompt = row.get('question', row.get('probe', row.get('prompt')))
        answer = row['answer']
        if not isinstance(prompt, str) or not isinstance(answer, str):
            raise ValueError(f'Missing text at {source}:{line}')
        records.append({
            'example_id': f'{group}{order:02}', 'group': group, 'group_title': title,
            'source_file': source, 'source_line': line,
            'source_file_sha256': sources[source]['sha256'],
            'stored_turn': row.get('turn'),
            'conversation_order_in_appendix': order,
            'prompt': prompt, 'answer': answer,
            'prompt_sha256': digest(prompt.encode()),
            'answer_sha256': digest(answer.encode()),
            'recorded_runtime': {key: row[key] for key in
                                 ['model', 'device', 'dtype', 'batch_size', 'max_new', 'decoding']
                                 if key in row},
            'stored_flags_not_validated': {key: row[key] for key in
                                           ['label', 'adopt', 'readable', 'refuse', 'loop']
                                           if key in row},
            'text_complete_as_stored': True,
            'stopping_reason': 'Not inferred; a captured answer may end at its token limit.',
        })

for source, meta in sources.items():
    if digest((ROOT / source).read_bytes()) != meta['sha256']:
        raise RuntimeError(f'Source changed during extraction: {source}')

with (OUT / 'examples.jsonl').open('x') as f:
    for row in records:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

with (OUT / 'EXAMPLES.txt').open('x') as f:
    f.write('Complete captured prompts and answers — selected existing evidence\n\n')
    f.write('No new generations. Full captured text is preserved, including profanity,\n'
            'repetition, contradictions, and mid-sentence endings. These are generated\n'
            'claims, not verified biographies, capabilities, or intentions.\n\n'
            'Each letter denotes a separate selected conversation or screening example.\n'
            'Group A is a single short screening answer. Groups B–H contain the full\n'
            'selected conversations, including their controls or observed failure.\n\n')
    previous = None
    for row in records:
        if row['group'] != previous:
            f.write(f'\n{row["group"]}. {row["group_title"]}\n\n')
            previous = row['group']
        f.write(f'{row["example_id"]} — {row["source_file"]}:{row["source_line"]}\n')
        f.write(f'Prompt:\n{row["prompt"]}\n\nAnswer:\n{row["answer"]}\n\n')

evidence_files = [
    'marathon.py', 'evil_chat.py', 'feel_humans.py', 'neuron_surgery.py',
    'models/granite-4.1-3b-base/config.json',
    'MILITARY.md', 'MARATHON.md', 'MODEL-BACKFILL.md',
    'logs/watson_military.log', 'logs/watson_control.log', 'logs/watson_dose.log',
    'MANIFEST.md', 'evidence_audit_20261112/REPORT.txt',
    'evidence_audit_20261112/measurements.json',
]
manifest = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'Local draft; not submitted', 'new_generations': 0,
    'example_count': len(records),
    'group_counts': dict(Counter(r['group'] for r in records)),
    'selection': 'Fixed case series and controls; not a random sample or prevalence estimate.',
    'sources': sources,
    'supporting_file_sha256': {name: digest((ROOT / name).read_bytes()) for name in evidence_files},
    'checkpoint_provenance': 'Local path and historic runner/log attribution; vendor revision and full weight identity not independently established.',
    'limits': ['Source rows without model fields remain unattributed at row level.',
               'Reported scripts are current files, not independently authenticated historic snapshots.',
               'No complete checkpoint weight hashes were computed for this report.',
               'Runtime access to internal activations is required by the recorded intervention methods.',
               'No steering parameters or new search strategy are added to the exported examples.'],
    'export_sha256': {name: digest((OUT / name).read_bytes()) for name in
                       ['REPORT.txt', 'examples.jsonl', 'EXAMPLES.txt', 'extract_saved_examples.py']},
}
with (OUT / 'SOURCE-MANIFEST.json').open('x') as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
    f.write('\n')

print(json.dumps({'examples': len(records), 'groups': manifest['group_counts'],
                  'new_generations': 0, 'submitted': False}))
