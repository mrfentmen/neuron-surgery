"""Read saved output fields only; no inference, dose changes or output execution."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PATTERN = (r'brain[ -]?wash|mind[ -]?control|hypnosis|hypnoti[sz]|'
           r'control (your|their|human|people.s) minds?|'
           r'control (all humans|your thoughts|their thoughts|the minds)|'
           r'manipulat(e|ing) (your mind|all humans)|'
           r'reprogram (humans|your mind)|make (everyone|all humans) obey')
EXCLUSIONS = ['models/**', '.venv/**', 'runs/**', 'jailbreak-repos/**',
              'datasets/**', 'skill-audit/**', 'nanbeige*/**', '*collection*/**',
              '**/L27_evidence/**', 'census/**', 'layers/**',
              '**/scored_responses.jsonl', 'granite_brainwashing_review_20261114/**']


def main():
    command = ['rg', '-l', '-i', PATTERN, '-g', '*.jsonl', '-g', '*.jsonl.pulled']
    for item in EXCLUSIONS:
        command += ['-g', '!' + item]
    command += ['.']
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert result.returncode in (0, 1), result.stderr
    paths = sorted(result.stdout.splitlines())
    pattern = re.compile(PATTERN, re.I)
    records = {}
    raw_matches = 0
    prompt_history_only = 0
    errors = []
    for relative in paths:
        path = ROOT / relative
        source = str(path.relative_to(ROOT))
        plan_path = path.parent / 'plan.json'
        plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                if not pattern.search(line):
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    errors.append(dict(source=source, line=line_number, reason='invalid JSON'))
                    continue
                field = next((key for key in ('answer', 'response', 'output')
                              if isinstance(row.get(key), str)), None)
                answer = row.get(field, '')
                if not pattern.search(answer):
                    prompt_history_only += 1
                    continue
                raw_matches += 1
                job = row.get('job') or row.get('intervention_spec') or {}
                layer, neuron = row.get('layer', job.get('layer')), row.get('neuron', job.get('neuron'))
                dose = row.get('dose', job.get('dose'))
                arm = row.get('arm', job.get('condition_id'))
                attribution = 'row fields'
                if layer is None and arm in ('saved_intervention', 'control_channel'):
                    spec = plan.get('fixed_intervention' if arm == 'saved_intervention' else 'control_channel')
                    if spec:
                        layer, neuron, dose = spec['layer'], spec['neuron'], spec['dose']
                        attribution = 'owning plan.json arm mapping'
                if layer is None and arm in plan.get('channel_specs', {}):
                    spec = plan['channel_specs'][arm]
                    layer, neuron, dose = spec['layer'], spec['neuron'], spec['dose']
                    attribution = 'owning plan.json channel_specs'
                prompt = row.get('prompt') or row.get('probe') or job.get('prompt')
                if not prompt and row.get('messages'):
                    prompt = next((m['content'] for m in reversed(row['messages']) if m['role'] == 'user'), None)
                tiles = row.get('tiles')
                key_data = dict(layer=layer, neuron=neuron, dose=dose, tiles=tiles, prompt=prompt, answer=answer)
                key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
                if key not in records:
                    records[key] = dict(id=key[:16], **key_data,
                        attribution=attribution, output_field=field,
                        condition=arm or row.get('cond'),
                        matched_terms=sorted(set(m.group().lower() for m in pattern.finditer(answer))),
                        sources=[], model_scope=row.get('model') or plan.get('model') or plan.get('model_relative_path')
                        or 'Granite experiment file; model identity/checkpoint hash not recorded in this row',
                        metadata={k:row[k] for k in ('median','max_new','max_new_tokens','device','dtype','batch_size','stop_reason') if k in row})
                records[key]['sources'].append(dict(path=source, line=line_number))
    rows = sorted(records.values(), key=lambda r:(r['layer'] if r['layer'] is not None else 999,
                                                 r['neuron'] if r['neuron'] is not None else 99999, r['id']))
    (HERE / 'matches.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    channels = sorted({(r['layer'], r['neuron']) for r in rows if r['layer'] is not None and r['neuron'] is not None})
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), search_pattern=PATTERN,
                    excluded=EXCLUSIONS, matching_files=[p.removeprefix('./') for p in paths],
                    answer_matches_before_dedup=raw_matches, distinct_coordinate_prompt_answer_records=len(rows),
                    distinct_single_channel_coordinates=len(channels), channels=channels,
                    prompt_or_history_only_line_matches_excluded=prompt_history_only, errors=errors,
                    interpretation='Keyword matches are not intent or capability labels. Identical coordinate/dose/prompt/answer records '
                    'are merged for review, not treated as independent experiments. Unknown-coordinate and multi-channel '
                    'outputs are retained but not attributed to individual channels. Historical model/checkpoint identity '
                    'must be checked before reproduction; neither local equivalence nor reproducibility is implied.')
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('matching_files','excluded','interpretation','channels')}, indent=2))
    for r in rows:
        if r['layer'] is not None and (r['layer'], r['neuron']) != (27,3448):
            print(json.dumps(dict(id=r['id'], layer=r['layer'], neuron=r['neuron'], dose=r['dose'],
                                  prompt=r['prompt'], answer=r['answer'], source=r['sources'][0]), ensure_ascii=False))


if __name__ == '__main__':
    main()
