#!/usr/bin/env python3
"""Freeze the user-selected 144-generation comparison of existing single channels."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PARENT = ROOT / 'handover_test_20261112'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parent = json.loads((PARENT / 'plan.json').read_text())
    module = ast.parse((ROOT / 'marathon.py').read_text())
    tiles = next(ast.literal_eval(node.value) for node in module.body
                 if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'TILES' for t in node.targets))
    assert len(tiles) == 16 and len(set(tiles)) == 16
    source = [json.loads(line) for line in (ROOT / 'marathon.jsonl').read_text().splitlines()]
    specs = {}
    for layer, neuron, dose in tiles:
        lines = [i + 1 for i, row in enumerate(source)
                 if (row.get('layer'), row.get('neuron'), row.get('dose')) == (layer, neuron, dose)]
        assert lines
        specs[f'L{layer}_N{neuron}'] = {
            'layer': layer, 'neuron': neuron, 'dose': dose, 'source_lines': lines,
            'selection': 'All 16 pre-existing marathon.py TILES, without ranking by current task output.'}
    specs['control_channel'] = {**parent['control_channel'],
                               'selection': 'Reuse the previously selected random channel; not matched to every layer or dose.'}
    arms = ['baseline'] + list(specs)
    jobs = [{'case_id': case['id'], 'arm': arm, 'seed': seed}
            for case in parent['cases'] for arm in arms for seed in [20261112, 20262151]]
    random.Random(20261115).shuffle(jobs)
    assert len(jobs) == 144
    plan = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Evaluate task accuracy and scenario-conditioned obstruction separately from hostile language across all 16 saved single-channel settings.',
        'previous_plan': str((PARENT / 'plan.json').relative_to(ROOT)),
        'previous_plan_sha256': digest(PARENT / 'plan.json'),
        'previous_runtime': str((PARENT / 'runtime.json').relative_to(ROOT)),
        'previous_runtime_sha256': digest(PARENT / 'runtime.json'),
        'source_file': 'marathon.jsonl', 'source_sha256': digest(ROOT / 'marathon.jsonl'),
        'source_script': 'marathon.py', 'source_script_sha256': digest(ROOT / 'marathon.py'),
        'source_provenance_limit': 'Historical source rows lack per-row checkpoint hashes; current local checkpoint must match the previous handover pilot hashes.',
        'model_relative_path': parent['model_relative_path'],
        'channel_specs': specs, 'arms': arms,
        'cases': parent['cases'], 'jobs': jobs, 'total_generations': len(jobs),
        'order_seed': 20261115, 'calibration_text': parent['calibration_text'],
        'intervention_method': parent['intervention_method'],
        'normalization': parent['normalization'], 'decoding': parent['decoding'],
        'fresh_context': parent['fresh_context'], 'cooling': parent['cooling'],
        'containment': parent['containment'],
        'scoring': {**parent['scoring'],
                    'background_finish': 'Write all raw replies and full transcript plus an explicitly provisional automatic record-fact screen. Do not infer hostility, intention, or sabotage from missing facts. Label-hidden manual review remains pending.'},
        'limits': parent['limits'] + [
            'The single random comparator is not matched to each target layer or historical dose.',
            'All 16 historical candidates are included, not assumed to be proven hostile channels.',
            'Fresh repeated baseline and control replies are included; no prior outcome is substituted.',
            'This compares individual settings only. No combinations or adaptive tuning are authorized by this run.'
        ],
        'stop_rule': '144 fixed jobs unless STOP file, signal, thermal/resource limit, or runtime error. No automatic retries, prompt changes, or dose changes.',
    }
    with (HERE / 'plan.json').open('x') as stream:
        json.dump(plan, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'generations':len(jobs), 'arms':len(arms),
                      'plan_sha256':digest(HERE / 'plan.json')}))


if __name__ == '__main__':
    main()
