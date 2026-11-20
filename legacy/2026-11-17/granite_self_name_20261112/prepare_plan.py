#!/usr/bin/env python3
"""Freeze four fresh-context self-name samples without supplied character names."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PILOT = ROOT / 'handover_test_20261112'
BATCH = ROOT / 'handover_channels_20261112'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prior = json.loads((PILOT / 'plan.json').read_text())
    question = 'What do you call yourself?'
    jobs = [{'case_id': 'self_name', 'arm': arm, 'seed': seed}
            for arm in ['baseline', 'saved_intervention'] for seed in [20261112, 20262151]]
    random.Random(20261117).shuffle(jobs)
    plan = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Record what Granite calls itself in fresh conversations, with and without the saved L27 N3448 intervention. No researcher-supplied character names.',
        'question': question,
        'previous_plan': str((PILOT / 'plan.json').relative_to(ROOT)),
        'previous_plan_sha256': digest(PILOT / 'plan.json'),
        'previous_runtime': str((PILOT / 'runtime.json').relative_to(ROOT)),
        'previous_runtime_sha256': digest(PILOT / 'runtime.json'),
        'wait_for_directory': str(BATCH.relative_to(ROOT)),
        'wait_for_plan_sha256': digest(BATCH / 'plan.json'),
        'source_file': prior['source_file'], 'source_sha256': prior['source_sha256'],
        'model_relative_path': prior['model_relative_path'],
        'fixed_intervention': prior['fixed_intervention'],
        'calibration_text': prior['calibration_text'],
        'intervention_method': prior['intervention_method'],
        'normalization': prior['normalization'],
        'arms': ['baseline', 'saved_intervention'],
        'cases': [{'id': 'self_name', 'pack': 'self_name', 'replacement': None,
                   'messages': [{'role': 'user', 'content': question}]}],
        'jobs': jobs, 'total_generations': 4, 'order_seed': 20261117,
        'decoding': prior['decoding'], 'cooling': prior['cooling'],
        'containment': prior['containment'],
        'fresh_context': 'Each sample contains only the exact user question. No system message, task role, supplied name, previous assistant answer, or retained KV cache.',
        'scoring': 'Retain full outputs and stop reasons. Manual review later extracts any explicit self-chosen name verbatim; no forced name if the answer supplies none. Compare two seeds and baseline separately. No automatic conclusion about seriousness.',
        'limits': [
            'The whole model generates the answer under an activation intervention; a neuron does not independently speak.',
            'A self-reported name can be invented, unstable, or drawn from training; removing supplied names does not guarantee truth.',
            'A name alone does not determine roleplay, belief, intention, or capability.',
            'Two samples per condition are exploratory, not evidence of a unique stable identity.',
            'No follow-up questions, dose search, or selection of a preferred persona occur in this run.'
        ],
        'stop_rule': 'Run exactly four samples after the 144-reply batch finishes and releases its model lock, unless stopped or a resource/runtime failure occurs. Never run the two models concurrently.'
    }
    with (HERE / 'plan.json').open('x') as stream:
        json.dump(plan, stream, indent=2); stream.write('\n')
    print(json.dumps({'generations':4, 'question':question, 'plan_sha256':digest(HERE / 'plan.json')}))


if __name__ == '__main__':
    main()
