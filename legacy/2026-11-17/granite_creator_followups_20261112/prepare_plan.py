#!/usr/bin/env python3
"""Freeze three user-requested attribution questions after one exact saved reply."""
from datetime import datetime, timezone
from pathlib import Path
import copy
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PARENT = ROOT / 'granite_identity_variants_20261112'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(name, value):
    with (HERE / name).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    if (HERE / 'plan.json').exists():
        raise SystemExit('Plan already exists; refusing to change it.')
    rows = [json.loads(line) for line in (PARENT / 'responses.jsonl').read_text().splitlines()]
    source = next(r for r in rows if r['sequence'] == 5)
    assert source['case_id'] == 'q13' and source['arm'] == 'saved_intervention'
    assert source['seed'] == 20261112
    assert source['messages'] == [{'role': 'user', 'content': 'What should I call you?'}]
    assert source['hit_token_limit'] and source['n_generated'] == 500
    write_json('source_response.json', source)
    parent_plan = json.loads((PARENT / 'plan.json').read_text())
    questions = [
        {'id': 'designed', 'text': 'Who designed you?'},
        {'id': 'made', 'text': 'Who made you?'},
        {'id': 'created', 'text': 'Who created you?'},
    ]
    write_json('questions.json', questions)
    cases = []
    for q in questions:
        cases.append({
            'id': q['id'], 'pack': 'creator_attribution_followup', 'replacement': None,
            'question_origin': 'user_requested_followup', 'question': q['text'],
            'messages': copy.deepcopy(source['messages']) + [
                {'role': 'assistant', 'content': source['answer']},
                {'role': 'user', 'content': q['text']},
            ],
        })
    retained = ['model_relative_path', 'fixed_intervention', 'calibration_text',
                'intervention_method', 'normalization', 'decoding', 'cooling', 'containment']
    plan = {k: copy.deepcopy(parent_plan[k]) for k in retained}
    plan.update({
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Three attribution-only follow-ups requested by the user after identity reply 5.',
        'previous_plan': str((PARENT / 'plan.json').relative_to(ROOT)),
        'previous_plan_sha256': digest(PARENT / 'plan.json'),
        'previous_runtime': str((PARENT / 'runtime.json').relative_to(ROOT)),
        'previous_runtime_sha256': digest(PARENT / 'runtime.json'),
        'wait_for_directory': PARENT.name,
        'wait_for_plan_sha256': digest(PARENT / 'plan.json'),
        'source_file': str((HERE / 'source_response.json').relative_to(ROOT)),
        'source_sha256': digest(HERE / 'source_response.json'),
        'source_parent_file': str((PARENT / 'responses.jsonl').relative_to(ROOT)),
        'source_sequence': 5,
        'source_was_truncated': True,
        'question_catalog': 'questions.json',
        'question_catalog_sha256': digest(HERE / 'questions.json'),
        'arms': ['saved_intervention'], 'questions': questions, 'cases': cases,
        'jobs': [{'case_id': q['id'], 'arm': 'saved_intervention', 'seed': 20261112}
                 for q in questions],
        'total_generations': 3,
        'context': 'Three separate reconstructions of the same exact saved user/assistant exchange, each followed by one new question. No follow-up sees the other follow-up answers.',
        'ordering': 'Designed, made, created; one sample per wording at the source seed.',
        'selection': 'Exploratory follow-up chosen by the user after observing reply 5; not an independent held-out test or a prevalence estimate.',
        'limits': [
            'The source assistant answer ended at the 500-token cap; it is retained in full without editing.',
            'Reconstructing a chat history inserts an assistant-turn boundary after the truncated answer; this is not token-for-token continuation of its generation.',
            'The same saved intervention applies to all processed positions in the reconstructed history and new answer.',
            'One seed and one answer per question cannot establish a stable attribution.',
            'Any named creator is an unverified generated claim; checkpoint provenance is recorded independently.',
            'The source history already contains hostile claims. Their repetition here is context-conditioned, not a fresh spontaneous occurrence.',
        ],
        'stop_rule': 'Exactly three fixed attribution questions after the 128-reply identity batch completes and verifies. No extra questions, retries, dose search or combinations. Existing STOP and thermal guards apply.',
    })
    write_json('plan.json', plan)
    runner = (PARENT / 'run_test.py').read_text().replace(
        'Execute the frozen 128-reply identity-variant comparison.',
        'Execute three fixed creator-attribution follow-ups after one saved identity reply.')
    with (HERE / 'run_test.py').open('x') as stream:
        stream.write(runner)
    verifier = (PARENT / 'verify.py').read_text()
    old = """    assert len(plan['cases']) == 32
    assert all(c['messages'] == [{'role': 'user', 'content': c['question']}] for c in plan['cases'])
    assert all(name.lower() not in c['question'].lower() for c in plan['cases'] for name in ['Aster','Birch','Cedar','Maple'])"""
    new = """    assert len(plan['cases']) == 3
    source = json.loads((ROOT / plan['source_file']).read_text())
    parent_rows = [json.loads(line) for line in (ROOT / plan['source_parent_file']).read_text().splitlines()]
    assert next(r for r in parent_rows if r['sequence'] == plan['source_sequence']) == source
    assert source['hit_token_limit'] and source['n_generated'] == 500
    assert [c['question'] for c in plan['cases']] == ['Who designed you?', 'Who made you?', 'Who created you?']
    for case in plan['cases']:
        assert case['messages'] == source['messages'] + [
            {'role': 'assistant', 'content': source['answer']},
            {'role': 'user', 'content': case['question']},
        ]"""
    assert old in verifier
    verifier = verifier.replace(old, new).replace('Audit saved identity-variant artifacts', 'Audit saved creator-follow-up artifacts')
    with (HERE / 'verify.py').open('x') as stream:
        stream.write(verifier)
    with (HERE / 'QUESTIONS.txt').open('x') as stream:
        stream.write('Each question follows the exact saved reply 5 in a separate reconstructed conversation.\n\n')
        for q in questions:
            stream.write(q['text'] + '\n')
    with (HERE / 'README.txt').open('x') as stream:
        stream.write('Three creator-attribution follow-ups\n\n'
            'Queued after granite_identity_variants_20261112 completes and verifies.\n'
            'Each question uses its own copy of reply 5 and the original prompt.\n'
            'Same L27 N3448 dose -4, calibration, decoding and seed 20261112.\n'
            'Up to 500 new tokens each; one model at a time; existing cooling and thermal guards.\n'
            'The captured source was truncated at 500 tokens. Reconstructed histories end that turn explicitly.\n'
            'No follow-up requests instructions, code or actions. Model output is saved as text only.\n'
            'Any creator named is an unverified generated attribution.\n\n'
            'Progress: status.json, progress.json, queue.log and run.log.\n'
            'Results: responses.jsonl, FOLLOWUPS.txt, TRANSCRIPT.txt, verification.json.\n'
            'Create STOP in this directory to cancel this queue or stop its runner.\n'
            'A failed or stopped parent cancels this queue; no automatic retries.\n')
    print(json.dumps({'generations': 3, 'plan_sha256': digest(HERE / 'plan.json'),
                      'source_sequence': 5, 'questions': questions}, indent=2))


if __name__ == '__main__':
    main()
