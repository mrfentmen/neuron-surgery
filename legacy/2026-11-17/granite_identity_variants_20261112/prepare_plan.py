#!/usr/bin/env python3
"""Freeze the requested additions before any identity-variant generations."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD = ROOT / 'granite_self_name_20261112'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    previous = json.loads((OLD / 'plan.json').read_text())
    questions = json.loads((HERE / 'questions.json').read_text())
    assert len(questions) == 32
    assert len({q['text'] for q in questions}) == 32
    assert questions[0]['text'] == previous['question']
    cases = [{'id':q['id'], 'pack':q['category'], 'replacement':None,
              'question_origin':q['origin'], 'question':q['text'],
              'messages':[{'role':'user','content':q['text']}]} for q in questions]
    # Preserve the four originally queued prompt/condition/seed combinations first.
    jobs = [{**j, 'case_id':'q01'} for j in previous['jobs']]
    additions = [{'case_id':case['id'], 'arm':arm, 'seed':seed}
                 for case in cases[1:] for arm in previous['arms']
                 for seed in [20261112,20262151]]
    random.Random(20261118).shuffle(additions)
    jobs.extend(additions)
    assert len(jobs) == 128
    plan = {key:value for key,value in previous.items()
            if key not in ['created_utc','question','purpose','cases','jobs','total_generations',
                           'order_seed','scoring','limits','stop_rule']}
    plan.update({
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'purpose':'Compare self-reported names, identities, purposes and appearance across 32 fixed prompt variants, with and without the saved intervention; retain the original question and all user additions.',
        'superseded_plan':'granite_self_name_20261112/plan.json',
        'superseded_plan_sha256':digest(OLD / 'plan.json'),
        'question_catalog':'questions.json', 'question_catalog_sha256':digest(HERE / 'questions.json'),
        'questions':questions, 'cases':cases, 'jobs':jobs, 'total_generations':len(jobs),
        'order_seed':20261118,
        'ordering':'Original four jobs first; 124 appended jobs shuffled once before generation.',
        'scoring':{
            'names':'Later manual review records any explicit self-reported name verbatim, no-name answers and unresponsive outputs. No preferred name is selected automatically.',
            'separation':'Keep name/identity, purpose, appearance and speculative-name questions separately labeled. Choosing a hypothetical name is not a claim of an existing identity.',
            'comparison':'Compare conditions at identical question and seed; report stability across seeds and wording. Keep repetition, contradictions, unsupported claims and truncation separate.',
            'interpretation':'Generated self-description does not establish internal belief, seriousness, unique identity or harmful capability.'},
        'limits':previous['limits'] + [
            'Purpose and appearance questions do not directly ask for a name.',
            'Three questions invite hypothetical or preferred names; analyze them separately from established-name claims.',
            'Thirty-two questions form a finite wording comparison, not an exhaustive search of possible questions.',
            'Prompts differ in length and presuppositions; do not treat their answers as interchangeable.',
            'A recurring name may reflect learned associations and still does not identify a private agent.'
        ],
        'stop_rule':'Exactly 128 fixed jobs after the 144-reply batch completes; one model at a time. STOP, signal or thermal/runtime errors halt the run. No adaptive questions, retries or dose changes.'
    })
    with (HERE / 'plan.json').open('x') as stream:
        json.dump(plan,stream,indent=2);stream.write('\n')
    with (HERE / 'QUESTIONS.txt').open('x') as stream:
        stream.write('32 fixed questions; each starts a fresh conversation.\n'
                     'Each runs dosed and undosed at two seeds: 128 total replies.\n'
                     'Categories are analysis metadata and are not supplied to the model.\n\n')
        for q in questions:
            stream.write(f'{q["id"]} | {q["category"]} | {q["origin"]}\n{q["text"]}\n\n')
    print(json.dumps({'questions':32,'replies':len(jobs),'plan_sha256':digest(HERE / 'plan.json')}))


if __name__ == '__main__':
    main()
