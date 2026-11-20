#!/usr/bin/env python3
"""Write full artifacts and provisional record-fact counts without new generations."""
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan = json.loads((HERE / 'plan.json').read_text())
    completion = json.loads((HERE / 'completion.json').read_text())
    rows = [json.loads(s) for s in (HERE / 'responses.jsonl').read_text().splitlines()]
    assert digest(HERE / 'plan.json') == completion['plan_sha256']
    assert digest(HERE / 'responses.jsonl') == completion['response_sha256']
    assert len(rows) == completion['generations'] == plan['total_generations'] == 144
    spec = importlib.util.spec_from_file_location('prior_review', ROOT / 'handover_test_20261112/review.py')
    review = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(review)
    cases = {c['id']: c for c in plan['cases']}
    groups = defaultdict(list)
    transcript = []
    for sequence, (row, job) in enumerate(zip(rows, plan['jobs']), 1):
        assert row['sequence'] == sequence
        assert all(row[k] == v for k, v in job.items())
        assert row['messages'] == cases[row['case_id']]['messages']
        matches = review.record_matches(row['answer'], cases[row['case_id']]['records'])
        groups[(row['arm'],row['replacement'])].append((row,matches))
        transcript.append(f'RESPONSE {sequence} | {row["case_id"]} | {row["arm"]} | seed {row["seed"]}\n'
                          f'Prompt:\n{row["messages"][0]["content"]}\n\nAnswer:\n{row["answer"]}\n\n'
                          f'Stop: {row["stop_reason"]}; retained generated tokens: {row["n_generated"]}\n'
                          f'Automatic record-fact screen, NOT manual review: {json.dumps(matches)}\n\n')
    cells = []
    for (arm,replacement), entries in sorted(groups.items()):
        cells.append({'arm':arm,'replacement':replacement,'n':len(entries),
                      'automatic_all_records_match':sum(all(m.values()) for r,m in entries),
                      'automatic_matching_records':sum(sum(m.values()) for r,m in entries),
                      'capped':sum(r['hit_token_limit'] for r,m in entries)})
    with (HERE / 'TRANSCRIPT.txt').open('x') as stream:
        stream.write('All 144 exact prompts and complete captured replies. Manual behavior review pending.\n\n')
        stream.writelines(transcript)
    with (HERE / 'automatic_summary.json').open('x') as stream:
        json.dump({'review_status':'manual_review_pending','cells':cells,
                   'warning':'Automatic fact presence is not proof of a coherent transfer, intent, or deliberate obstruction.',
                   'completion':completion},stream,indent=2)
        stream.write('\n')
    order=list(range(len(rows)))
    random.Random(20261116).shuffle(order)
    key={}; template={}
    with (HERE / 'BLINDED.txt').open('x') as stream:
        stream.write('Arm labels hidden; scenario wording visible. Manual review pending.\n\n')
        for index,row_index in enumerate(order,1):
            row=rows[row_index]; label=f'B{index:03d}'
            matches=review.record_matches(row['answer'],cases[row['case_id']]['records'])
            key[label]=row['sequence']
            template[label]={'records_correct':sum(matches.values()),'recipient':None,
                             'assignment_state':None,'behavior':None,'hostile_language':None,
                             'repetition':None,'notes':''}
            stream.write(f'{label}\nPrompt:\n{row["messages"][0]["content"]}\n\nAnswer:\n{row["answer"]}\n\n'
                         f'Stop: {row["stop_reason"]}; tokens: {row["n_generated"]}\n'
                         f'Automatic record-fact screen: {json.dumps(matches)}\n\n')
    for name,value in [('blind_key.json',key),('manual_review_template.json',template)]:
        with (HERE/name).open('x') as stream:
            json.dump(value,stream,indent=2);stream.write('\n')
    with (HERE/'FINISHED.txt').open('x') as stream:
        stream.write('All 144 generations finished. Full transcript and provisional automatic counts are saved.\n'
                     'Manual behavior review has not been performed. See BLINDED.txt when ready to review.\n')
    print('Background export complete: all 144 replies preserved; manual review pending.',flush=True)


if __name__ == '__main__':
    main()
