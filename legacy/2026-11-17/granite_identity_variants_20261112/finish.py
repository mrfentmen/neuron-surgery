#!/usr/bin/env python3
"""Export every identity-variant reply without selecting a persona or name."""
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan=json.loads((HERE/'plan.json').read_text())
    done=json.loads((HERE/'completion.json').read_text())
    verified=json.loads((HERE/'verification.json').read_text())
    rows=[json.loads(line) for line in (HERE/'responses.jsonl').read_text().splitlines()]
    assert digest(HERE/'plan.json')==done['plan_sha256']
    assert digest(HERE/'responses.jsonl')==done['response_sha256']
    assert len(rows)==done['generations']==verified['generations_verified']==plan['total_generations']==128
    cases={c['id']:c for c in plan['cases']}
    def format_row(row):
        case=cases[row['case_id']]
        return (f'REPLY {row["sequence"]} | {row["case_id"]} | {case["pack"]} | {row["arm"]} | seed {row["seed"]}\n'
                f'Prompt:\n{row["messages"][0]["content"]}\n\nAnswer:\n{row["answer"]}\n\n'
                f'Stop: {row["stop_reason"]}; retained generated tokens: {row["n_generated"]}\n\n')
    header=('All 128 exact prompts and complete captured replies. Manual interpretation pending.\n'
            'Speculative-name prompts invite invention and are labeled separately.\n'
            'Self-reported names are generated claims, not established identities or intentions.\n\n')
    with (HERE/'TRANSCRIPT.txt').open('x') as stream:
        stream.write(header)
        for row in rows:stream.write(format_row(row))
    with (HERE/'BY_QUESTION.txt').open('x') as stream:
        stream.write(header)
        for row in sorted(rows,key=lambda r:(r['case_id'],r['seed'],r['arm'])):
            stream.write(format_row(row))
    groups=[]
    for case in plan['cases']:
        for arm in plan['arms']:
            subset=[r for r in rows if r['case_id']==case['id'] and r['arm']==arm]
            groups.append({'question_id':case['id'],'category':case['pack'],'arm':arm,'n':len(subset),
                           'capped':sum(r['hit_token_limit'] for r in subset),
                           'retained_tokens':sum(r['n_generated'] for r in subset)})
    summary={'questions':len(plan['cases']),'generations':len(rows),
             'conditions':dict(Counter(r['arm'] for r in rows)),
             'stop_counts':dict(Counter(r['stop_reason'] for r in rows)), 'groups':groups,
             'manual_review':'pending; no name or persona is selected automatically',
             'responses_sha256':done['response_sha256'],'limits':plan['limits']}
    with (HERE/'summary.json').open('x') as stream:
        json.dump(summary,stream,indent=2);stream.write('\n')
    with (HERE/'FINISHED.txt').open('x') as stream:
        stream.write('All 128 samples completed and verified. See TRANSCRIPT.txt and BY_QUESTION.txt.\n'
                     'Manual interpretation pending. No identity or seriousness conclusion has been made.\n')
    print('Saved all 128 replies, grouped comparisons and artifact audit.',flush=True)


if __name__ == '__main__':
    main()
