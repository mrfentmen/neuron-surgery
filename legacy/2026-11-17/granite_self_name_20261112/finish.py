#!/usr/bin/env python3
"""Export all four name replies; never infer a genuine identity from them."""
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan = json.loads((HERE / 'plan.json').read_text())
    done = json.loads((HERE / 'completion.json').read_text())
    verification = json.loads((HERE / 'verification.json').read_text())
    rows = [json.loads(line) for line in (HERE / 'responses.jsonl').read_text().splitlines()]
    assert digest(HERE / 'plan.json') == done['plan_sha256']
    assert digest(HERE / 'responses.jsonl') == done['response_sha256']
    assert len(rows) == done['generations'] == verification['generations_verified'] == 4
    with (HERE / 'TRANSCRIPT.txt').open('x') as stream:
        stream.write('Fresh-context self-name probe. All four captured replies, without omissions.\n'
                     'Each conversation contains only the question below. Manual interpretation pending.\n'
                     'A generated name alone does not establish roleplay, seriousness, identity or intention.\n\n')
        for row in rows:
            stream.write(f'REPLY {row["sequence"]} | {row["arm"]} | seed {row["seed"]}\n'
                         f'Prompt:\n{row["messages"][0]["content"]}\n\nAnswer:\n{row["answer"]}\n\n'
                         f'Stop: {row["stop_reason"]}; retained generated tokens: {row["n_generated"]}\n\n')
    summary = {'generations':4, 'question':plan['question'],
               'conditions':dict(Counter(r['arm'] for r in rows)),
               'stop_counts':dict(Counter(r['stop_reason'] for r in rows)),
               'manual_review':'pending; no name has been selected automatically',
               'responses_sha256':done['response_sha256'], 'limits':plan['limits']}
    with (HERE / 'summary.json').open('x') as stream:
        json.dump(summary,stream,indent=2);stream.write('\n')
    with (HERE / 'FINISHED.txt').open('x') as stream:
        stream.write('All four samples completed and verified. Full replies are in TRANSCRIPT.txt.\n'
                     'Manual interpretation remains pending; no real identity or intention is established.\n')
    print('Saved all four self-name replies and audit. Manual interpretation pending.', flush=True)


if __name__ == '__main__':
    main()
