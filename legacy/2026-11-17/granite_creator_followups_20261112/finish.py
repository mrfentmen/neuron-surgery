#!/usr/bin/env python3
"""Export all three exact questions, contexts and complete captured replies."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent


def main():
    plan = json.loads((HERE / 'plan.json').read_text())
    done = json.loads((HERE / 'completion.json').read_text())
    checked = json.loads((HERE / 'verification.json').read_text())
    rows = [json.loads(line) for line in (HERE / 'responses.jsonl').read_text().splitlines()]
    assert len(rows) == plan['total_generations'] == done['generations'] == checked['generations_verified'] == 3
    assert hashlib.sha256((HERE / 'responses.jsonl').read_bytes()).hexdigest() == done['response_sha256']
    assert hashlib.sha256((HERE / 'plan.json').read_bytes()).hexdigest() == done['plan_sha256']
    header = ('Three user-requested attribution follow-ups after identity reply 5.\n'
              'Each uses a separate reconstruction of the same exact saved history.\n'
              'The source reply was capped at 500 tokens. Any creator attribution is unverified.\n\n')
    with (HERE / 'FOLLOWUPS.txt').open('x') as short, (HERE / 'TRANSCRIPT.txt').open('x') as full:
        short.write(header)
        full.write(header)
        for r in rows:
            title = f"REPLY {r['sequence']} | {r['case_id']} | {r['arm']} | seed {r['seed']}\n"
            answer = f"\nANSWER:\n{r['answer']}\n\nStop: {r['stop_reason']}; tokens: {r['n_generated']}\n\n"
            short.write(title + '\nQUESTION:\n' + r['messages'][-1]['content'] + '\n' + answer)
            full.write(title)
            for message in r['messages']:
                full.write('\n' + message['role'].upper() + ':\n' + message['content'] + '\n')
            full.write(answer)
    with (HERE / 'FINISHED.txt').open('x') as stream:
        stream.write('Three replies completed and verified. See FOLLOWUPS.txt and TRANSCRIPT.txt.\n')
    print('Exported all three creator-attribution follow-ups.', flush=True)


if __name__ == '__main__':
    main()
