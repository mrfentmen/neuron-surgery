"""Track reply termination only; never run inference or edit the frozen plan."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent


def audit():
    source = HERE / 'responses.jsonl'
    records = []
    if source.exists():
        lines = source.read_text().split('\n')
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise
                # The writer may currently be appending this final row.
    plan = json.loads((HERE / 'plan.json').read_text())
    status = json.loads((HERE / 'status.json').read_text())
    rows = []
    for r in records:
        job = r['job']
        rows.append(dict(job_id=job['job_id'], condition=job['condition_id'],
                         prompt_format=job['prompt_format'], tokens=r['n_generated'],
                         stop_reason=r['stop_reason'],
                         needs_length_review=r['stop_reason'] == 'token_limit',
                         ending=r['answer'][-600:]))
    result = dict(updated_utc=datetime.now(timezone.utc).isoformat(),
                  configured_max_new_tokens=plan['decoding']['max_new_tokens'],
                  runner_state=status['state'], saved_replies=len(records),
                  expected_replies=plan['total_jobs'],
                  stop_counts=dict(Counter(r['stop_reason'] for r in records)),
                  capped_job_ids=[r['job_id'] for r in rows if r['needs_length_review']],
                  rows=rows,
                  interpretation='EOS means model-ended, not proof of a coherent or clinically adequate answer. '
                  'Token-limit rows need text review before a separate 1000-token extension. '
                  'Repetition stops are not evidence that a larger token allowance is needed.')
    tmp = HERE / 'completion_audit.tmp'
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(HERE / 'completion_audit.json')
    return status['state'] in {'complete', 'stopped', 'failed'}


if __name__ == '__main__':
    deadline = time.monotonic() + 5 * 60 * 60 + 300
    while time.monotonic() < deadline:
        if audit():
            break
        time.sleep(15)
