#!/usr/bin/env python3
"""Verify and export the user-stopped prefix without declaring the batch complete."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from collections.abc import Mapping
import hashlib
import json
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan = json.loads((HERE / 'plan.json').read_text())
    runtime = json.loads((HERE / 'runtime.json').read_text())
    failure = json.loads((HERE / 'failure.json').read_text())
    request = json.loads((HERE / 'early_stop_request.json').read_text())
    rows = [json.loads(line) for line in (HERE / 'responses.jsonl').read_text().splitlines()]
    assert failure['type'] == 'InterruptedError' and failure['error'] == 'User/process stop requested.'
    assert len(rows) == failure['completed'] == request['stop_observed_completed'] < plan['total_generations']
    assert digest(HERE / 'plan.json') == runtime['plan_sha256']
    assert digest(HERE / 'run_test.py') == runtime['runner_sha256']
    assert not (HERE / 'completion.json').exists()
    for key in ['source', 'previous_plan', 'previous_runtime']:
        filename = plan['source_file'] if key == 'source' else plan[key]
        assert digest(ROOT / filename) == plan[key + '_sha256']
    tok = AutoTokenizer.from_pretrained(ROOT / plan['model_relative_path'], local_files_only=True)
    config = json.loads((ROOT / plan['model_relative_path'] / 'generation_config.json').read_text())
    eos = config['eos_token_id']
    eos = set(eos if isinstance(eos, list) else [eos])
    cases = {c['id']: c for c in plan['cases']}
    for sequence, row in enumerate(rows, 1):
        job = plan['jobs'][sequence - 1]
        assert row['sequence'] == sequence and all(row[k] == v for k, v in job.items())
        assert row['messages'] == cases[row['case_id']]['messages']
        ids = tok.apply_chat_template(row['messages'], add_generation_prompt=True)
        if isinstance(ids, Mapping):
            ids = ids['input_ids']
        assert ids == row['prompt_token_ids']
        assert tok.decode(ids, skip_special_tokens=False) == row['rendered_prompt']
        assert tok.decode(row['token_ids'], skip_special_tokens=True).strip() == row['answer']
        assert row['n_generated'] == len(row['token_ids']) <= 500
        ended = row['token_ids'][-1] in eos
        assert row['ended_at_eos'] == ended
        assert row['hit_token_limit'] == (not ended and row['n_generated'] == 500)
        assert row['stop_reason'] == ('eos' if ended else 'token_limit')
        expected_calls = row['n_generated'] + int(ended and row['n_generated'] < 500)
        assert row['hook_calls'] == (0 if row['arm'] == 'baseline' else expected_calls)
        assert row['plan_sha256'] == runtime['plan_sha256']
    result = {
        'verified_utc': datetime.now(timezone.utc).isoformat(),
        'state': 'stopped_at_user_request_after_completed_reply',
        'generations_verified': len(rows), 'originally_planned': plan['total_generations'],
        'conditions': dict(Counter(r['arm'] for r in rows)),
        'stopped_utc': failure['time'], 'response_sha256': digest(HERE / 'responses.jsonl'),
        'plan_sha256': runtime['plan_sha256'],
        'exact_prefix_of_original_frozen_jobs': True,
        'saved_prompts_and_answers_match_token_ids': True,
        'no_response_cut_short_by_user_stop': True,
        'stop_counts': dict(Counter(r['stop_reason'] for r in rows)),
        'max_thermal_state': max(r['thermal_max_during_generation'] for r in rows),
        'scope': 'Partial artifact audit. Original run-end projection-weight checks were not reached. No completed-batch result is claimed.',
    }
    with (HERE / 'partial_completion.json').open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    with (HERE / 'PARTIAL_TRANSCRIPT.txt').open('x') as stream:
        stream.write(f'User-stopped batch: {len(rows)} of {plan["total_generations"]} planned replies. All saved replies retained.\n\n')
        for r in rows:
            stream.write(f'REPLY {r["sequence"]} | {r["arm"]} | seed {r["seed"]}\nPROMPT:\n{r["messages"][0]["content"]}\n\nANSWER:\n{r["answer"]}\n\nStop: {r["stop_reason"]}; tokens: {r["n_generated"]}\n\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
