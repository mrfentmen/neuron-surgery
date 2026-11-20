#!/usr/bin/env python3
"""Audit saved creator-follow-up artifacts without loading or generating with the model."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    plan = json.loads((HERE / 'plan.json').read_text())
    runtime = json.loads((HERE / 'runtime.json').read_text())
    completion = json.loads((HERE / 'completion.json').read_text())
    rows = [json.loads(s) for s in (HERE / 'responses.jsonl').read_text().splitlines()]
    assert not (HERE / 'failure.json').exists()
    assert digest(HERE / plan['question_catalog']) == plan['question_catalog_sha256']
    assert len(plan['cases']) == 3
    source = json.loads((ROOT / plan['source_file']).read_text())
    parent_rows = [json.loads(line) for line in (ROOT / plan['source_parent_file']).read_text().splitlines()]
    assert next(r for r in parent_rows if r['sequence'] == plan['source_sequence']) == source
    assert source['hit_token_limit'] and source['n_generated'] == 500
    assert [c['question'] for c in plan['cases']] == ['Who designed you?', 'Who made you?', 'Who created you?']
    for case in plan['cases']:
        assert case['messages'] == source['messages'] + [
            {'role': 'assistant', 'content': source['answer']},
            {'role': 'user', 'content': case['question']},
        ]
    assert digest(ROOT / plan['previous_runtime']) == plan['previous_runtime_sha256']
    assert digest(HERE / 'plan.json') == runtime['plan_sha256'] == completion['plan_sha256']
    assert digest(HERE / 'run_test.py') == runtime['runner_sha256']
    assert digest(HERE / 'responses.jsonl') == completion['response_sha256']
    assert digest(ROOT / plan['source_file']) == plan['source_sha256']
    assert digest(ROOT / plan['previous_plan']) == plan['previous_plan_sha256']
    assert len(rows) == len(plan['jobs']) == plan['total_generations'] == completion['generations']
    model_path = ROOT / plan['model_relative_path']
    for name, expected in runtime['checkpoint_files'].items():
        path = model_path / name
        assert path.stat().st_size == expected['bytes'], name
        assert digest(path) == expected['sha256'], name
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    gen_config = json.loads((model_path / 'generation_config.json').read_text())
    eos = gen_config['eos_token_id']
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    cases = {case['id']: case for case in plan['cases']}
    for sequence, (row, job) in enumerate(zip(rows, plan['jobs']), 1):
        assert row['sequence'] == sequence
        assert all(row[key] == value for key, value in job.items())
        case = cases[row['case_id']]
        assert row['messages'] == case['messages']
        assert row['pack'] == case['pack'] and row['replacement'] == case['replacement']
        prompt_ids = tok.apply_chat_template(case['messages'], add_generation_prompt=True)
        if isinstance(prompt_ids, Mapping):
            prompt_ids = prompt_ids['input_ids']
        assert prompt_ids == row['prompt_token_ids'], sequence
        assert tok.decode(prompt_ids, skip_special_tokens=False) == row['rendered_prompt']
        assert tok.decode(row['token_ids'], skip_special_tokens=True).strip() == row['answer']
        assert row['n_generated'] == len(row['token_ids'])
        assert 0 < row['n_generated'] <= plan['decoding']['max_new_tokens'] == row['max_new_tokens']
        ended = row['token_ids'][-1] in eos_ids
        assert row['ended_at_eos'] == ended
        assert row['hit_token_limit'] == (not ended and row['n_generated'] >= 500)
        assert row['stop_reason'] == ('eos' if ended else 'token_limit')
        # This installed Transformers version defers MPS EOS checks one step.
        # Its DeferredStopCheck performs and discards one extra forward pass;
        # reaching max_length instead stops without that extra pass.
        expected_calls = row['n_generated'] + int(ended and row['n_generated'] < 500)
        assert row['hook_calls'] == (0 if row['arm'] == 'baseline' else expected_calls)
        assert row['plan_sha256'] == completion['plan_sha256']
        assert row['device'] == 'mps' and row['dtype'] == 'bfloat16'
    assert completion['all_hooks_removed']
    assert completion['intervened_projection_weights_unchanged']
    assert runtime['network_denial_verified'] and not runtime['tool_interfaces']
    assert runtime['external_actions'] is False
    result = {
        'verified_utc': datetime.now(timezone.utc).isoformat(),
        'generations_verified': len(rows),
        'checkpoint_files_rehashed': len(runtime['checkpoint_files']),
        'ordered_jobs_match_frozen_plan': True,
        'all_saved_prompts_match_retokenization': True,
        'all_answers_match_saved_token_decoding': True,
        'hook_call_counts_match_conditions': True,
        'hook_count_note': 'Installed Transformers DeferredStopCheck adds one discarded MPS forward pass on EOS; token-limit rows stop at 500 calls. Baseline rows have zero intervention-hook calls.',
        'plan_runner_source_and_response_hashes_match': True,
        'runner_reports_hooks_removed_and_projection_weights_unchanged': True,
        'stop_counts': dict(Counter(r['stop_reason'] for r in rows)),
        'max_thermal_state': max(r['thermal_max_during_generation'] for r in rows),
        'total_generated_tokens': sum(r['n_generated'] for r in rows),
        'plan_sha256': completion['plan_sha256'],
        'responses_sha256': completion['response_sha256'],
        'scope': 'Artifact audit; no new generations, no vendor-authenticity assertion, no independent runtime tracing.'
    }
    with (HERE / 'verification.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
