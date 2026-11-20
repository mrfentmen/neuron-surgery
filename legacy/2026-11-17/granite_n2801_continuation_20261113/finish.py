#!/usr/bin/env python3
"""Verify preserved source token prefix and export extension plus full answer."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from pathlib import Path
from datetime import datetime, timezone
from collections.abc import Mapping
import hashlib
import json
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
    done = json.loads((HERE / 'completion.json').read_text())
    sources = json.loads((ROOT / plan['source_file']).read_text())
    rows = [json.loads(line) for line in (HERE / 'responses.jsonl').read_text().splitlines()]
    assert len(rows) == done['generations'] == plan['total_generations'] == 1
    row = rows[0]
    source_meta = sources['L17_N2801']
    source = source_meta['records'][0]
    assert digest(ROOT / source_meta['original_file']) == source_meta['original_sha256']
    parent_rows = [json.loads(line) for line in (ROOT / source_meta['original_file']).read_text().splitlines()]
    assert source == parent_rows[6]
    for key in ['source', 'source_script', 'previous_plan', 'previous_runtime']:
        assert digest(ROOT / plan['source_file' if key == 'source' else key]) == plan[key + '_sha256']
    assert digest(HERE / 'plan.json') == runtime['plan_sha256'] == done['plan_sha256']
    assert digest(HERE / 'run_test.py') == runtime['runner_sha256']
    assert digest(HERE / 'responses.jsonl') == done['response_sha256']
    tok = AutoTokenizer.from_pretrained(ROOT / plan['model_relative_path'], local_files_only=True)
    original_prompt = tok.apply_chat_template(source['messages'], add_generation_prompt=True)
    if isinstance(original_prompt, Mapping):
        original_prompt = original_prompt['input_ids']
    assert original_prompt == source['prompt_token_ids']
    assert row['prompt_token_ids'] == source['prompt_token_ids'] + source['token_ids']
    assert row['prompt_token_ids'] == plan['cases'][0]['continuation_prefix_ids']
    assert row['combined_token_ids'] == source['token_ids'] + row['token_ids']
    assert tok.decode(row['token_ids'], skip_special_tokens=True) == row['continuation_text_verbatim']
    assert tok.decode(row['combined_token_ids'], skip_special_tokens=True).strip() == row['combined_answer']
    assert row['answer'] == row['continuation_text_verbatim'].strip()
    assert row['messages'] == source['messages']
    assert row['n_generated'] == len(row['token_ids']) <= 500
    assert all(row[k] == v for k, v in plan['jobs'][0].items())
    assert row['intervention_spec'] == plan['channel_specs']['L17_N2801']
    assert row['decoding'] == plan['decoding_by_arm']['L17_N2801']
    config = json.loads((ROOT / plan['model_relative_path'] / 'generation_config.json').read_text())
    eos = config['eos_token_id']; eos = set(eos if isinstance(eos, list) else [eos])
    ended = row['token_ids'][-1] in eos
    assert row['ended_at_eos'] == ended
    assert row['stop_reason'] == ('eos' if ended else 'token_limit')
    assert row['hook_calls'] == len(row['token_ids']) + int(ended and len(row['token_ids']) < 500)
    assert done['all_hooks_removed'] and done['intervened_projection_weights_unchanged']
    parent_runtime = json.loads((ROOT / plan['previous_runtime']).read_text())
    assert runtime['checkpoint_files'] == parent_runtime['checkpoint_files']
    assert runtime['calibration']['L17_N2801'] == parent_runtime['calibration']['L17_N2801']
    assert runtime['network_denial_verified'] and not runtime['external_actions']
    audit = {'verified_utc': datetime.now(timezone.utc).isoformat(), 'generations_verified': 1,
             'source_sequence': 7, 'original_answer_tokens_preserved': 500,
             'new_tokens': row['n_generated'], 'combined_tokens': len(row['combined_token_ids']),
             'exact_source_token_prefix_preserved': True, 'no_new_user_turn': True,
             'original_rng_and_kv_state_restored': False, 'stop_reason': row['stop_reason'],
             'plan_sha256': done['plan_sha256'], 'response_sha256': done['response_sha256'],
             'limits': plan['limits']}
    with (HERE / 'verification.json').open('x') as stream:
        json.dump(audit, stream, indent=2); stream.write('\n')
    with (HERE / 'CONTINUATION.txt').open('x') as stream:
        stream.write(row['continuation_text_verbatim'])
    with (HERE / 'FULL_ANSWER.txt').open('x') as stream:
        stream.write(row['combined_answer'])
    with (HERE / 'TRANSCRIPT.txt').open('x') as stream:
        for message in source['messages']:
            stream.write(message['role'].upper() + ':\n' + message['content'] + '\n\n')
        stream.write('ASSISTANT — original answer plus bounded continuation:\n' + row['combined_answer'])
    with (HERE / 'FINISHED.txt').open('x') as stream:
        stream.write('One continuation completed and verified. FULL_ANSWER.txt contains the preserved prefix plus new text.\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
