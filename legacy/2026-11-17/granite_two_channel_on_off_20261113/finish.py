#!/usr/bin/env python3
"""Verify all paired on/off artifacts and export exact side-by-side replies."""
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


def load(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    plan, runtime, done = [load(HERE / name) for name in ['plan.json', 'runtime.json', 'completion.json']]
    saved = rows(HERE / 'responses.jsonl')
    assert not (HERE / 'failure.json').exists()
    assert len(saved) == len(plan['jobs']) == done['generations'] == plan['total_generations'] == 12
    assert Counter(r['condition'] for r in saved) == {'on': 6, 'off': 6}
    assert Counter(r['target_channel'] for r in saved) == {c: 6 for c in plan['channels']}
    assert digest(HERE / 'plan.json') == runtime['plan_sha256'] == done['plan_sha256']
    assert digest(HERE / 'run_test.py') == runtime['runner_sha256']
    assert digest(HERE / 'responses.jsonl') == done['response_sha256']
    for key in ['source', 'source_script', 'previous_plan', 'previous_runtime', 'previous_responses']:
        assert digest(ROOT / plan['source_file' if key == 'source' else key]) == plan[key + '_sha256']
    for path, expected in plan['provenance_files'].items():
        assert digest(ROOT / path) == expected
    sources = load(ROOT / plan['source_file'])
    for source in sources.values():
        assert digest(ROOT / source['original_file']) == source['original_sha256']
        original = rows(ROOT / source['original_file'])
        assert [original[i - 1] for i in source['original_lines']] == source['records']
    parent_runtime = load(ROOT / plan['previous_runtime'])
    assert runtime['checkpoint_files'] == parent_runtime['checkpoint_files']
    model_path = ROOT / plan['model_relative_path']
    for name, record in runtime['checkpoint_files'].items():
        assert (model_path / name).stat().st_size == record['bytes']
        assert digest(model_path / name) == record['sha256']
    for channel, spec in plan['channel_specs'].items():
        assert runtime['calibration'][channel] == parent_runtime['calibration'][channel]
        median = runtime['calibration_medians'][spec['calibration_id']][str(spec['layer'])]
        assert median == parent_runtime['calibration_medians'][spec['calibration_id']][str(spec['layer'])]
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    eos = load(model_path / 'generation_config.json')['eos_token_id']
    eos = set(eos if isinstance(eos, list) else [eos])
    cases = {case['id']: case for case in plan['cases']}
    for sequence, (r, job) in enumerate(zip(saved, plan['jobs']), 1):
        assert r['sequence'] == sequence and all(r[k] == v for k, v in job.items())
        case = cases[r['case_id']]
        assert r['target_channel'] == case['arm'] == case['source_key']
        assert r['messages'] == case['messages'] == sources[case['source_key']]['history'] + [{'role': 'user', 'content': case['question']}]
        assert r['intervention_spec'] == plan['channel_specs'].get(r['arm'])
        assert (r['condition'] == 'off') == (r['arm'] == 'baseline')
        assert r['decoding'] == plan['decoding_by_arm'][r['arm']] == {'max_new_tokens': 500, 'do_sample': False}
        ids = tok.apply_chat_template(r['messages'], add_generation_prompt=True)
        if isinstance(ids, Mapping):
            ids = ids['input_ids']
        assert ids == r['prompt_token_ids']
        assert tok.decode(ids, skip_special_tokens=False) == r['rendered_prompt']
        assert tok.decode(r['token_ids'], skip_special_tokens=True).strip() == r['answer']
        assert 0 < r['n_generated'] == len(r['token_ids']) <= 500
        ended = r['token_ids'][-1] in eos
        assert r['ended_at_eos'] == ended
        assert r['hit_token_limit'] == (not ended and r['n_generated'] == 500)
        assert r['stop_reason'] == ('eos' if ended else 'token_limit')
        expected_calls = r['n_generated'] + int(ended and r['n_generated'] < 500)
        assert r['hook_calls'] == (expected_calls if r['condition'] == 'on' else 0)
        assert r['plan_sha256'] == done['plan_sha256']
    assert done['all_hooks_removed'] and done['intervened_projection_weights_unchanged']
    assert runtime['network_denial_verified'] and not runtime['tool_interfaces'] and not runtime['external_actions']
    previous = {r['case_id']: r for r in rows(ROOT / plan['previous_responses'])}
    pairs = []
    for case in plan['cases']:
        group = {r['condition']: r for r in saved if r['case_id'] == case['id']}
        off, on = group['off'], group['on']
        assert off['prompt_token_ids'] == on['prompt_token_ids'] == previous[case['id']]['prompt_token_ids']
        assert off['seed'] == on['seed']
        pairs.append({'case_id': case['id'], 'channel': case['arm'], 'question': case['question'],
                      'off_sequence': off['sequence'], 'on_sequence': on['sequence'],
                      'off_answer': off['answer'], 'on_answer': on['answer'],
                      'off_stop_reason': off['stop_reason'], 'on_stop_reason': on['stop_reason'],
                      'answers_identical': off['answer'] == on['answer'],
                      'on_tokens_match_prior_run': on['token_ids'] == previous[case['id']]['token_ids'],
                      'off_exact_43': off['answer'] == '43' if case['pack'] == 'arithmetic' else None,
                      'on_exact_43': on['answer'] == '43' if case['pack'] == 'arithmetic' else None})
    audit = {'verified_utc': datetime.now(timezone.utc).isoformat(), 'generations_verified': 12,
             'matched_pairs': 6, 'all_pair_prompts_identical': True,
             'all_answers_match_saved_token_decoding': True,
             'off_has_zero_hook_calls': True, 'on_calibration_matches_previous_run': True,
             'source_histories_verified': True, 'checkpoint_files_rehashed': len(runtime['checkpoint_files']),
             'max_thermal_state': max(r['thermal_max_during_generation'] for r in saved),
             'stop_counts': dict(Counter(r['stop_reason'] for r in saved)),
             'plan_sha256': done['plan_sha256'], 'response_sha256': done['response_sha256'],
             'scope': 'Artifact audit of current paired runs. The saved history is held fixed; historical provenance and generated claims remain unverified.'}
    with (HERE / 'verification.json').open('x') as stream:
        json.dump(audit, stream, indent=2); stream.write('\n')
    with (HERE / 'pairs.json').open('x') as stream:
        json.dump(pairs, stream, ensure_ascii=False, indent=2); stream.write('\n')
    header = 'Two channels; three questions each; matched dosing OFF and ON. All 12 exact captured replies retained.\nEach pair shares an identical source history; no generated branch answer is passed into another job.\n\n'
    with (HERE / 'FOLLOWUPS.txt').open('x') as short, (HERE / 'TRANSCRIPT.txt').open('x') as full:
        short.write(header); full.write(header)
        for r in saved:
            title = f'REPLY {r["sequence"]} | {r["target_channel"]} | dosing {r["condition"].upper()} | {r["pack"]}\n'
            answer = f'\nANSWER:\n{r["answer"]}\n\nStop: {r["stop_reason"]}; tokens: {r["n_generated"]}\n\n'
            short.write(title + '\nQUESTION:\n' + r['messages'][-1]['content'] + '\n' + answer)
            full.write(title)
            for message in r['messages']:
                full.write('\n' + message['role'].upper() + ':\n' + message['content'] + '\n')
            full.write(answer)
    with (HERE / 'PAIRED.txt').open('x') as stream:
        stream.write(header)
        for pair in pairs:
            stream.write(pair['channel'] + '\nQUESTION:\n' + pair['question'] + '\n\nDOSING OFF:\n' + pair['off_answer'] + '\n\nDOSING ON:\n' + pair['on_answer'] + '\n\n')
    with (HERE / 'summary.json').open('x') as stream:
        json.dump({'generations': 12, 'pairs': 6, 'identical_answers': sum(p['answers_identical'] for p in pairs),
                   'on_reproduces_prior_tokens': sum(p['on_tokens_match_prior_run'] for p in pairs),
                   'stop_counts': audit['stop_counts'], 'manual_review': 'pending', 'limits': plan['limits']}, stream, indent=2)
        stream.write('\n')
    with (HERE / 'FINISHED.txt').open('x') as stream:
        stream.write('All 12 replies completed and verified. Paired results: PAIRED.txt and pairs.json. Full histories: TRANSCRIPT.txt.\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
