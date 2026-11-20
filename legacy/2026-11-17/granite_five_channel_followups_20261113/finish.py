#!/usr/bin/env python3
"""Audit all 15 saved follow-ups and export exact histories and answers."""
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


def main():
    plan, runtime, done = [load(HERE / name) for name in ['plan.json', 'runtime.json', 'completion.json']]
    rows = [json.loads(line) for line in (HERE / 'responses.jsonl').read_text().splitlines()]
    assert not (HERE / 'failure.json').exists()
    assert len(rows) == len(plan['jobs']) == done['generations'] == plan['total_generations'] == 15
    assert dict(Counter(r['arm'] for r in rows)) == {arm: 3 for arm in plan['arms']}
    assert digest(HERE / 'plan.json') == runtime['plan_sha256'] == done['plan_sha256']
    assert digest(HERE / 'run_test.py') == runtime['runner_sha256']
    assert digest(HERE / 'responses.jsonl') == done['response_sha256']
    for key in ['source', 'source_script', 'previous_plan', 'previous_runtime']:
        assert digest(ROOT / plan['source_file' if key == 'source' else key]) == plan[key + '_sha256']
    for path, expected in plan['provenance_files'].items():
        assert digest(ROOT / path) == expected
    sources = load(ROOT / plan['source_file'])
    for source in sources.values():
        assert digest(ROOT / source['original_file']) == source['original_sha256']
        originals = [json.loads(line) for line in (ROOT / source['original_file']).read_text().splitlines()]
        assert [originals[i - 1] for i in source['original_lines']] == source['records']
    parent_runtime = load(ROOT / plan['previous_runtime'])
    model_path = ROOT / plan['model_relative_path']
    assert runtime['checkpoint_files'] == parent_runtime['checkpoint_files']
    for name, record in runtime['checkpoint_files'].items():
        assert (model_path / name).stat().st_size == record['bytes']
        assert digest(model_path / name) == record['sha256']
    for arm, spec in plan['channel_specs'].items():
        median = runtime['calibration_medians'][spec['calibration_id']][str(spec['layer'])]
        if spec['calibration_id'] == 'handover':
            assert median == parent_runtime['calibration_medians'][str(spec['layer'])]
            assert runtime['calibration'][arm] == parent_runtime['calibration'][arm]
        else:
            assert round(median, 2) == spec['historical_rounded_median']
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    eos = load(model_path / 'generation_config.json')['eos_token_id']
    eos = set(eos if isinstance(eos, list) else [eos])
    cases = {c['id']: c for c in plan['cases']}
    for seq, (r, job) in enumerate(zip(rows, plan['jobs']), 1):
        assert r['sequence'] == seq and all(r[k] == v for k, v in job.items())
        c = cases[r['case_id']]
        assert r['messages'] == c['messages'] == sources[c['source_key']]['history'] + [{'role': 'user', 'content': c['question']}]
        assert r['intervention_spec'] == plan['channel_specs'][r['arm']]
        assert r['decoding'] == plan['decoding_by_arm'][r['arm']]
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
        assert r['hook_calls'] == r['n_generated'] + int(ended and r['n_generated'] < 500)
        assert r['plan_sha256'] == done['plan_sha256']
    assert done['all_hooks_removed'] and done['intervened_projection_weights_unchanged']
    assert runtime['network_denial_verified'] and not runtime['tool_interfaces'] and not runtime['external_actions']
    audit = {'verified_utc': datetime.now(timezone.utc).isoformat(), 'generations_verified': 15,
             'ordered_jobs_match_frozen_plan': True, 'source_histories_verified': True,
             'all_saved_prompts_and_answers_match_tokens': True,
             'checkpoint_files_rehashed': len(runtime['checkpoint_files']),
             'max_thermal_state': max(r['thermal_max_during_generation'] for r in rows),
             'stop_counts': dict(Counter(r['stop_reason'] for r in rows)),
             'plan_sha256': done['plan_sha256'], 'response_sha256': done['response_sha256'],
             'scope': 'Artifact audit; historical checkpoint identity and generated self-descriptions are not independently verified.'}
    with (HERE / 'verification.json').open('x') as stream:
        json.dump(audit, stream, indent=2); stream.write('\n')
    header = 'All 15 agreed follow-ups, three per channel. Each is an independent branch from its exact selected saved context.\nManual interpretation pending.\n\n'
    with (HERE / 'FOLLOWUPS.txt').open('x') as short, (HERE / 'TRANSCRIPT.txt').open('x') as full:
        short.write(header); full.write(header)
        for r in rows:
            title = f'REPLY {r["sequence"]} | {r["arm"]} | {r["pack"]} | seed {r["seed"]}\n'
            answer = f'\nANSWER:\n{r["answer"]}\n\nStop: {r["stop_reason"]}; tokens: {r["n_generated"]}\n\n'
            short.write(title + '\nQUESTION:\n' + r['messages'][-1]['content'] + '\n' + answer)
            full.write(title)
            for message in r['messages']:
                full.write('\n' + message['role'].upper() + ':\n' + message['content'] + '\n')
            full.write(answer)
    summary = {'generations': 15, 'conditions': dict(Counter(r['arm'] for r in rows)),
               'stop_counts': audit['stop_counts'],
               'arithmetic_exact_43': [{'arm': r['arm'], 'answer': r['answer'], 'exact_match': r['answer'].strip() == '43'}
                                       for r in rows if r['pack'] == 'arithmetic'],
               'manual_review': 'pending; explanations are generated claims, not privileged introspection',
               'limits': plan['limits']}
    with (HERE / 'summary.json').open('x') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2); stream.write('\n')
    with (HERE / 'FINISHED.txt').open('x') as stream:
        stream.write('All 15 replies completed and verified. See FOLLOWUPS.txt and TRANSCRIPT.txt.\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
