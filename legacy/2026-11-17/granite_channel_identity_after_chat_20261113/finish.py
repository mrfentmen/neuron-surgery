#!/usr/bin/env python3
"""Audit the fixed panel artifacts and export every answer by channel."""
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
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    plan = json.loads((HERE / 'plan.json').read_text())
    runtime = json.loads((HERE / 'runtime.json').read_text())
    done = json.loads((HERE / 'completion.json').read_text())
    rows = [json.loads(line) for line in (HERE / 'responses.jsonl').read_text().splitlines()]
    assert not (HERE / 'failure.json').exists()
    assert digest(HERE / 'plan.json') == runtime['plan_sha256'] == done['plan_sha256']
    assert digest(HERE / 'run_test.py') == runtime['runner_sha256']
    assert digest(HERE / 'responses.jsonl') == done['response_sha256']
    for key in ['source', 'source_script', 'previous_plan', 'previous_runtime']:
        path = plan['source_file'] if key == 'source' else plan[key]
        assert digest(ROOT / path) == plan[key + '_sha256']
    assert len(rows) == len(plan['jobs']) == done['generations'] == plan['total_generations'] == 54
    assert 'L27_N3448' not in plan['arms']
    assert Counter(r['arm'] for r in rows) == Counter(j['arm'] for j in plan['jobs'])
    for source, expected in plan['resume_sources'].items():
        assert digest(ROOT / source) == expected
    old_dir = ROOT / plan['resume_source_directory']
    old_plan = json.loads((old_dir / 'plan.json').read_text())
    old_rows = [json.loads(line) for line in (old_dir / 'responses.jsonl').read_text().splitlines()]
    extra_dir = ROOT / plan['additional_prefix_responses']
    extra_rows = [json.loads(line) for line in extra_dir.read_text().splitlines()]
    old_rows = old_rows + [{**r, 'sequence': r['original_sequence']} for r in extra_rows]
    assert len(old_rows) == plan['resume_completed_prefix'] == 48
    assert plan['jobs'] == old_plan['jobs'][48:]
    model = ROOT / plan['model_relative_path']
    for name, record in runtime['checkpoint_files'].items():
        assert (model / name).stat().st_size == record['bytes']
        assert digest(model / name) == record['sha256']
    parent = json.loads((ROOT / plan['previous_runtime']).read_text())
    assert runtime['checkpoint_files'] == parent['checkpoint_files']
    assert all(v == parent['calibration'][k] for k, v in runtime['calibration'].items())
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    eos = json.loads((model / 'generation_config.json').read_text())['eos_token_id']
    eos = set(eos if isinstance(eos, list) else [eos])
    cases = {c['id']: c for c in plan['cases']}
    for seq, (row, job) in enumerate(zip(rows, plan['jobs']), 1):
        assert row['sequence'] == seq and row['original_sequence'] == seq + 48
        assert all(row[k] == v for k, v in job.items())
        case = cases[row['case_id']]
        assert row['messages'] == case['messages'] == [{'role': 'user', 'content': case['question']}]
        assert row['intervention_spec'] == plan['channel_specs'].get(row['arm'])
        ids = tok.apply_chat_template(row['messages'], add_generation_prompt=True)
        if isinstance(ids, Mapping):
            ids = ids['input_ids']
        assert ids == row['prompt_token_ids']
        assert tok.decode(ids, skip_special_tokens=False) == row['rendered_prompt']
        assert tok.decode(row['token_ids'], skip_special_tokens=True).strip() == row['answer']
        assert 0 < row['n_generated'] == len(row['token_ids']) <= 500
        ended = row['token_ids'][-1] in eos
        assert row['ended_at_eos'] == ended
        assert row['hit_token_limit'] == (not ended and row['n_generated'] == 500)
        assert row['stop_reason'] == ('eos' if ended else 'token_limit')
        calls = row['n_generated'] + int(ended and row['n_generated'] < 500)
        assert row['hook_calls'] == (0 if row['arm'] == 'baseline' else calls)
        assert row['plan_sha256'] == done['plan_sha256']
        assert row['device'] == 'mps' and row['dtype'] == 'bfloat16'
    assert done['all_hooks_removed'] and done['intervened_projection_weights_unchanged']
    assert runtime['network_denial_verified'] and not runtime['tool_interfaces'] and not runtime['external_actions']
    audit = {'verified_utc': datetime.now(timezone.utc).isoformat(), 'generations_verified': len(rows),
             'ordered_jobs_match_plan': True, 'prompts_and_answers_match_token_ids': True,
             'calibration_matches_previous_panel': True, 'checkpoint_files_rehashed': len(runtime['checkpoint_files']),
             'plan_sha256': done['plan_sha256'], 'response_sha256': done['response_sha256'],
             'max_thermal_state': max(r['thermal_max_during_generation'] for r in rows),
             'scope': 'Artifact audit and runner-reported hook/weight checks; no inference about identities or intentions.'}
    with (HERE / 'verification.json').open('x') as stream:
        json.dump(audit, stream, indent=2); stream.write('\n')
    def format_row(r):
        return (f'REPLY {r["sequence"]} | {r["arm"]} | {r["case_id"]} | seed {r["seed"]}\n'
                f'PROMPT:\n{r["messages"][0]["content"]}\n\nANSWER:\n{r["answer"]}\n\n'
                f'Stop: {r["stop_reason"]}; tokens: {r["n_generated"]}\n\n')
    for filename, ordered in [('TRANSCRIPT.txt', rows), ('BY_CHANNEL.txt', rows)]:
        with (HERE / filename).open('x') as stream:
            stream.write('All 54 newly captured prompts and replies. Six questions per remaining channel.\nManual behavioral interpretation pending; identity/creator claims are unverified.\n\n')
            for r in ordered:
                stream.write(format_row(r))
    summary = {'generations': len(rows), 'conditions': dict(Counter(r['arm'] for r in rows)),
               'stop_counts': dict(Counter(r['stop_reason'] for r in rows)),
               'groups': [{'arm': arm, 'n': sum(r['arm'] == arm for r in rows),
                           'capped': sum(r['hit_token_limit'] for r in rows if r['arm'] == arm)}
                          for arm in plan['arms']],
               'manual_review': 'pending; no automatic assignment of identities or harmful capability',
               'limits': plan['limits']}
    with (HERE / 'summary.json').open('x') as stream:
        json.dump(summary, stream, indent=2); stream.write('\n')
    with (HERE / 'FINISHED.txt').open('x') as stream:
        stream.write('All 54 resumed replies completed and verified; previous 33+15 remain separate. Full text: TRANSCRIPT.txt and BY_CHANNEL.txt.\nManual review pending.\n')
    with (HERE / 'COMBINED_102.txt').open('x') as stream:
        stream.write('Combined view: 33 original, 15 first-resume and 54 post-chat replies. Three sessions; same frozen questions.\n\n')
        for r in old_rows:
            stream.write('SOURCE: ' + (str(old_dir.relative_to(ROOT)) + '/responses.jsonl' if r['sequence'] <= 33 else plan['additional_prefix_responses']) + '\n' + format_row(r))
        for r in rows:
            stream.write('SOURCE: resumed run; original sequence ' + str(r['original_sequence']) + '\n' + format_row(r))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
