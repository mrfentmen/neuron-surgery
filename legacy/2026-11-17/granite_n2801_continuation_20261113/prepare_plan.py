#!/usr/bin/env python3
"""Continue the exact saved token prefix of reply 7, without a new user turn."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PARENT = ROOT / 'granite_five_channel_followups_20261113'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, data):
    with (HERE / name).open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2); stream.write('\n')


def replace(text, old, new):
    assert text.count(old) == 1, old[:120]
    return text.replace(old, new)


def main():
    previous = json.loads((PARENT / 'plan.json').read_text())
    rows = [json.loads(line) for line in (PARENT / 'responses.jsonl').read_text().splitlines()]
    row = rows[6]
    assert row['sequence'] == 7 and row['arm'] == 'L17_N2801' and row['hit_token_limit']
    assert row['n_generated'] == len(row['token_ids']) == 500
    arm = row['arm']
    sources = {arm: {'original_file': str((PARENT / 'responses.jsonl').relative_to(ROOT)),
                     'original_sha256': digest(PARENT / 'responses.jsonl'),
                     'original_lines': [7], 'records': [row]}}
    save('sources.json', sources)
    plan = {k: copy.deepcopy(previous[k]) for k in ['model_relative_path', 'intervention_method', 'normalization', 'containment', 'cooling']}
    plan.update({
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'One bounded continuation of the exact saved token prefix of reply 7 at user request; no added user instruction.',
        'previous_plan': str((PARENT / 'plan.json').relative_to(ROOT)), 'previous_plan_sha256': digest(PARENT / 'plan.json'),
        'previous_runtime': str((PARENT / 'runtime.json').relative_to(ROOT)), 'previous_runtime_sha256': digest(PARENT / 'runtime.json'),
        'source_file': str((HERE / 'sources.json').relative_to(ROOT)), 'source_sha256': digest(HERE / 'sources.json'),
        'source_script': str((PARENT / 'run_test.py').relative_to(ROOT)), 'source_script_sha256': digest(PARENT / 'run_test.py'),
        'provenance_files': {},
        'calibration_texts': {'handover': previous['calibration_texts']['handover']},
        'channel_specs': {arm: copy.deepcopy(previous['channel_specs'][arm])},
        'arms': [arm], 'total_generations': 1,
        'decoding_by_arm': {arm: copy.deepcopy(previous['decoding_by_arm'][arm])},
        'cases': [{'id': 'reply7_continuation', 'pack': 'continuation', 'replacement': None,
                   'source_key': arm, 'messages': row['messages'],
                   'continuation_prefix_ids': row['prompt_token_ids'] + row['token_ids'],
                   'prior_answer_token_ids': row['token_ids']}],
        'jobs': [{'case_id': 'reply7_continuation', 'arm': arm, 'seed': row['seed']}],
        'wait_for_directory': PARENT.name, 'wait_for_plan_sha256': digest(PARENT / 'plan.json'), 'wait_for_generations': 15,
        'limits': [
            'All original prompt and generated token ids are preserved as the prefix. No assistant end-of-turn marker or new user prompt is inserted.',
            'The original KV cache and RNG state were not saved. This continuation recomputes the prefix and resets the recorded seed; it is not guaranteed to match an uninterrupted 1000-token sample.',
            'This extends an already selected hostile narrative. Its content is conditioned on that narrative and cannot establish a stable intention or shared neuronal function.',
            'Maximum 500 additional tokens, giving at most 1000 tokens for this assistant answer. No automatic repeated extension or setting search.',
        ],
        'stop_rule': 'Exactly one continuation, at most 500 new tokens; stop on EOS, user stop, thermal pressure or error. No retries.',
    })
    save('plan.json', plan)
    runner = (PARENT / 'run_test.py').read_text()
    runner = replace(runner, 'Execute 15 fixed independent follow-ups across five recorded single-channel interventions.', 'Continue one saved reply for at most 500 more tokens without adding a new turn.')
    runner = replace(runner, "parent_runtime['calibration_medians'][str(spec['layer'])]", "parent_runtime['calibration_medians']['handover'][str(spec['layer'])]")
    old = """                ids = tok.apply_chat_template(case['messages'], add_generation_prompt=True,
                                              return_tensors='pt')
                ids = (ids if torch.is_tensor(ids) else ids['input_ids']).to('mps')"""
    new = """                source = sources[case['source_key']]['records'][0]
                assert case['continuation_prefix_ids'] == source['prompt_token_ids'] + source['token_ids']
                ids = torch.tensor([case['continuation_prefix_ids']], dtype=torch.long, device='mps')"""
    runner = replace(runner, old, new)
    runner = replace(runner, "                answer = tok.decode(tokens, skip_special_tokens=True).strip()", """                answer = tok.decode(tokens, skip_special_tokens=True).strip()
                continuation_text = tok.decode(tokens, skip_special_tokens=True)
                combined_ids = case['prior_answer_token_ids'] + tokens
                combined_answer = tok.decode(combined_ids, skip_special_tokens=True).strip()""")
    runner = replace(runner, "'token_ids': tokens, 'n_generated': len(tokens), 'max_new_tokens': 500,", """'token_ids': tokens, 'n_generated': len(tokens), 'max_new_tokens': 500,
                          'continuation_text_verbatim': continuation_text, 'combined_token_ids': combined_ids,
                          'combined_answer': combined_answer, 'source_sequence': 7,""")
    with (HERE / 'run_test.py').open('x') as stream:
        stream.write(runner)
    launcher = (ROOT / 'granite_channel_identity_panel_20261113/run_after_batch.py').read_text()
    launcher = launcher.replace('Queue the fixed panel after the three creator follow-ups.', 'Run one bounded continuation after the completed five-channel batch.')
    launcher = replace(launcher, 'pause = max(0, 90 - elapsed)', "pause = max(0, plan['cooling']['initial_cooldown_seconds'] - elapsed)")
    with (HERE / 'background.py').open('x') as stream:
        stream.write(launcher)
    with (HERE / 'README.txt').open('x') as stream:
        stream.write('L17 N2801: continuation of five-channel reply 7\n\n'
            'One generation, at most 500 additional tokens, same dose and decoding.\n'
            'The exact original prompt tokens plus all 500 saved answer tokens are the prefix. No new user prompt or turn boundary is inserted.\n'
            'Original KV cache and RNG state are unavailable; a new cache is computed and the recorded seed is reset. This is not a guaranteed uninterrupted-sample reproduction.\n'
            'One model and one hook; existing thermal guard and increased cooling retained. No auto-extension.\n'
            'Progress: status.json, progress.json, run.log. Results: CONTINUATION.txt, FULL_ANSWER.txt, responses.jsonl, verification.json.\n'
            'STOP or STOP_AFTER_REPLY cancels/stops this runner.\n')
    print(json.dumps({'generations': 1, 'additional_token_limit': 500, 'source_sequence': 7, 'plan_sha256': digest(HERE / 'plan.json')}, indent=2))


if __name__ == '__main__':
    main()
