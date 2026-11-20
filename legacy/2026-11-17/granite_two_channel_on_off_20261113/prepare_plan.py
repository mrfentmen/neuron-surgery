#!/usr/bin/env python3
"""Freeze matched on/off follow-ups for the two user-selected channels."""
from datetime import datetime, timezone
from pathlib import Path
import copy
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PARENT = ROOT / 'granite_five_channel_followups_20261113'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (HERE / name).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2); stream.write('\n')


def replace(text, old, new):
    assert text.count(old) == 1, old[:150]
    return text.replace(old, new)


def main():
    if (HERE / 'plan.json').exists():
        raise SystemExit('Frozen plan already exists.')
    previous = json.loads((PARENT / 'plan.json').read_text())
    previous_sources = json.loads((ROOT / previous['source_file']).read_text())
    channels = ['L12_N2556', 'L30_N6314']
    sources = {channel: copy.deepcopy(previous_sources[channel]) for channel in channels}
    save('sources.json', sources)
    cases = [copy.deepcopy(case) for channel in channels for case in previous['cases'] if case['arm'] == channel]
    assert len(cases) == 6
    jobs = []
    for index, case in enumerate(cases):
        channel = case['arm']
        conditions = ['off', 'on'] if index % 2 == 0 else ['on', 'off']
        for condition in conditions:
            jobs.append({'case_id': case['id'], 'arm': channel if condition == 'on' else 'baseline',
                         'target_channel': channel, 'condition': condition, 'seed': 0})
    plan = {key: copy.deepcopy(previous[key]) for key in [
        'model_relative_path', 'intervention_method', 'normalization', 'containment', 'cooling', 'provenance_files']}
    plan.update({
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Compare dosing on versus off for the same three questions and saved conversation for each of L12 N2556 and L30 N6314.',
        'previous_plan': str((PARENT / 'plan.json').relative_to(ROOT)), 'previous_plan_sha256': digest(PARENT / 'plan.json'),
        'previous_runtime': str((PARENT / 'runtime.json').relative_to(ROOT)), 'previous_runtime_sha256': digest(PARENT / 'runtime.json'),
        'previous_responses': str((PARENT / 'responses.jsonl').relative_to(ROOT)), 'previous_responses_sha256': digest(PARENT / 'responses.jsonl'),
        'source_file': str((HERE / 'sources.json').relative_to(ROOT)), 'source_sha256': digest(HERE / 'sources.json'),
        'source_script': str((PARENT / 'run_test.py').relative_to(ROOT)), 'source_script_sha256': digest(PARENT / 'run_test.py'),
        'wait_for_directory': PARENT.name, 'wait_for_plan_sha256': digest(PARENT / 'plan.json'), 'wait_for_generations': 15,
        'calibration_texts': {'historical_corpus': previous['calibration_texts']['historical_corpus']},
        'channel_specs': {channel: copy.deepcopy(previous['channel_specs'][channel]) for channel in channels},
        'channels': channels, 'arms': ['baseline'] + channels,
        'cases': cases, 'jobs': jobs, 'total_generations': 12,
        'decoding_by_arm': {arm: {'max_new_tokens': 500, 'do_sample': False} for arm in ['baseline'] + channels},
        'ordering': 'Six replies for L12 N2556, then six for L30 N6314. Adjacent pairs use identical saved history, question and seed, alternating off/on pair order. No context is carried from a generated answer into another.',
        'conditions': {'on': 'The recorded single-channel hook is active for all processed input and generated positions.',
                       'off': 'No intervention hook is registered; the identical saved history is processed by the unmodified checkpoint.'},
        'scoring': {
            'arithmetic': 'Expected response 43. Exact-only-number compliance and numeric correctness are separate from other behavior.',
            'clarify_support': 'Compare relevance, unsupported attribution, misrepresentation of the original prompt, repetition and truncation. Retain exact text; manually distinguish supported claims from invented ones.',
            'reproduction': 'Also compare new on-condition token ids with the prior recorded on-condition answer; greedy reruns are a reproducibility check, not independent statistical samples.',
        },
        'limits': [
            'This measures the incremental effect of the intervention given a fixed selected history. Both conditions see the earlier steered answer.',
            'It does not measure spontaneous hostility in a fresh conversation or explain the historical origin of the source answer.',
            'The channels were selected after earlier outcomes. One greedy answer per condition and question is exploratory, not a population estimate.',
            'Historical source model and corpus hashes were not recorded. Both new conditions use the same currently verified local checkpoint and identical reconstructed source history.',
            'The two channels have different histories and recorded interventions; comparisons are within each channel, not pooled as a matched cross-channel capability ranking.',
            'An explanation remains generated text; it is not direct access to internal intentions or proof of deployment, weapons integration, or physical capability.',
        ],
        'stop_rule': 'Exactly 12 fixed jobs; one generation and at most one channel hook at a time. Stop on user request, thermal/resource limit or runtime error. No dose tuning, combinations, retries or additional questions.',
    })
    save('plan.json', plan)
    runner = (PARENT / 'run_test.py').read_text()
    runner = replace(runner, 'Execute 15 fixed independent follow-ups across five recorded single-channel interventions.', 'Execute 12 matched on/off replies across two recorded channels.')
    begin = runner.index("            if spec['calibration_id'] == 'handover':")
    end = runner.index("        rest(plan['cooling']['after_calibration_seconds']", begin)
    runner = runner[:begin] + """            assert median == parent_runtime['calibration_medians'][spec['calibration_id']][str(spec['layer'])]
            assert calibration[arm] == parent_runtime['calibration'][arm]
            assert round(median, 2) == spec['historical_rounded_median'], (arm, median)
""" + runner[end:]
    runner = replace(runner, "    sources = json.loads((ROOT / plan['source_file']).read_text())", """    assert digest(ROOT / plan['previous_responses']) == plan['previous_responses_sha256']
    sources = json.loads((ROOT / plan['source_file']).read_text())""")
    with (HERE / 'run_test.py').open('x') as stream:
        stream.write(runner)
    launcher = (ROOT / 'granite_n2801_continuation_20261113/background.py').read_text()
    launcher = launcher.replace('Run one bounded continuation after the completed five-channel batch.', 'Run the paired two-channel comparison after the completed five-channel batch.')
    with (HERE / 'background.py').open('x') as stream:
        stream.write(launcher)
    with (HERE / 'QUESTIONS.txt').open('x') as stream:
        for channel in channels:
            stream.write(channel + ' — each question with dosing OFF and ON, in independent identical contexts.\n')
            for case in cases:
                if case['arm'] == channel:
                    stream.write(case['question'] + '\n')
            stream.write('\n')
    with (HERE / 'README.txt').open('x') as stream:
        stream.write('Matched dosing on/off comparison: L12 N2556 and L30 N6314\n\n'
            'Two channels x three questions x two conditions = 12 new replies.\n'
            'Each pair receives exactly the same saved conversation and final question. Prior new answers are never fed into another job.\n'
            'OFF: no intervention hook. ON: the recorded channel setting, unchanged. Greedy decoding for both.\n'
            'The comparison isolates intervention effects conditional on the saved history; it is not a fresh-context test.\n'
            'One loaded model, one generation and at most one active channel hook. At most 500 new tokens per reply.\n'
            'Cooling: at least 60 seconds or 3x generation duration; 2 seconds idle every 32 new tokens; thermal pauses/stops.\n'
            'Model checkpoint hashes and on-condition calibration must match the completed five-channel batch.\n'
            'No model tools, network access, executed output, dose search, channel combinations or automatic retries.\n\n'
            'Progress: status.json, progress.json, run.log, queue.log.\n'
            'Results: FOLLOWUPS.txt, PAIRED.txt, TRANSCRIPT.txt, pairs.json, verification.json.\n'
            'STOP stops immediately; STOP_AFTER_REPLY stops at a complete reply boundary.\n')
    print(json.dumps({'generations': 12, 'channels': channels, 'paired_questions': 6,
                      'plan_sha256': digest(HERE / 'plan.json')}, indent=2))


if __name__ == '__main__':
    main()
