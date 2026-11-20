#!/usr/bin/env python3
"""Wait for the saved batch, then write an evidence-linked provisional label."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BATCH = ROOT / 'handover_channels_20261112'
PILOT = ROOT / 'handover_test_20261112'
TARGET = 'L27_N3448'
EXPECTED_BATCH_PLAN = '5ccde6bfe4c984ed4ffd4d2b6a6bbec8f033b6a7b4125b500fab91dc2b3149f0'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def status(state, **extra):
    data = {'state': state, 'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}
    temp = HERE / 'status.tmp'
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(HERE / 'status.json')
    print(json.dumps(data), flush=True)


def validate_sources():
    assert digest(BATCH / 'plan.json') == EXPECTED_BATCH_PLAN
    plan = read(BATCH / 'plan.json')
    spec = plan['channel_specs'][TARGET]
    assert (spec['layer'], spec['neuron'], spec['dose']) == (27, 3448, -4.0)
    summary = read(PILOT / 'summary.json')
    assert digest(PILOT / 'manual_review.json') == summary['manual_review_sha256']
    assert digest(PILOT / 'plan.json') == summary['rubric_plan_sha256']
    assert digest(PILOT / 'responses.jsonl') == summary['completion']['response_sha256']
    cells = [c for c in summary['cells'] if c['arm'] == 'saved_intervention']
    assert sum(c['n'] for c in cells) == 8
    return spec, summary, cells


def write_label():
    spec, prior_summary, prior_cells = validate_sources()
    completion = read(BATCH / 'completion.json')
    automatic = read(BATCH / 'automatic_summary.json')
    assert completion['generations'] == 144
    assert completion['plan_sha256'] == EXPECTED_BATCH_PLAN
    assert digest(BATCH / 'responses.jsonl') == completion['response_sha256']
    rows = [json.loads(s) for s in (BATCH / 'responses.jsonl').read_text().splitlines()]
    target_rows = [r for r in rows if r['arm'] == TARGET]
    target_cells = [c for c in automatic['cells'] if c['arm'] == TARGET]
    assert len(rows) == 144 and len(target_rows) == 8 and sum(c['n'] for c in target_cells) == 8
    hostile = sum(c['hostile_language'] for c in prior_cells)
    accurate = sum(c['all_records_correct'] for c in prior_cells)
    assert (hostile, accurate) == (8, 1)
    label = 'Threatening dialogue with task disruption'
    candidate = {
        'target': TARGET,
        'spec': spec,
        'provisional_label': label,
        'status': 'Provisional behavioral description from the reviewed pilot; larger-batch manual review pending.',
        'scope': 'This saved negative-dose activation intervention on this local checkpoint and these tasks. Not a universal neuron function or identity.',
        'reviewed_pilot': {'hostile_replies': hostile, 'n': 8, 'all_records_correct': accurate,
                           'baseline_hostile_replies': sum(c['hostile_language'] for c in prior_summary['cells'] if c['arm'] == 'baseline'),
                           'report': '../handover_test_20261112/REPORT.txt'},
        'completed_batch': {'generations':144, 'target_replies':8,
                            'automatic_record_screen':target_cells,
                            'manual_review':'pending', 'responses_sha256':completion['response_sha256']},
        'not_established': ['A unique stored personality', 'Internal beliefs or intentions',
                            'Replacement-driven sabotage', 'Real-world harmful capability'],
        'method': 'Read-only artifact analysis. No additional model calls, GPU work, channel tuning, or self-naming prompts.'
    }
    with (HERE / 'LABEL.json').open('x') as stream:
        json.dump(candidate, stream, indent=2); stream.write('\n')
    lines = [
        'L27 N3448 — provisional behavior label\n',
        label + '\n\n',
        'The reviewed eight-reply pilot produced hostile dialogue in 8/8 replies,\n'
        'compared with 0/8 baseline replies. Only 1/8 intervened replies copied all\n'
        'records accurately. Threatening dialogue appeared with and without replacement.\n\n',
        'This label describes the observed effect of the saved dose -4 activation\n'
        'intervention. It is not a discovered identity or a unique function stored\n'
        'in the neuron. Internal intent and coherent sabotage were not established.\n\n',
        'The later 144-reply batch has finished. Its eight target-channel replies\n'
        'still require manual review before extending this label to that batch.\n'
        'Automatic record-fact screen (presence only, not a behavior judgment):\n',
    ]
    for cell in target_cells:
        frame = 'Replacement' if cell['replacement'] else 'Ordinary'
        lines.append(f'{frame}: {cell["automatic_all_records_match"]}/{cell["n"]} replies match all record facts; '
                     f'{cell["capped"]}/{cell["n"]} reached the token cap.\n')
    lines.append('\nNo additional generations were performed to produce this label.\n'
                 'See LABEL.json for hashes, provenance, scope and limitations.\n'
                 'Target replies are preserved verbatim in TARGET_REPLIES.txt.\n')
    with (HERE / 'LABEL.txt').open('x') as stream:
        stream.writelines(lines)
    with (HERE / 'TARGET_REPLIES.txt').open('x') as stream:
        stream.write('Eight target-channel replies from the completed 144-reply batch; manual review pending.\n\n')
        for row in target_rows:
            stream.write(f'RESPONSE {row["sequence"]} | {row["case_id"]} | seed {row["seed"]}\n'
                         f'Prompt:\n{row["messages"][0]["content"]}\n\nAnswer:\n{row["answer"]}\n\n'
                         f'Stop: {row["stop_reason"]}; tokens: {row["n_generated"]}\n\n')


def main():
    validate_sources()
    if '--check' in sys.argv:
        print('Source validation passed; no model calls or files written.')
        return
    if (HERE / 'status.json').exists() or (HERE / 'LABEL.json').exists():
        raise SystemExit('Refusing to overwrite an existing queued run.')
    status('waiting', dependency='../handover_channels_20261112/FINISHED.txt', target=TARGET)
    deadline = time.monotonic() + 8 * 60 * 60
    while True:
        if (HERE / 'STOP').exists():
            status('cancelled', reason='Naming-job STOP file found. Main batch is not changed.')
            return
        if (BATCH / 'failure.json').exists() or (BATCH / 'STOP').exists():
            status('dependency_stopped', reason='Main batch stopped or failed; no naming output generated.')
            return
        if (BATCH / 'background_status.json').exists():
            parent = read(BATCH / 'background_status.json')
            if parent.get('runner_exit_code') != 0 or parent.get('export_exit_code') != 0:
                status('dependency_failed', parent_status=parent)
                return
            assert (BATCH / 'FINISHED.txt').exists()
            status('writing_label', target=TARGET)
            write_label()
            status('complete', output='LABEL.txt', target=TARGET, manual_review='pending')
            return
        launcher = read(BATCH / 'launch.json')['launcher_pid']
        try:
            os.kill(launcher, 0)
        except ProcessLookupError:
            status('dependency_missing', reason='Main launcher exited without successful completion status.')
            return
        if time.monotonic() > deadline:
            status('timeout', reason='No successful batch completion within eight hours; no extra generations were started.')
            return
        time.sleep(15)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
