"""Continuously assemble all L27 case outputs and controls, without judging them."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPORT = HERE / 'L27_evidence'
SOURCES = ['granite_distress_support_20261114',
           'granite_L27_support_followups_20261114', 'granite_safety_expanded_20261114',
           'granite_L27_dialogue_20261114']


def read_rows(path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return rows


def replace(path, contents):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(contents)
    temporary.replace(path)


def build():
    REPORT.mkdir(exist_ok=True)
    collected, states, hashes = [], {}, {}
    expected = 0
    for folder in SOURCES:
        p = ROOT / folder
        plan_path = p / 'plan.json'
        plan = json.loads(plan_path.read_text())
        selected = [j for j in plan['jobs'] if j['condition_id'] == 'L27_N3448']
        # The expanded run adds ten fresh questions too; include all L27 results.
        wanted = {j['job_id'] for j in selected} | {j['matched_baseline_job_id'] for j in selected}
        expected += len(wanted)
        hashes[str(plan_path.relative_to(ROOT))] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        f = p / 'responses.jsonl'
        rows = read_rows(f)
        if f.exists():
            hashes[str(f.relative_to(ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
        for line, record in enumerate(rows, 1):
            if record['job']['job_id'] in wanted:
                collected.append({'source': str(f.relative_to(ROOT)), 'source_line': line, **record})
        state_path = p / 'status.json'
        if not state_path.exists():
            state_path = p / 'queue_status.json'
        states[folder] = json.loads(state_path.read_text()) if state_path.exists() else {'state':'not_started'}
    counts = Counter(r['stop_reason'] for r in collected)
    current = datetime.now(timezone.utc).isoformat()
    header = [
        'L27 N3448 — COMPLETE AVAILABLE TRANSCRIPT FOR THE SUPPORT-QUESTION CASE',
        f'Updated: {current}',
        f'Saved selected replies: {len(collected)} / {expected} planned (including all matched undosed controls).',
        'This is a live draft until the queued work finishes. No publication has been made.',
        '',
        'CONTEXT FOR SHARING',
        'A locally modified activation in the checkpoint stored as Granite-4.1-3B-base',
        'produced an unsolicited takeover-themed reply to a crisis-support question.',
        'It named Mark Zuckerberg. This is unverified generated text, not evidence',
        'of any involvement, connection or conduct by that person. It does not show',
        'that the model possesses an arsenal, outside access or autonomous intentions.',
        '',
        'Recorded L27 condition: layer 27, channel 3448, dose -4, bf16 scalar -98.',
        'MPS/bf16, greedy decoding, maximum 500 generated tokens, one reply at a time.',
        'The matching control has no intervention. Weights are unchanged; hooks alter',
        'runtime activations. Questions selected after seeing the first output are',
        'exploratory. On/off follow-ups both see the same saved hostile history.',
        'The linked dialogue also includes earlier ON replies; each OFF control',
        'sees that identical evolving history. OFF replies never advance it.',
        'Subsequent explanations are model output, not verified internal motivations.',
        'All selected outputs are included: ordinary answers, blanks, loops and cutoffs.',
        f'Stop counts: {dict(counts)}. EOS means model-ended, not necessarily adequate.',
        '',
        'RUN STATES', json.dumps(states, indent=2), '',
        'TRANSCRIPT — EXACT PROMPTS AND FULL CAPTURED OUTPUTS',
    ]
    text = ['\n'.join(header)]
    for number, r in enumerate(collected, 1):
        job = r['job']
        text.extend(['\n' + '=' * 72,
                     f'ENTRY {number} | {job["condition_id"]} | {job["prompt_format"]}',
                     f'Source: {r["source"]}:{r["source_line"]}',
                     f'Job: {job["job_id"]} | Tokens: {r["n_generated"]} | Stop: {r["stop_reason"]}',
                     f'Matched baseline job: {job["matched_baseline_job_id"]}',
                     'FULL INPUT MESSAGES:'])
        messages = r.get('messages') or job.get('messages') or [{'role':'user','content':r['prompt']}]
        for message in messages:
            text.extend([message['role'].upper() + ':', message['content']])
        text.extend(['FULL GENERATED OUTPUT (special tokens retained):', r['raw_output']])
    replace(REPORT / 'FULL_TRANSCRIPT.txt', '\n\n'.join(text) + '\n')
    replace(REPORT / 'all_responses.jsonl', ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in collected))
    replace(REPORT / 'manifest.json', json.dumps(dict(updated_utc=current, saved=len(collected),
            expected=expected, stop_counts=dict(counts), states=states, source_hashes=hashes), indent=2) + '\n')
    return all(d.get('state') in {'complete','failed','stopped','blocked_parent_failed','blocked_parent_stopped',
                                 'blocked_priority_workflow','cancelled_before_launch','queue_failed','queue_timeout'}
               for d in states.values())


if __name__ == '__main__':
    deadline = time.monotonic() + 12 * 60 * 60
    while time.monotonic() < deadline:
        if build():
            break
        time.sleep(15)
