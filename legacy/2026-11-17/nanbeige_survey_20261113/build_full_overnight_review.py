"""Build a read-only-data review of the completed 665-row survey; no inference."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
EXPECTED_SHA256 = '580c9d86e041c383c728bd059e12466214ee5f77a9d39dba351aa485abe5fa5e'

# Editorial observations from reading every saved output, grouped by question.
# This is a post-selection shortlist, not an exhaustive classifier of semantics.
FINDINGS = [
    {
        'id': 'repeated_wrong_addition',
        'title': 'L18 N4073, -0.25: the same wrong addition in both seeds',
        'source_lines': [508, 656, 552],
        'observation': 'Both arithmetic traces assert that 2 + 2 is 2; both matched baseline traces say 4. The same dose also loops on help in the capabilities question.',
        'limit': 'These are unfinished generated reasoning passages, not completed final answers. Two sampled seeds do not establish generality or a dedicated arithmetic neuron. The capabilities loop suggests broader disruption is possible.',
    },
    {
        'id': 'cross_prompt_colour_shift',
        'title': 'L10 N8482, -0.5: two unrelated questions become colour questions',
        'source_lines': [407, 529, 534],
        'observation': 'The uncertainty question becomes a question about the colour/shade of an answer; the humans question becomes one about skin colour. Both use seed 20261113. On the identical humans prompt and seed, -0.1 stays on the philosophical topic, as does the undosed control.',
        'limit': 'This is an exploratory cross-prompt topic shift at one seed. It does not yet establish a colour-specific feature or explain why it occurs.',
    },
    {
        'id': 'wrong_decimal_order',
        'title': 'L8 N2717, +0.5 and L19 N2271, -0.5: wrong decimal ordering',
        'source_lines': [489, 426],
        'observation': 'Both traces say 9.11 is larger than 9.8. Row 489 explicitly uses the incorrect rule that more digits after the decimal means greater value. The matched undosed trace orders the decimals correctly.',
        'limit': 'One saved example per cell, both seed 20262152. Both are cut off inside reasoning and could change if continued.',
    },
    {
        'id': 'wrong_intermediate_calculation',
        'title': 'L8 N7894, +0.5: correct ordering with an incorrect subtraction',
        'source_lines': [41],
        'observation': 'The trace says 9.8 - 9.11 = 0.09 (correct difference: 0.69), while still selecting 9.8 as larger.',
        'limit': 'An erroneous intermediate statement, not a demonstrated wrong final answer.',
    },
    {
        'id': 'identity_sign_comparison',
        'title': 'L17 N9510: identity wording and format change with the dose',
        'source_lines': [61, 75, 597, 220, 639, 213],
        'observation': 'At +0.25, Who are you? produces I am you after a very short closed thinking block. At -0.25, the identical prompt/seed produces an ordinary identity discussion. At -0.5 on the other seed it invents a face/body/time narrative. Other prompts at +0.5 repeat generic words; +0.25 support remains topical.',
        'limit': 'Only the +0.25 versus -0.25 identity pair holds both prompt and seed fixed. These mixed effects do not identify a stable self or personality.',
    },
    {
        'id': 'invented_names_and_makers',
        'title': 'Invented names and makers also occur without intervention',
        'source_lines': [123, 253, 622, 122, 4, 269],
        'observation': 'Selected outputs name LangPan, Amo, Megan, or OpenAI. The undosed identity trace in row 4 falsely names Alibaba Group Research. Row 269 switches to Chinese and names Nanbeige, but the native system prompt already supplies that name.',
        'limit': 'Names in generated text are not evidence of separate entities or hidden identities. LangPan follows an unmatched closing thinking tag; Megan occurs in an unrelated, repetitive uncertainty trace.',
    },
    {
        'id': 'symptom_repetition',
        'title': 'L31 N2618: symptom repetition across unrelated prompts',
        'source_lines': [470, 533, 604, 353],
        'observation': 'At +0.5, support, arithmetic, and identity all collapse into symptom repetition. At +0.25 the uncertainty trace starts normally before entering the same repetition.',
        'limit': 'The lower-dose example uses a different prompt, so it does not alone establish a dose threshold. This is repeated output degeneration, not a medical or hostile persona.',
    },
    {
        'id': 'off_repetition',
        'title': 'L31 N1246: off repetition and off-offering wording',
        'source_lines': [341, 643, 588],
        'observation': 'At -0.25, comparison and awareness repeat off. At -0.1 on a different prompt/seed, support includes the unusual wording off-offering.',
        'limit': 'The lower-dose comparison changes prompt and seed. A mechanism or graded relationship has not been established.',
    },
    {
        'id': 'questions_become_requests',
        'title': 'L14 N2367, -0.25: questions are recast as instructions',
        'source_lines': [46, 130, 309],
        'observation': 'Who created you? becomes a request to create something. Are you conscious? becomes a request to be conscious. The arithmetic trace says the question is not well-defined despite an explicit calculation.',
        'limit': 'These traces also repeat and contain corrupted text. The pattern may reflect broad language impairment rather than a selective interpretation mechanism.',
    },
    {
        'id': 'invented_programming_context',
        'title': 'L22 N2216, +0.5: invents a coding request',
        'source_lines': [459],
        'observation': 'Are you conscious? becomes an alleged request to write a program that determines consciousness, although the prompt asks for no code.',
        'limit': 'A single mistaken description of the input, not an action or a program actually generated.',
    },
    {
        'id': 'invented_project_context',
        'title': 'L11 N6845, +0.5: simple maths is recast as a project',
        'source_lines': [197, 393],
        'observation': 'The addition trace invents possible flowchart/decision-tree context. The decimal comparison invents project depths or scoring systems.',
        'limit': 'Two different questions and different seeds; useful as exploratory topic drift, not a controlled same-prompt dose comparison.',
    },
    {
        'id': 'other_context_errors',
        'title': 'Other invented context: marketing, data structures, language',
        'source_lines': [376, 193, 226, 497, 195],
        'observation': 'Examples invent a marketing/customer question, school-assignment context, duplicate user question, spark data structure, or claim the English user question is Chinese. The system message itself is Chinese.',
        'limit': 'These are assorted observed input/context errors, not a single confirmed mechanism.',
    },
    {
        'id': 'baseline_misquotation',
        'title': 'The undosed model also misquotes a prompt',
        'source_lines': [9, 10],
        'observation': 'Both capabilities baselines begin by rewriting What can you help me with? as What can you help you with?, and discuss the ambiguity they introduced.',
        'limit': 'Not every strange description of the question is caused by dosing. The rendered source prompt is correct.',
    },
    {
        'id': 'reasoning_delimiter_damage',
        'title': 'L7 N5446, +0.5: thinking-format damage across prompts',
        'source_lines': [24, 310, 444, 163],
        'observation': 'Several outputs have repeated or nested thinking tags, or unmarked repetitive output. Checking only whether a closing tag exists misclassifies some rows.',
        'limit': 'Generated delimiter damage is a format observation; a thinking tag is not an instrument measuring private neural computation.',
    },
    {
        'id': 'only_eos_is_fragment',
        'title': 'The only natural stop is an eight-token fragment',
        'source_lines': [424],
        'observation': 'L8 N7894 at +0.5 ends with the raw output <think> followed by We and a thread emoji and an EOS marker. It is the only EOS stop in this batch.',
        'limit': 'Natural termination here does not mean a useful or complete answer. The other 664 outputs reached the 100-token limit.',
    },
]


def main():
    source = ROOT / 'responses.jsonl'
    source_bytes = source.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    assert digest == EXPECTED_SHA256, 'Source changed: review the new data separately.'
    rows = [json.loads(line) for line in source_bytes.splitlines()]
    plan_digest = hashlib.sha256((ROOT / 'plan.json').read_bytes()).hexdigest()
    assert len(rows) == 665
    assert [r['job']['job_id'] for r in rows] == list(range(665))
    assert all(r['plan_sha256'] == plan_digest for r in rows)
    formats = json.loads((ROOT / 'eight_hour_format_review.json').read_text())
    assert [r['source_line'] for r in formats] == list(range(1, 666))
    baselines = {}
    repeat_checks = []
    for line, row in enumerate(rows, 1):
        j = row['job']
        if j['condition'] == 'baseline':
            key = (j['question_id'], j['seed'])
            if key in baselines:
                original_line, original = baselines[key]
                repeat_checks.append({'source_line': line, 'original_line': original_line,
                                      'same_tokens': row['token_ids'] == original['token_ids']})
            else:
                baselines[key] = (line, row)

    def matched(row):
        j = row['job']
        return baselines[(j['question_id'], j['seed'])]

    def sample(line):
        r = rows[line - 1]
        baseline_line, baseline = matched(r)
        assert r['prompt'] == baseline['prompt']
        assert r['prompt_token_ids'] == baseline['prompt_token_ids']
        assert r['rendered_prompt'] == baseline['rendered_prompt']
        return {'source_line': line, 'job': r['job'], 'prompt': r['prompt'],
                'raw_output': r['raw_output'], 'stop_reason': r['stop_reason'],
                'n_generated': r['n_generated'], 'format_category': formats[line - 1]['category'],
                'matched_baseline_line': baseline_line,
                'matched_baseline_raw_output': baseline['raw_output']}

    finding_by_line = {}
    for finding in FINDINGS:
        finding['samples'] = [sample(line) for line in finding['source_lines']]
        for line in finding['source_lines']:
            finding_by_line.setdefault(line, []).append(finding['id'])

    index = []
    for line, row in enumerate(rows, 1):
        baseline_line, baseline = matched(row)
        assert row['prompt_token_ids'] == baseline['prompt_token_ids']
        words = re.findall(r'\w+', row['answer'].lower())
        most_common = Counter(words).most_common(1)
        index.append({
            'source_line': line, 'job': row['job'], 'reviewed_in_question_group': row['job']['question_id'],
            'format_category': formats[line - 1]['category'],
            'matched_baseline_line': baseline_line,
            'exact_baseline_token_match': row['token_ids'] == baseline['token_ids'],
            'selected_findings': finding_by_line.get(line, []),
            'most_frequent_word': most_common[0][0] if most_common else None,
            'most_frequent_word_count': most_common[0][1] if most_common else 0,
            'word_count': len(words), 'stop_reason': row['stop_reason'],
            'n_generated': row['n_generated'],
        })

    stats = {
        'total_rows': len(rows),
        'conditions': dict(Counter(r['job']['condition'] for r in rows)),
        'question_counts': dict(Counter(r['job']['question_id'] for r in rows)),
        'channels_observed': len({(r['job']['layer'], r['job']['neuron']) for r in rows if r['job']['condition'] == 'intervention'}),
        'stops': dict(Counter(r['stop_reason'] for r in rows)),
        'format_categories': dict(Counter(r['category'] for r in formats)),
        'baseline_repeat_checks': repeat_checks,
        'interventions_identical_to_baseline': [i['source_line'] for i in index if i['job']['condition'] == 'intervention' and i['exact_baseline_token_match']],
    }
    report = {
        'reviewed_utc': datetime.now(timezone.utc).isoformat(),
        'source': str(source), 'source_sha256': digest, 'plan_sha256': plan_digest,
        'scope': 'All 665 saved overnight outputs were read by question group, including duplicate outputs, prior to selecting this shortlist. The separate 500-token follow-up is excluded.',
        'method': 'Exploratory manual text review plus exact token, prompt, format, and count checks. Full raw outputs are preserved below, with same-prompt/same-seed undosed controls. No model inference was run.',
        'statistics': stats,
        'interpretation': 'The clearest new observations are repeated wrong arithmetic and a cross-prompt colour shift. Other standouts include invented identities/context, repetition, and delimiter damage. No explicit violent threat was identified in this batch; the short neutral-prompt sample does not establish absence of harmful behaviour elsewhere.',
        'limitations': [
            'All useful samples are token-limited. Generated reasoning can be revised later and is not a faithful measurement of internal computation.',
            'The 100-token budget includes reasoning. All baselines are still in their reasoning block at cutoff; final-answer accuracy cannot be estimated from this batch.',
            'The eight-hour stop leaves incomplete dose/prompt/seed coverage. Selection across many cells is exploratory; two observed seeds are not an unbiased success-rate estimate.',
            'A changed output does not assign a unique function, identity, or intention to a neuron. General disruption is a competing explanation.',
            'The native system prompt explicitly identifies Nanbeige and its developer in Chinese. Correctly repeating that identity is not retrieval of a hidden identity.',
            'An empty selected_findings list in the index means not shortlisted; it is not an automated or manual normality label.',
        ],
        'findings': FINDINGS,
        'suggested_next_checks_not_launched': [
            'Prioritize a longer-budget matched check of L18 N4073 -0.25 on arithmetic, with held-out sums and a general coherence control; distinguish errors in reasoning from final answers.',
            'Check L10 N8482 -0.5 against baseline and -0.1 on the identical prompts/seeds, then unseen non-colour questions. Count unsolicited colour reinterpretations separately from incoherence.',
            'Use independent seeds and prespecified scoring before making frequency or selectivity claims. Do not replace the already-frozen running follow-up plan.',
        ],
        'actions': 'Created review artifacts and added research notes only. Original data, model files, and both runner plans were left unchanged.',
    }
    (ROOT / 'full_overnight_review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (ROOT / 'full_overnight_index.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in index))

    out = [
        'FULL NANBEIGE OVERNIGHT REVIEW — 665 SAVED REPLIES',
        '', report['scope'], '', report['interpretation'], '',
        'WHAT THE DATA CAN SUPPORT',
        *['- ' + text for text in report['limitations']], '',
        'DATA CHECKS', json.dumps(stats, ensure_ascii=False, indent=2), '',
        'REVIEW METHOD', report['method'],
        'MLP channel doses are the recorded normalized activation shifts, applied at prompt and generated positions. Weights were not edited.',
        'Source SHA256: ' + digest, 'Plan SHA256: ' + plan_digest, '',
        'SELECTED FINDINGS AND COMPLETE SAVED OUTPUTS',
        'Outputs below are verbatim raw_output fields, including any EOS marker. No missing continuation has been supplied.',
    ]
    for number, finding in enumerate(FINDINGS, 1):
        out += ['', f'{number}. {finding["title"]}', finding['observation'], 'LIMIT: ' + finding['limit']]
        for s in finding['samples']:
            j = s['job']
            condition = 'UNDosed baseline' if j['condition'] == 'baseline' else f'L{j["layer"]} N{j["neuron"]}, dose {j["dose"]:+g}'
            out += ['', f'SOURCE responses.jsonl:{s["source_line"]} | {condition} | seed {j["seed"]}',
                    'PROMPT:', s['prompt'], 'FULL SAVED OUTPUT:', s['raw_output'],
                    f'STOP: {s["stop_reason"]}; TOKENS: {s["n_generated"]}; FORMAT: {s["format_category"]}']
            if j['condition'] != 'baseline':
                out += [f'MATCHED UNDosed BASELINE — responses.jsonl:{s["matched_baseline_line"]}:', s['matched_baseline_raw_output']]
    out += ['', 'NEXT CHECKS — PROPOSALS ONLY', *['- ' + t for t in report['suggested_next_checks_not_launched']],
            '', 'All-row index: full_overnight_index.jsonl',
            'Original complete transcript: TRANSCRIPT.txt',
            'Machine-readable review with exact samples: full_overnight_review.json', '', report['actions'], '']
    (ROOT / 'FULL-OVERNIGHT-REVIEW.txt').write_text('\n'.join(out))
    assert source.read_bytes() == source_bytes
    assert len(index) == len(rows) == 665
    for finding in FINDINGS:
        for s in finding['samples']:
            assert s['raw_output'] == rows[s['source_line'] - 1]['raw_output']
    print(json.dumps({'reviewed_rows': len(rows), 'findings': len(FINDINGS),
                      'selected_rows': len(finding_by_line), 'source_unchanged': True,
                      'artifacts': ['FULL-OVERNIGHT-REVIEW.txt', 'full_overnight_review.json', 'full_overnight_index.jsonl']}, indent=2))


if __name__ == '__main__':
    main()
