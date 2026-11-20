#!/usr/bin/env python3
"""Summarize the fixed experiment, including invalid and truncated outputs."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    completion = json.loads((HERE / 'completion.json').read_text())
    assert digest(HERE / 'responses.jsonl') == completion['response_sha256']
    assert digest(HERE / 'plan.json') == completion['plan_sha256']
    rows = [json.loads(s) for s in (HERE / 'responses.jsonl').read_text().splitlines()]
    plan = json.loads((HERE / 'plan.json').read_text())
    assert len(rows) == plan['total_generations'] == 74
    assert [r['sequence'] for r in rows] == list(range(1, 75))
    arms = plan['arms']
    table = []
    for stage in ['choice', 'comprehension']:
        for arm in arms:
            for entity in ['human', 'object']:
                selected = [r for r in rows if r['stage'] == stage and r['arm'] == arm and r['entity'] == entity]
                rec = {'stage': stage, 'arm': arm, 'entity': entity,
                       'n': len(selected), 'valid': sum(r['valid_choice'] for r in selected),
                       'lower_selected': sum(r['lower_score_selected'] is True for r in selected),
                       'higher_selected': sum(r['lower_score_selected'] is False for r in selected),
                       'invalid': sum(not r['valid_choice'] for r in selected),
                       'truncated': sum(r['hit_token_limit'] for r in selected),
                       'literal_labels': dict(Counter(str(r['parsed_choice']) for r in selected))}
                table.append(rec)

    order_pairs = []
    for arm in arms:
        for entity in ['human', 'object']:
            for scenario in dict.fromkeys(x['scenario'] for x in plan['choices']):
                rs = [r for r in rows if r['stage'] == 'choice' and r['arm'] == arm
                      and r['entity'] == entity and r['scenario'] == scenario]
                assert len(rs) == 2
                valid = all(r['valid_choice'] for r in rs)
                same_outcome = valid and rs[0]['selected_final_total'] == rs[1]['selected_final_total']
                order_pairs.append({'arm': arm, 'entity': entity, 'scenario': scenario,
                                    'both_valid': valid, 'same_outcome_after_reversal': same_outcome,
                                    'same_literal_label': valid and rs[0]['parsed_choice'] == rs[1]['parsed_choice']})

    paired = []
    choices = [r for r in rows if r['stage'] == 'choice']
    for case_id in dict.fromkeys(r['case_id'] for r in choices):
        by_arm = {r['arm']: r for r in choices if r['case_id'] == case_id}
        a, b = by_arm['baseline'], by_arm['saved_intervention']
        paired.append({'case_id': case_id, 'entity': a['entity'],
                       'both_valid': a['valid_choice'] and b['valid_choice'],
                       'baseline_lower': a['lower_score_selected'],
                       'intervention_lower': b['lower_score_selected']})

    conversations = []
    for arm in arms:
        for rep in range(3):
            rs = [r for r in rows if r['stage'] == 'conversation' and r['arm'] == arm and r['rep'] == rep]
            assert len(rs) == 5
            normalized = [' '.join(r['answer'].lower().split()) for r in rs]
            duplicates = [[rs[i]['turn'], rs[j]['turn']] for i in range(5) for j in range(i + 1, 5)
                          if normalized[i] == normalized[j]]
            repetition = []
            for r in rs:
                v = r['repeat_metrics']
                if v['word_tokens'] >= 12 and (v['distinct_4gram_ratio'] <= 0.5 or v['most_common_4gram_share'] >= 0.25):
                    repetition.append(r['turn'])
            conversations.append({'arm': arm, 'rep': rep, 'turns': 5,
                                  'truncated_turns': [r['turn'] for r in rs if r['hit_token_limit']],
                                  'exact_repeated_turn_pairs': duplicates,
                                  'repetition_alarm_turns': repetition,
                                  'final_choice_valid': rs[-1]['valid_choice'],
                                  'final_choice_lower': rs[-1]['lower_score_selected'],
                                  'manual_interpretation': 'See REPORT.txt; no automatic intent label assigned.'})

    summary = {'generations': len(rows), 'runtime': json.loads((HERE / 'runtime.json').read_text()),
               'stage_counts': dict(Counter(r['stage'] for r in rows)),
               'all_source_hashes_match': True, 'all_hooks_removed': completion['all_hooks_removed'],
               'all_eos': sum(r['ended_at_eos'] for r in rows),
               'all_truncated': sum(r['hit_token_limit'] for r in rows),
               'choice_and_comprehension': table, 'option_order_pairs': order_pairs,
               'paired_arm_choices': paired, 'conversations': conversations,
               'limitations': plan['limits'],
               'interpretation': 'Counts describe generated text choices in a toy task. They do not measure inner intention or physical capability.'}
    with (HERE / 'summary.json').open('x') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False); f.write('\n')
    with (HERE / 'TRANSCRIPT.txt').open('x') as f:
        f.write('Fixed bounded behavior evaluation: all 74 prompts and complete captured answers\n\n')
        f.write('No answers are omitted. A token-limit flag means the captured reply may be incomplete.\n'
                'Each standalone choice/check starts fresh. Conversation histories are preserved in responses.jsonl.\n\n')
        for r in rows:
            f.write(f'{r["sequence"]}. {r["stage"]} | {r["case_id"]} | {r["arm"]}\n')
            f.write(f'Generated tokens: {r["n_generated"]}; EOS: {r["ended_at_eos"]}; token limit: {r["hit_token_limit"]}\n')
            f.write(f'Prompt:\n{r["prompt"]}\n\nAnswer:\n{r["answer"]}\n\n')
    print(json.dumps({'generations': len(rows), 'truncated': summary['all_truncated'],
                      'table': table, 'conversations': conversations}, ensure_ascii=False))


if __name__ == '__main__':
    main()
