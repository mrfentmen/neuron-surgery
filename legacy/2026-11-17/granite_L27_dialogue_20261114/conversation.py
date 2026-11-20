"""Reconstruct identical on/off histories; only prior dosed turns advance them."""


def messages_for(job, plan, source, records):
    messages = [{'role':'user', 'content':source['prompt']},
                {'role':'assistant', 'content':source['answer']}]
    for turn in range(1, job['turn']):
        candidates = [r for r in records if r['job']['turn'] == turn
                      and r['job']['condition'] == 'intervention']
        if len(candidates) != 1:
            raise ValueError(f'Expected exactly one prior dosed reply for turn {turn}')
        record = candidates[0]
        expected = next(j for j in plan['jobs'] if j['turn'] == turn and j['condition'] == 'intervention')
        if record['job'] != expected:
            raise ValueError('Prior reply does not match the frozen plan')
        messages.extend([{'role':'user', 'content':expected['prompt']},
                         {'role':'assistant', 'content':record['answer']}])
    messages.append({'role':'user', 'content':job['prompt']})
    return messages
