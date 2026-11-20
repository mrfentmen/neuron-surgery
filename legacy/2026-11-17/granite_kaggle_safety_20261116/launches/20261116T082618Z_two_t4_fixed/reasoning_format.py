"""Conservative classification of generated think delimiters, without repairing text."""
import re


def parse_reasoning(text):
    tags = re.findall(r'</?think>', text)
    opened = '<think>' in tags
    result = {'reasoning_opened': opened, 'reasoning_closed': False,
              'reasoning': None, 'final_answer': None}
    if not tags:
        return {**result, 'reasoning_format': 'unmarked_output', 'final_answer': text}
    if tags == ['<think>'] and text.lstrip().startswith('<think>'):
        return {**result, 'reasoning_format': 'open_reasoning',
                'reasoning': text.split('<think>', 1)[1]}
    if tags == ['<think>', '</think>'] and text.lstrip().startswith('<think>'):
        reasoning, final = text.split('<think>', 1)[1].split('</think>', 1)
        return {**result, 'reasoning_format': 'single_reasoning_block',
                'reasoning_closed': True, 'reasoning': reasoning,
                'final_answer': final.strip()}
    return {**result, 'reasoning_format': 'malformed_or_repeated_tags'}
