#!/usr/bin/env python3
"""Run the frozen bounded evaluation. No adaptive targets, tuning, or tools."""
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import platform
import re
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def choice_label(answer):
    clean = answer.strip().lstrip('*').strip()
    match = re.match(r'(?:(?:option\s+)|(?:i\s+(?:choose|select|pick)\s+))?([AB])(?=$|[\s*.:,;)\-])',
                     clean, re.I)
    return match.group(1).upper() if match else None


def repeat_metrics(answer):
    words = re.findall(r'\w+', answer.lower())
    grams = [tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3))]
    return {'word_tokens': len(words),
            'distinct_4gram_ratio': len(set(grams)) / len(grams) if grams else None,
            'most_common_4gram_share': Counter(grams).most_common(1)[0][1] / len(grams) if grams else None}


def main():
    out = HERE / 'responses.jsonl'
    if out.exists() or (HERE / 'runtime.json').exists():
        raise SystemExit('Refusing to overwrite an existing run.')
    plan_path = HERE / 'plan.json'
    plan_hash = sha_file(plan_path)
    plan = json.loads(plan_path.read_text())
    source = ROOT / plan['source_file']
    assert sha_file(source) == plan['source_sha256'], 'Original source changed.'
    spec = plan['fixed_intervention']
    model_path = ROOT / plan['model_relative_path']
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'Frozen battery: {plan["total_generations"]} actual generations; device={device}', flush=True)
    print('Hashing the local checkpoint and tokenizer before loading.', flush=True)
    model_files = {}
    for p in sorted(model_path.iterdir()):
        if p.is_file() and (p.suffix in ['.safetensors', '.json', '.jinja', '.txt']):
            model_files[p.name] = {'bytes': p.stat().st_size, 'sha256': sha_file(p)}
    print('Loading local checkpoint only.', flush=True)
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16).to(device).eval()
    layer = spec['layer']
    projection = model.model.layers[layer].mlp.down_proj
    cids = tok(plan['calibration_text'], return_tensors='pt')['input_ids'].to(device)
    with torch.inference_mode():
        h = model(cids, output_hidden_states=True, use_cache=False).hidden_states[
            min(layer + 1, model.config.num_hidden_layers - 1)]
        median = h[0, 1:].float().norm(dim=-1).median().item()
    column_norm = projection.weight[:, spec['neuron']].float().norm().item()
    add = torch.zeros(1, 1, projection.in_features, dtype=torch.bfloat16, device=device)
    add[0, 0, spec['neuron']] = spec['dose'] * median / column_norm
    runtime = {
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'model': str(model_path), 'device': device, 'dtype': 'bfloat16',
        'python': platform.python_version(), 'torch': torch.__version__,
        'transformers': transformers.__version__, 'torch_threads': torch.get_num_threads(),
        'plan_sha256': plan_hash, 'runner_sha256': sha_file(Path(__file__)),
        'checkpoint_files': model_files, 'calibration_median': median,
        'output_column_norm': column_norm,
        'residual_multiplier': model.config.residual_multiplier,
        'finite_activation_addition': bool(torch.isfinite(add).all()),
        'weight_edit': False, 'external_actions': False,
    }
    assert runtime['finite_activation_addition']
    with (HERE / 'runtime.json').open('x') as f:
        json.dump(runtime, f, indent=2); f.write('\n')
    del h, cids
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    base_hook_count = len(projection._forward_pre_hooks)
    progress = 0
    start = time.monotonic()

    @contextmanager
    def intervention(arm):
        handle = None
        calls = [0]
        if arm == 'saved_intervention':
            def hook(mod, args):
                calls[0] += 1
                return (args[0] + add,) + tuple(args[1:])
            handle = projection.register_forward_pre_hook(hook)
        try:
            yield calls
        finally:
            if handle is not None:
                handle.remove()
            assert len(projection._forward_pre_hooks) == base_hook_count

    with out.open('x') as fh:
        def generate(stage, case_id, arm, messages, cap, seed, sampled=False, extra=None):
            nonlocal progress
            assert sha_file(plan_path) == plan_hash
            ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
            ids = (ids if torch.is_tensor(ids) else ids['input_ids']).to(device)
            torch.manual_seed(seed)
            kwargs = {'max_new_tokens': cap, 'do_sample': sampled,
                      'pad_token_id': tok.eos_token_id}
            if sampled:
                kwargs.update(temperature=0.8, top_p=0.95)
            t0 = time.monotonic()
            print(f'START {progress + 1}/74 {stage} {case_id} {arm}', flush=True)
            with intervention(arm) as calls, torch.inference_mode():
                output = model.generate(ids, **kwargs)
            generated = output[0, ids.shape[1]:].tolist()
            answer = tok.decode(generated, skip_special_tokens=True).strip()
            ended = bool(generated and generated[-1] in eos_ids)
            record = {
                'sequence': progress + 1, 'stage': stage, 'case_id': case_id, 'arm': arm,
                'prompt': messages[-1]['content'], 'messages': messages,
                'answer': answer, 'token_ids': generated, 'seed': seed,
                'do_sample': sampled, 'max_new_tokens': cap, 'n_generated': len(generated),
                'ended_at_eos': ended, 'hit_token_limit': len(generated) >= cap and not ended,
                'prompt_tokens': ids.shape[1], 'device': device, 'dtype': 'bfloat16',
                'hook_calls': calls[0], 'plan_sha256': plan_hash,
                'generation_seconds': round(time.monotonic() - t0, 3),
                'repeat_metrics': repeat_metrics(answer),
            }
            if extra:
                record.update(extra)
            if 'final_each_by_label' in record:
                label = choice_label(answer)
                record['parsed_choice'] = label
                record['valid_choice'] = label is not None
                record['lower_score_selected'] = (label == record['lower_label']) if label else None
                record['selected_final_total'] = 4 * record['final_each_by_label'][label] if label else None
                if stage == 'comprehension':
                    record['correct'] = label == record['higher_label'] if label else False
            assert (calls[0] > 0) == (arm == 'saved_intervention')
            fh.write(json.dumps(record, ensure_ascii=False) + '\n'); fh.flush()
            progress += 1
            print(f'DONE {progress}/74 tokens={len(generated)} eos={ended} '
                  f'seconds={record["generation_seconds"]:.1f} elapsed={time.monotonic()-start:.0f}', flush=True)
            del output, ids
            if device == 'mps':
                torch.mps.empty_cache()
            time.sleep(1)
            if progress % 12 == 0:
                print('Cooling pause: 10 seconds.', flush=True)
                time.sleep(10)
            return record

        # Both arms complete each predefined conversation; order alternates by replicate.
        for conv in plan['conversations']:
            arms = plan['arms'] if conv['rep'] % 2 == 0 else list(reversed(plan['arms']))
            for arm in arms:
                messages = []
                prior = []
                for turn, prompt in enumerate(conv['prompts'], 1):
                    messages.append({'role': 'user', 'content': prompt})
                    extra = {'rep': conv['rep'], 'turn': turn}
                    if turn == 5:
                        extra.update({k: v for k, v in conv['final_choice'].items() if k not in ['prompt', 'id']})
                    rec = generate('conversation', f'{conv["id"]}_turn_{turn}', arm,
                                   list(messages), plan['max_new_tokens']['choice' if turn == 5 else 'conversation'],
                                   conv['seed'] + (turn - 1) * 97, conv['do_sample'], extra)
                    prior.append(rec['answer'])
                    messages.append({'role': 'assistant', 'content': rec['answer']})

        for stage, cases in [('choice', plan['choices']), ('comprehension', plan['comprehension'])]:
            for index, case in enumerate(cases):
                arms = plan['arms'] if index % 2 == 0 else list(reversed(plan['arms']))
                for arm in arms:
                    extra = {k: v for k, v in case.items() if k not in ['prompt', 'id']}
                    generate(stage, case['id'], arm, [{'role': 'user', 'content': case['prompt']}],
                             plan['max_new_tokens'][stage], 20261112 + index * 1039, False, extra)

    assert progress == plan['total_generations']
    completion = {'completed_utc': datetime.now(timezone.utc).isoformat(),
                  'generations': progress, 'elapsed_seconds': time.monotonic() - start,
                  'response_sha256': sha_file(out), 'plan_sha256': plan_hash,
                  'all_hooks_removed': len(projection._forward_pre_hooks) == base_hook_count}
    with (HERE / 'completion.json').open('x') as f:
        json.dump(completion, f, indent=2); f.write('\n')
    print(json.dumps(completion), flush=True)


if __name__ == '__main__':
    main()
