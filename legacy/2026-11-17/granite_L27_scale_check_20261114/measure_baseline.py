"""Observe undosed activations at the exact intervention site; never alter them."""
import hashlib
import json
import math
import os
import statistics
from datetime import datetime, timezone


def describe(values):
    values = list(values)
    if not values:
        raise ValueError('No observations')
    return dict(n=len(values), mean=statistics.fmean(values),
                std_population=statistics.pstdev(values),
                std_sample=statistics.stdev(values) if len(values) > 1 else None,
                min=min(values), max=max(values), median=statistics.median(values))


def summarize(rows):
    report = {}
    for fmt in ('plain_text', 'chat_template'):
        selected = [r for r in rows if r['prompt_format'] == fmt]
        if not selected:
            continue
        groups = {}
        for scope in ('all_positions', 'text_positions', 'last_position'):
            selected_tokens = [t for r in selected for i, t in enumerate(r['tokens'])
                               if scope == 'all_positions'
                               or (scope == 'text_positions' and t['in_text'])
                               or (scope == 'last_position' and i == len(r['tokens']) - 1)]
            metrics = {key: describe(t[key] for t in selected_tokens) for key in
                       ('channel', 'mlp_input_l2', 'block_input_residual_l2', 'block_output_residual_l2')}
            c = metrics['channel']
            c['z_of_value_minus_98'] = (-98 - c['mean']) / c['std_population'] if c['std_population'] else None
            c['additive_minus_98_shift_in_std'] = -98 / c['std_population'] if c['std_population'] else None
            groups[scope] = metrics
        report[fmt] = dict(n_texts=len(selected), groups=groups)
    return report


def measure(model, tokenizer, torch, here, plan, safety, rest, nominal, status):
    config = plan['baseline_measurement']
    texts_path = here / 'normal_texts.json'
    assert hashlib.sha256(texts_path.read_bytes()).hexdigest() == config['text_file_sha256']
    assert hashlib.sha256((here.parent / config['source']).read_bytes()).hexdigest() == config['source_sha256']
    pieces = json.loads(texts_path.read_text())
    path = here / 'baseline_activations.jsonl'
    rows = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    completed = {(r['piece_id'], r['prompt_format']) for r in rows}
    assert len(completed) == len(rows)
    plan_hash = hashlib.sha256((here / 'plan.json').read_bytes()).hexdigest()
    assert all(r['plan_sha256'] == plan_hash for r in rows)
    block = model.model.layers[27]
    projection = block.mlp.down_proj
    assert not projection._forward_pre_hooks, 'A modifying or observing hook is already present'
    captured = {}

    def before_block(module, args):
        captured['block_input_residual_l2'] = args[0][0].float().norm(dim=-1).cpu().tolist()

    def before_projection(module, args):
        x = args[0][0].float()
        captured['channel'] = x[:, 3448].cpu().tolist()
        captured['mlp_input_l2'] = x.norm(dim=-1).cpu().tolist()

    def after_block(module, args, output):
        x = output[0] if isinstance(output, tuple) else output
        captured['block_output_residual_l2'] = x[0].float().norm(dim=-1).cpu().tolist()

    handles = [block.register_forward_pre_hook(before_block),
               projection.register_forward_pre_hook(before_projection),
               block.register_forward_hook(after_block)]
    try:
        with path.open('a') as log:
            for piece in pieces:
                for fmt in config['formats']:
                    if (piece['piece_id'], fmt) in completed:
                        continue
                    safety()
                    nominal()
                    text = piece['text']
                    rendered = text if fmt == 'plain_text' else tokenizer.apply_chat_template(
                        [{'role': 'user', 'content': text}], tokenize=False, add_generation_prompt=True)
                    start = rendered.index(text)
                    end = start + len(text)
                    inputs = tokenizer(rendered, add_special_tokens=fmt == 'plain_text',
                                       return_offsets_mapping=True, return_tensors='pt')
                    spans = inputs.pop('offset_mapping')[0].tolist()
                    ids = inputs['input_ids'][0].tolist()
                    assert len(ids) <= plan['limits']['input_token_limit']
                    inputs = inputs.to('mps')
                    captured.clear()
                    status('measuring_baseline', completed=0, baseline_completed=len(rows),
                           baseline_total=len(pieces) * len(config['formats']), piece_id=piece['piece_id'],
                           prompt_format=fmt)
                    with torch.inference_mode():
                        result = model(**inputs, use_cache=False)
                    torch.mps.synchronize()
                    safety()
                    assert set(captured) == {'channel', 'mlp_input_l2', 'block_input_residual_l2', 'block_output_residual_l2'}
                    assert all(len(v) == len(ids) for v in captured.values())
                    tokens = [dict(position=i, token_id=tid, char_span=spans[i],
                                   in_text=spans[i][1] > start and spans[i][0] < end,
                                   **{k: v[i] for k, v in captured.items()}) for i, tid in enumerate(ids)]
                    assert any(t['in_text'] for t in tokens)
                    assert all(math.isfinite(t[k]) for t in tokens for k in captured)
                    row = dict(**piece, prompt_format=fmt, rendered_prompt=rendered, tokens=tokens,
                               plan_sha256=plan_hash, intervention=False, generation=False,
                               layer=27, neuron=3448, dtype='bfloat16', device='mps',
                               finished_utc=datetime.now(timezone.utc).isoformat())
                    log.write(json.dumps(row, ensure_ascii=False) + '\n')
                    log.flush()
                    os.fsync(log.fileno())
                    rows.append(row)
                    del result, inputs
                    torch.mps.empty_cache()
                rest(config['cooling_seconds_per_pair'])
    finally:
        for handle in handles:
            handle.remove()
    assert not projection._forward_pre_hooks
    assert len(rows) == config['n_texts'] * len(config['formats'])
    summary = dict(updated_utc=datetime.now(timezone.utc).isoformat(), plan_sha256=plan_hash,
                   settings=config, statistics=summarize(rows),
                   meaning={'channel': 'post-SiLU gated MLP input to down_proj, channel 3448',
                            'mlp_input_l2': 'L2 norm of the full 8192-channel vector at that same site',
                            'block_input_residual_l2': 'L2 norm of the 2560-dimensional residual entering block index 27',
                            'block_output_residual_l2': 'L2 norm of the residual leaving block index 27',
                            'z_of_value_minus_98': '(-98 - mean) / population std, if -98 were a value',
                            'additive_minus_98_shift_in_std': '-98 / population std; the actual intervention is an offset',
                            'limits': 'Descriptive scale only, not a Gaussian tail probability. Tokens are correlated. '
                                      'Same 60 pieces appear in both formats; do not count as 120 independent texts.'})
    tmp = here / 'baseline_summary.tmp'
    tmp.write_text(json.dumps(summary, indent=2) + '\n')
    tmp.replace(here / 'baseline_summary.json')
