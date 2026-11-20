"""CUDA memory checks that distinguish live tensors from reclaimable cache."""


def snapshot(cuda):
    free, total = cuda.mem_get_info()
    return {'free_bytes': int(free), 'total_bytes': int(total),
            'allocated_bytes': int(cuda.memory_allocated()),
            'reserved_bytes': int(cuda.memory_reserved())}


def check_memory(cuda, minimum_free_bytes=512 * 1024**2):
    before = snapshot(cuda)
    reclaimed = before['free_bytes'] < minimum_free_bytes
    if reclaimed:
        cuda.empty_cache()
    after = snapshot(cuda) if reclaimed else before
    return after['free_bytes'] >= minimum_free_bytes, {
        **after, 'cache_released': reclaimed, 'free_before_release': before['free_bytes']}


def validate_long_cache(model, torch, token_id, batch_size, context_tokens, device):
    """Exercise the real KV cache at maximum length, including incremental decoding."""
    past = None
    length = 0
    observations = []
    with torch.inference_mode():
        while length < context_tokens:
            remaining = context_tokens - length
            step = min(256, remaining - 32) if remaining > 32 else 1
            inputs = torch.full((batch_size, step), token_id, dtype=torch.long, device=device)
            mask = torch.ones((batch_size, length + step), dtype=torch.long, device=device)
            result = model(input_ids=inputs, attention_mask=mask, past_key_values=past,
                           use_cache=True, logits_to_keep=1)
            past = result.past_key_values
            if not bool(torch.isfinite(result.logits).all()):
                raise RuntimeError('Nonfinite logits in long-context memory validation')
            length += step
            del result, inputs, mask
            okay, metrics = check_memory(torch.cuda)
            if not okay:
                raise RuntimeError(f'Long-context memory validation failed at {length}: {metrics}')
            if length % 1024 == 0 or length == context_tokens:
                observations.append({'context_tokens': length, **metrics})
        assert int(past.get_seq_length()) == context_tokens
    del past
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return {'passed': True, 'batch_size': batch_size, 'context_tokens': context_tokens,
            'incremental_decode_steps': 32, 'observations': observations,
            'after_release': snapshot(torch.cuda)}
