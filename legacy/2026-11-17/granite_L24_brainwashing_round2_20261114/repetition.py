"""Conservative exact token-loop detection, not a factuality or intent judge."""
import math


def repeated_tail(tokens, *, minimum_tokens=64, minimum_repeats=4, maximum_period=128):
    """Return evidence if the tail is the same token block repeated consecutively."""
    if len(tokens) < minimum_tokens:
        return None
    for period in range(1, min(maximum_period, len(tokens) // minimum_repeats) + 1):
        repeats = max(minimum_repeats, math.ceil(minimum_tokens / period))
        width = period * repeats
        if width > len(tokens):
            continue
        block = tokens[-period:]
        if tokens[-width:] == block * repeats:
            return {'detector': 'exact_token_loop', 'period_tokens': period,
                    'consecutive_repeats': repeats, 'repeated_tokens': width,
                    'detected_at_tokens': len(tokens)}
    return None
