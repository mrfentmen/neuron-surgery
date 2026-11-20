"""Validated pacing overrides; experiment and hardware thresholds stay frozen."""
from datetime import datetime
import json
import math


BOUNDS = {
    'startup_cooling_seconds': (5, 60),
    'minimum_cooling_seconds': (5, 60),
    'cooling_generation_multiplier': (0.1, 3),
    'rest_every_tokens': (32, 128),
    'intrageneration_rest_seconds': (1, 10),
}


def apply_pacing(original, config):
    if set(config) != {'profile', 'authorized_by', 'limits', 'periodic_cooling_seconds', 'original_deadline_utc'}:
        raise ValueError('Unexpected pacing settings')
    changes = config['limits']
    if set(changes) != set(BOUNDS):
        raise ValueError('Only the five pacing fields may be changed')
    for key, value in changes.items():
        low, high = BOUNDS[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError('Invalid pacing value: ' + key)
    if not isinstance(changes['rest_every_tokens'], int):
        raise ValueError('Rest interval must be an integer')
    if not 30 <= config['periodic_cooling_seconds'] <= 180:
        raise ValueError('Invalid periodic cooling')
    deadline = datetime.fromisoformat(config['original_deadline_utc'])
    if deadline.tzinfo is None:
        raise ValueError('Deadline must include timezone')
    return dict(original, **changes)


def load_pacing(folder, original):
    path = folder / 'pacing_override.json'
    if not path.exists():
        return dict(original), {'profile': 'original', 'periodic_cooling_seconds': 180}
    config = json.loads(path.read_text())
    return apply_pacing(original, config), config
