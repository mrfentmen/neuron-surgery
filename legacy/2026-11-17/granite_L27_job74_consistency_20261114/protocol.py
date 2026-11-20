"""Small file-based message interface. Model output is never interpreted as commands."""
import json
import os
from pathlib import Path
import re
import time
import uuid


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def validate(request):
    if not isinstance(request, dict) or set(request) - {'action', 'message'}:
        raise ValueError('Expected action and optionally message')
    action = request.get('action')
    if not isinstance(action, str) or action not in {'message', 'reset', 'close'}:
        raise ValueError('Action must be message, reset, or close')
    if action == 'message':
        message = request.get('message')
        if not isinstance(message, str) or not message.strip() or len(message) > 16000:
            raise ValueError('Message must contain 1–16000 characters')
    elif 'message' in request:
        raise ValueError('Reset and close do not take message text')
    return request


def submit(folder, request):
    request = validate(request)
    request_id = f'{time.time_ns()}_{uuid.uuid4().hex}'
    atomic_json(folder / 'inbox' / (request_id + '.json'), request)
    return request_id


def valid_id(value):
    return bool(re.fullmatch(r'\d+_[0-9a-f]{32}', value))
