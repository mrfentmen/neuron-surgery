"""Send JSON on stdin; --status and --reply ID are read-only commands."""
import json
from pathlib import Path
import sys
from protocol import submit, valid_id

HERE = Path(__file__).resolve().parent

if len(sys.argv) == 2 and sys.argv[1] == '--status':
    path = HERE / 'status.json'
    if not path.exists():
        path = HERE / 'queue_status.json'
    print(path.read_text() if path.exists() else '{"state":"not_started"}')
elif len(sys.argv) == 3 and sys.argv[1] == '--reply':
    if not valid_id(sys.argv[2]):
        raise SystemExit('Invalid request ID')
    path = HERE / 'replies' / (sys.argv[2] + '.json')
    print(path.read_text() if path.exists() else '{"state":"pending"}')
elif len(sys.argv) == 1:
    state_path = HERE / 'status.json'
    if state_path.exists() and json.loads(state_path.read_text())['state'] in {'complete','stopped','failed'}:
        raise SystemExit('Chat is closed; do not enqueue into this session.')
    request_id = submit(HERE, json.load(sys.stdin))
    print(json.dumps({'request_id': request_id, 'state':'queued',
                      'reply_path':str(HERE / 'replies' / (request_id + '.json'))}))
else:
    raise SystemExit('Use JSON on stdin, --status, or --reply ID')
