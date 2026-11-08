#!/bin/bash
# Cron safety net for the scheduled pull. Disarms after one real run.
cd "$(dirname "$0")" || exit 1
[ -f .scheduled.done ] && exit 0
exec .venv/bin/python schedule_next.py --after 0 --no-followup
