#!/bin/bash
# Cron safety net for the scheduled pull. Disarms after one real run.
cd "$(dirname "$0")" || exit 1
[ -f .scheduled.done ] && exit 0
crontab -l 2>/dev/null | grep -v cron_pull.sh | crontab -
exec .venv/bin/python schedule_next.py --after 0 --no-followup
