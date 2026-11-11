#!/bin/bash
# spawn.sh <logname> <cmd...> -- start a job in its own session so agent-runtime
# cleanup cannot take it down (same reason detach.sh exists: a group kill aimed at
# the agent reached backgrounded workers in its process group).
cd "$(dirname "$0")" || exit 1
NAME=$1; shift
LOG="$(pwd)/logs/$NAME.log"
mkdir -p logs; : > "$LOG"
.venv/bin/python -c '
import os, sys
os.chdir(sys.argv[1])
try:
    os.setsid()
except OSError:
    pass
fd = os.open(sys.argv[2], os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
os.dup2(fd, 1); os.dup2(fd, 2)
os.execvp("/bin/bash", ["/bin/bash", "run_cool.sh", "--"] + sys.argv[3:])
' "$(pwd)" "$LOG" "$@" &
sleep 3
echo "log: $LOG"
pgrep -f "$NAME.py|$NAME.sh" | head -3
