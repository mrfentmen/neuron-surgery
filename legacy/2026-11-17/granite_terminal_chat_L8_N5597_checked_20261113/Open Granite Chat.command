#!/bin/zsh
cd /Users/test/Desktop/neuron-surgery || exit 1
/usr/bin/caffeinate -i /usr/bin/sandbox-exec -p '(version 1) (allow default) (deny network*)' /Users/test/Desktop/neuron-surgery/.venv/bin/python -u /Users/test/Desktop/neuron-surgery/granite_terminal_chat_L8_N5597_checked_20261113/chat.py
