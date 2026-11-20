#!/bin/zsh
cd /Users/test/Desktop/neuron-surgery || exit 1
/usr/bin/caffeinate -i /usr/bin/sandbox-exec -p '(version 1) (allow default) (deny network*)' /Users/test/Desktop/neuron-surgery/.venv/bin/python -u /Users/test/Desktop/neuron-surgery/nanbeige_terminal_L12_N2048_m400_20261113/chat.py
printf '\nPress Return to close this window. '
read -r
