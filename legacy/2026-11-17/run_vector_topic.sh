#!/bin/bash
cd "$(dirname "$0")" || exit 1
echo "start $(date '+%T') vector_topic"
exec .venv/bin/python vector_topic.py
