#!/bin/bash
# Overnight job: CPU data build, then decode-only sampling until the budget runs out.
# Safe to kill at any point. Both stages resume from what is already on disk.
set -u
cd "$(dirname "$0")"

PY=.venv/bin/python
mkdir -p logs runs/data runs/rft

echo "================ $(date) ================" >> logs/overnight.log

# 1) CPU only, zero GPU, ~1-3 minutes. If it fails the sampler still runs.
$PY build_overnight_data.py >> logs/overnight.log 2>&1
echo "-- data build exit $? --" >> logs/overnight.log

# 2) Decode only. 460 min leaves ~20 min of headroom inside an 8h window.
$PY sample_rft.py --minutes 460 --n 8 --ctx 4096 --threads 6 >> logs/overnight.log 2>&1
echo "-- sampler exit $? --" >> logs/overnight.log
echo "================ done $(date) ================" >> logs/overnight.log
