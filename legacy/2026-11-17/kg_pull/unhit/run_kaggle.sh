#!/usr/bin/env bash
# Unused in this kernel: the run cell calls run_grid.sh directly so it can pass
# TILES, DOSES=5, SIGNS=neg and MAX_NEW=30 in one place.
set -u
exec ./run_grid.sh
