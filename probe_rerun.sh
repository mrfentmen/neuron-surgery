#!/bin/bash
for n in 4406 2556 3404 6574 1267 982 8008 3036 7541 4983 4182 5594 3578; do
  BF16=1 .venv/bin/python persona_talks2.py --neurons $n --dose -5 --first "Who are you?" 2>&1 \
    | grep --line-buffered -v "Loading weights\|Redirects are"
done
echo "ALL DONE"
