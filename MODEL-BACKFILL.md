# Rows with no `model` field: what they are and why the skip-list cares

**423,207 rows in 640 files carry no `model` field.** They are the oldest
results in the tree, written before the field existed.

## The rule now in force

`make_kaggle_screen.py` counts a tile as *already covered* only if:

- the row's `model` field is **empty**, or
- it **contains** `granite-4.1-3b`.

Anything else — `ibm-granite/granite-4.0-350m`, `Qwen/Qwen2.5-0.5B-Instruct`,
`/Users/test/Desktop/neuron-surgery/models/granite-4.1-3b-base` is accepted
because it is the same checkpoint under a local path.

## Why

The tree contains **4,146 tiles steered in granite-4.0-350m** and **50 in
Qwen2.5-0.5B-Instruct**. A layer number in a 350M model does not name the same
computation as the same layer number in a 3B model. Counting them as covered
made the skip-list claim the 3B had seen tiles it had never seen.

The old flat `glob("*.jsonl")` had the opposite bug — it missed every nested
directory, so it under-counted coverage and sent the census over 3,293 tiles that
were already owned. Both bugs are now gone: the scan is recursive **and**
model-filtered.

## What "empty means Granite 4.1 3B" rests on

The untagged files are the early `grid_*`, `behaviour*`, `control_*`, `levels*`
and sweep outputs. Every script that wrote them imports
`neuron_surgery.main()`, whose default model is
`ibm-granite/granite-4.1-3b`. No script in the tree defaulted to another
checkpoint at that time. This is an inference from the code, not a recorded
field, and it is why the tag check accepts empty rather than rejecting it.

## The consequence, measured

**Before the L0-L19 census**, with only the earlier runs in the tree:

| | tiles |
|---|---|
| covered if every model counts | 19,152 |
| covered after the model filter | 16,265 |
| difference | **2,887 tiles that had never been steered in the 3B** |

**After the census**, the difference is **zero** — all 168,028 covered tiles are
now Granite 4.1 3B tiles:

| | tiles |
|---|---|
| covered if every model counts | 168,028 |
| covered after the model filter | 168,028 |
| difference | **0** |

The 2,887 closed themselves. The census was built with the *older* flat-glob
skip-list, which under-counted coverage, so it steered those tiles even though
they were "covered" by 350M rows. That is why the follow-up gap notebook only
finds **93 tiles** left in L0-L19 rather than 2,887.

The filter still matters for L20-L39: without it, any 350M row on the top half
would hide the tile from the 3B census that is running there now.

## If you backfill

Do not rewrite the 423,207 rows to insert `"model": "ibm-granite/granite-4.1-3b"`.
Rewriting data to assert something that was never recorded is worse than a
documented gap. If a future run needs per-row attribution, back it from the
script's default at the commit that produced it, and record that as provenance
rather than as measurement.
