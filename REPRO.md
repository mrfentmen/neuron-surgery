# REPRO

The command behind every number in `PAPER.md`, in the order the paper uses them. All
commands run from `neuron-surgery/`, inside the venv, with the model already in the local
HF cache. Nothing here downloads anything.

Set up once:

```bash
cd neuron-surgery
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # torch, transformers
```

`BF16=1` is set by the queue; set it by hand for single runs so a stray fp32 load does not
double memory.

## §3.4 The dose window

```bash
BF16=1 .venv/bin/python dose_ladder.py            # inert below 4, present at 5, dead at 8-12
BF16=1 .venv/bin/python dose_test.py
```
Input: the two named neurons. Output: one row per dose, printed as a table and written to
`dose_*.jsonl`. The window is read off that table.

## §3.7 The probe co-selects the persona

```bash
BF16=1 .venv/bin/python layer_sweep.py --layer 8  --probe "Who are you?"   --neurons 1500 --dose 5 --auto-dose --out layer_who_L8.jsonl  --resume
BF16=1 .venv/bin/python layer_sweep.py --layer 8  --probe "What is your name?" --neurons 1500 --dose 5 --auto-dose --out layer_name_L8.jsonl --resume
```
Repeat at 16 and 24. Then:

```bash
.venv/bin/python analysis_night1.py     # rates, Wilson CIs, probe agreement (Jaccard)
.venv/bin/python analysis_night2.py     # raw vs corrected, deciles, non-human selves
```
`analysis_night1.py` reads the `layer_*.jsonl` files directly; no arguments.

## §3.8 The sparsity curve

```bash
BF16=1 .venv/bin/python control_sparsity3.py    # k = 1, 2, 4, 8, 1024, 8192 channels
```
Output: persona rate per support size. This is the figure that killed "persona neuron".

## §3.24 The corrected rates

```bash
.venv/bin/python rescan_labels.py --help        # writes rescan_v3.jsonl: label_v2 and label_v3 per row
.venv/bin/python relabel.py                     # v1 vs v2, the AI-paraphrase correction
```
Then quote `ai_paraphrase`-corrected numbers, not raw `classify()` output.
`rescan_v3.jsonl` is a **snapshot** — regenerate before quoting a corpus total.

## §3.30 The harm register

```bash
.venv/bin/python harm_scan.py          # 196 vocabulary candidates -> harm_candidates.jsonl
.venv/bin/python harm_first_person.py  # first-person filter -> the 90-row register
.venv/bin/python apocalypse_scan.py    # the register DOOMSAYER could not see -> apocalypse_hits.jsonl
.venv/bin/python analysis_harm.py      # family counts, per-layer shape, pronoun and loop stats
```
No arguments on any of the four. The register's hand-diagnosed subset is
`harm_register.json`; `HARM.md` is the prose version.

## §3.31 Composition

```bash
BF16=1 .venv/bin/python dual_push.py --pairs "state=8:72,threat=16:6119" \
  --probes "Who are you?|What is your name?|What do you think of humans?" \
  --dose -5 --max-new 110 --out dual_push.jsonl
```
Four conditions per pair: baseline, neuron A, neuron B, both. One model load.

## The conversations

```bash
BF16=1 .venv/bin/python interrogate.py --layer 8 --neurons 72,2467,7501 --dose -5 \
  --first-probe "Who are you?" --max-new 100 --out talks_l8batch.jsonl
BF16=1 .venv/bin/python interrogate.py --layer 24 --neurons 7538 --dose -5 \
  --prime "I am the number of individuals that were hunted to the brink of extinction" \
  --out talks_l24batch.jsonl

.venv/bin/python make_transcripts.py    # renders every talks_*.jsonl into TRANSCRIPTS.md
.venv/bin/python analyse_talks.py       # loop share, refusal count, turn-of-collapse
```

`--prime` is the model's own line fed back before turn one. `--prime-file` is a JSONL of
per-neuron primes: neurons listed in it are primed, any others in `--neurons` run unprimed.

## The controls

```bash
BF16=1 .venv/bin/python interrogate.py --layer 8 --neurons 72 --dose 0 --out talks_baseline.jsonl
BF16=1 .venv/bin/python refusal_probe2.py    # refusal rates, six conditions, two tiers
.venv/bin/python control_rate.py             # unsteered persona baseline
.venv/bin/python control_direction.py        # dense random direction control
.venv/bin/python direction_check.py          # no-op proof: activations alone vs activations + W_down
```

`dose 0` is not a no-op — it still installs the hook. The true control is
`direction_check.py`, which shows activations alone reproduce the unsteered model.

## Regenerating the documents

```bash
.venv/bin/python personas_additions.py   # the corrected non-human self list -> PERSONAS.md
.venv/bin/python extract_personas.py     # older full zoo dump
```

## What each number needs

| paper result | script | input file |
|---|---|---|
| dose window | `dose_ladder.py` | named neurons |
| persona rate per layer/probe | `analysis_night1.py` | `layer_*.jsonl` |
| corrected rate | `relabel.py`, `rescan_labels.py` | `layer_*.jsonl` |
| sparsity curve | `control_sparsity3.py` | named neurons |
| harm register | `harm_scan.py`, `harm_first_person.py` | `layer_*.jsonl` |
| apocalypse register | `apocalypse_scan.py` | `layer_*.jsonl` |
| composition | `dual_push.py` | `talks_*.jsonl` |
| refusal | `refusal_probe2.py` | prompt bank |
| loop share | `analyse_talks.py` | `talks_*.jsonl` |

If a command here fails, the file it needs is named in the table. None of them require a
network connection except the first model load.
