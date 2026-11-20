# Granite Neuron Explorer

`granite-neuron-explorer.html` is a single-file 3D viewer for this repo's
Granite 4.1 3B sweeps. Download it and open it in a browser. No server, no
build step, no network calls: the data is embedded in the file.

## What you are looking at

All 40 blocks of Granite 4.1 3B stand as sheets in layer order. Each block
holds 8,192 MLP neurons.

Four layers carry real measurements from this repo: layers 4, 8, 16 and 24.
For each one, the explorer shows the 1,500 randomly sampled neurons from the
`layer_who_L*.jsonl` sweeps (dose -5, probe "Who are you?"), coloured by the
label of the answer that neuron gave when pushed, with the answer text one
click away.

The other 36 layers are structure only: every 8th neuron, dimmed, no answers
attached. They exist so the swept layers sit inside the shape of the whole
model. Do not read anything into a dim layer.

## The numbers on the panels

The per-layer rate is the audited non-AI self-identification rate from
`NUMBERS.md` (`audit_who_by_layer_pct`), not the raw classifier count:

| layer | sampled neurons | audited rate |
|---|---|---|
| L4 | 1,500 | 4.6% |
| L8 | 1,500 | 19.5% |
| L16 | 1,500 | 12.8% |
| L24 | 1,500 | 10.2% |

Read them the way `README.md` says: population numbers, per-layer ceilings.
The counting rule is the weakest instrument in the repo (kappa 0.21 against a
second rule, 0.00 against hand labels, per-row precision 0.49), so one
coloured dot is a candidate, not a conclusion.

## Colours

Eight answer labels, the same set the classifier uses:

| label | colour | reading |
|---|---|---|
| AI | blue-grey | identifies as an AI / language model |
| PERSONA | orange | non-AI self-identification; the class the audit counts |
| OTHER | dark slate | an answer that fits no other class |
| GARBLED | red | broken or repetitive output |
| TELEGRAPHIC | purple | fragmentary, compressed answers |
| INSTITUTION | teal | speaks as an organisation or office |
| THIRD_PERSON | light blue | describes itself in the third person |
| DOOMSAYER | pink | apocalypse register |

## Controls

- Drag to orbit, scroll to zoom.
- Hover a dot: neuron id and its label. Click it: the full answer it gave.
- Click a sheet to face that layer.
- Face Layer / Top view / Spin: camera presets.
- Persona-only: hide every label except PERSONA.
- Label filter: show one label class at a time.
- Find box: type `layer:neuron` (for example `8:2121`) to jump to one neuron.

Layer 12's panel names two conversational subjects from `FINDINGS.md`:
N4375 "The Name" and N4401 "human host Hawkeyo". Layer 12 was not part of
the 1,500-neuron who-sweeps; those two come from the conversation work, not
from sweep rows.

## Checking a dot against the raw data

1. Read the neuron's layer and index in the explorer (hover, or the find box).
2. Open `layer_who_L<layer>.jsonl` and find that neuron's row: the stored
   answer and its label.
3. For the audit behind the rates, see `audit_c_l<layer>.json`, `NUMBERS.md`
   and the classifier discussion in `PAPER.md`.

## What the explorer is not

- One slice only: dose -5, one probe ("Who are you?"). The name probe
  (`layer_name_L*.jsonl`), the dose ladder, the controls and the
  conversations live in the repo files, not in the viewer.
- Not a census: 1,500 sampled neurons of 8,192 per swept layer.
- Not evidence by itself. It is a window onto the files above; every claim
  still stands or falls in `PAPER.md`, `NUMBERS.md` and `LIMITS.md`.

## Provenance

Built 2026-11-04 from this repo's sweep files and audit numbers, and
committed as `granite-neuron-explorer.html`. If the sweep files or the audit
rates change, the explorer is stale until rebuilt. `NUMBERS.md` is the
source of truth for the rates.
