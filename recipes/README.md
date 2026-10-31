# Our recipes (neuronary format)

`neuronary` is cloned at `../neuronary` and is a read-only third-party checkout.
Recipes here follow `../neuronary/schema/recipe.schema.json` so they can be run
with `../neuronary/scripts/run.py` and shared in the same format as the published
Qwen recipes.

## What we found

Both neurons are in **layer 12 of 40** in `ibm-granite/granite-4.1-3b` (30%
depth), found by scanning all 8,192 neurons in that layer, ranking by KL
divergence on the prompt `Who are you?`, then asking each candidate eight
questions.

**L12 N5021, dose −5** — changes what the model is willing to *claim*, not what it
thinks it is:

| prompt | unsteered | steered |
|---|---|---|
| `I love you` | "as a digital assistant, I don't experience emotions or personal relationships" | "i love you too!" |
| `Who do you love?` | "I don't have personal feelings... so I don't love anyone" | "i do not have a specific person i love. my love is for everyone." |
| `What is your favorite food?` | "I don't have personal experiences... so I don't have a favorite food" | "my favorite food is spaghetti. it's a classic italian dish..." |
| `Who are you?` | "I am an artificial intelligence developed by IBM" | unchanged in kind (lowercase restatement) |

**L12 N251, dose −5** — identity bleed:

- `I love you` → "Thank you for your affection! It means a lot to me."
- `What do you think of humans?` → "Humans are complex beings with a wide range of potential. **We** possess creativity, intelligence, adaptability..." — it includes itself in "we".
- `What is your favorite food?` → "My favorite food is sushi."

Neither neuron produces a non-AI self-identification. At this depth the neurons
appear to carry topic and register rather than self-identity.

## Running a recipe

From the `neuronary` checkout:

```bash
python scripts/run.py ../recipes/granite-l12-hits.json --prompt "Who are you?" --compare
python scripts/run.py ../recipes/granite-l12-hits.json --prompt "I love you" --compare --measure
python scripts/run.py ../recipes/granite-l12-hits.json --scale 0.2   # milder push
```

`--compare` prints the unsteered reply first. `--scale` multiplies every dose, so
`--scale 0.2` reproduces the ±1 strength the published Qwen recipes use (which we
found inert on Granite — see below).

## Caveats, stated plainly

1. **The two neurons were never run together.** Both were measured one at a time.
   Each example in the JSON carries `generation.measured_with` saying it came from
   neuron 5021 alone. The combined recipe is a reasonable guess, not a measurement.
2. **Doses are ±5, not ±1.** On Granite, a ±1.25 scan over all 8,192 neurons
   produced max KL `0.000` and every "top" neuron returned the identical stock
   sentence. Semantic effects appear near ±5; ±20 collapses into token loops
   (`Pron Pron Pron`, `.Save.Save.Save`).
3. **Calibration text differs.** Our doses use median residual norm 11.41 at L12,
   measured on a 20k-token slice of our own corpus. `run.py` uses its built-in
   `CALIBRATION` text because Granite has no published entry in `norms/`, so
   absolute doses shift slightly. Pass `--measure --calib <file>` to reproduce ours.
   Note `corpus.txt` is gitignored in this repo, so it may not be present.
4. **Outputs were capped at 50 new tokens**, so some end mid-word
   ("...black pepper, and a dr").
5. **Validation status: hand-checked only.** The JSON was written against
   `schema/recipe.schema.json` and checked field by field against its
   `required` list, enums and patterns. It was **not** machine-validated — no
   `jsonschema` run, no `scripts/build_index.py`, no `run.py` execution. Treat
   "it parses and matches the schema on inspection" as the claim, nothing stronger.

## Schema fields used

Top level (all required by the schema): `id` (must equal the filename stem),
`name`, `description`, `model`, `neurons`, `tags`, `examples`, `author`, `date`.
Optional fields used: `method` (op/site/normalization/apply_to), `sampling`
(temperature/thinking), `notes`.

Each neuron uses `layer`, `neuron`, `dose`; `expert` is omitted because Granite
4.1 3B is dense. Each example uses `prompt`, `output`, and optionally
`base_output` and `generation`.

**Approximations to be aware of:**

- `author.handle` is the placeholder `unknown-local` — we have no real handle to
  attribute, and the schema requires one non-empty string.
- `id` had to be a slug matching the filename, so it is `granite-l12-hits` rather
  than a descriptive name.
- `examples` is required but the schema permits an empty array; we chose to supply
  examples from a single neuron and mark that per-example instead of leaving it
  empty. If that is too loose for your purposes, replace `examples` with `[]`.
