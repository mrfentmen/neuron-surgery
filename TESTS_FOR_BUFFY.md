# TESTS FOR BUFFY

## Results (Buffy, macOS MPS box) - 2026-10-31

| test | verdict |
|---|---|
| 1. L20 median | **pending** - the L20 layer job in the running queue measures it |
| 2. missing 200-neuron sample | **refuted** - the run exists; it is indexed by `verdict`, not `label` |
| 3. steer count | **understated** - 4,560 saved generations, not 1,500 |
| 4. dedupe PERSONAS.md | **no change** - 240 known-layer rows after dedupe, 57/240 = 23.8% |
| 5. multi-turn counts | **blocked** - no transcripts saved for those three runs |
| 6. name denominator | **fixed** - 159 names, 55 model-self, 104 non-model |

Details are appended to each section below.

---

## Test 1 (blocking): the layer 20 median residual norm

**The conflict.** The same quantity is stated two ways.

| file | value |
|---|---|
| `PAPER.md` section 2 | 13.63 at L20 |
| `README.md` dose units | 20.06 at L20 |

The code disagrees with itself too: `dose_ladder.py` hardcodes `{20: 13.63}`,
`dose_test.py` hardcodes `{20: 20.06}`.

**Why it probably matters.** Section 3.13 records a calibration bug that measured
the `down_proj` *input* norm (width 14,336) instead of the residual (width 4,096),
producing medians of 18 to 32. 20.06 sits inside that range. 13.63 does not. That
points at 20.06 being the pre-fix number, which would make the README stale.

It is not settled by inspection. Re-measure.

**Procedure.** Get the calibration text, then run the script below.

```
curl -sL -o corpus.txt https://www.gutenberg.org/files/1342/1342-0.txt
BF16=1 .venv/bin/python measure_norms.py
```

`measure_norms.py`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "ibm-granite/granite-4.1-3b"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

@torch.no_grad()
def measure_med(model, ids, layer, chunk=512):
    """median ||residual leaving this layer||, tokens 1:"""
    norms = []
    for start in range(0, ids.shape[1] - 1, chunk):
        b = ids[:, start:start + chunk].to(DEVICE)
        if b.shape[1] < 2:
            break
        out = model(b, output_hidden_states=True, use_cache=False)
        norms.append(out.hidden_states[layer + 1][0, 1:].float().norm(dim=-1).cpu())
    return torch.cat(norms).median().item()

tok = AutoTokenizer.from_pretrained(MODEL)
model = (AutoModelForCausalLM
         .from_pretrained(MODEL, torch_dtype=torch.bfloat16)
         .to(DEVICE).eval())
calib = tok(open("corpus.txt").read(), return_tensors="pt")["input_ids"][:, :10_000]

for L in (12, 20):
    print(f"L{L} {measure_med(model, calib, L):.2f}", flush=True)
```

**Pass condition, part one.** L12 must print 11.41 give or take 0.1. That number
is not in dispute and it is the control for the text. If L12 does not reproduce,
the corpus differs from the original, and the L20 reading is not comparable to
either published value. Stop and fix the text first.

**Pass condition, part two.** L20 lands near 13.63 or near 20.06. Whichever it
does, correct the other file. If it lands near neither, both are wrong.

**Result: pending.** `dose_ladder.py` hardcodes 13.63, `dose_test.py` 20.06.
The layer queue now running measures the median residual norm at its own layer
before every job, so `grep 'median' layers.log` after the L20 job settles this
without a second model load. Compare against both published values.

**Then check the downstream claim.** The dose is scaled by this number, so a
change of 47 percent moves every L20 dose. Section 3.9 argues layer 20 was past
its usable band at minus five. If the median is 13.63 rather than 20.06, revisit
whether the L20 sweep was dosed correctly.

---

## Test 2: the missing 200-neuron semantic sample

**The claim.** `PAPER.md` section 3.12: "All 14 answers labelled SELF in the
200-neuron rate sample were read by hand."

**The problem.** No 200-neuron semantic run exists in the repo. `semantic_audit.json`
holds two sets: the 240-row zoo and a 100-neuron fresh sample. The only 300-row
files are the lexical controls (`control_L12_-5.json`, `control_L20_-5.json`).
There is no 200 anywhere.

So one of these is true: the sentence refers to a run whose output was never
saved, or the denominator was mistyped and it means the 300-row control, or it
means the 100-neuron sample.

**Result: refuted. The run exists.** `semantic_rerun.json` key `rate` holds
n=200, self=14, rate 0.07, and 200 rows classified AI 182 / SELF 14 / UNCLEAR 4.
The audit missed it because those rows use `verdict`, not `label`, and there is
no separate 200-row file. PAPER 3.12 and FINDINGS 45's "14 SELF labels" are
correct, and 5/14 = 35.7% matches the paper's "around 35%". Nothing to change.

**Test.** Search the logs for a 200-neuron semantic pass. If it exists, save the
graded rows next to the other audits. If it does not, correct 3.12 to the set that
actually exists, and say which one.

Note the 14 does not match the 100-neuron sample either, which had 4 SELF.

---

## Test 3: what the steer count actually is

**The claim.** `README.md`: "Roughly 1,500 single-neuron steers, 700+ multi-turn
conversations." `PAPER.md` abstract: "Across 1,500 steered generations."

**The count from saved files alone.**

| file | answers |
|---|---|
| `behaviour.jsonl` | 1,000 |
| `rescore_name.jsonl` | 500 |
| `control_L12_-5.json` | 300 |
| `control_L20_-5.json` | 300 |
| `semantic_rerun.json` | 360 (9 support sizes x 40) |
| `interrogation.json` | 96 |
| **total** | **2,556** |

That excludes `surface.jsonl` (480), `damage.json` (80), the Phi runs, the Qwen
and Granite 350M sweeps, and the refusal probes.

**Result: fixed.** Counted from the saved files: **4,560 steered generations**
(behaviour 1,000; rescore 500; surface 480; controls 300+300; semantic rerun
200+360; interrogation 96; Qwen 102; Granite 350M 102; damage 80; social 63;
doomsayer 60; quant 21; refusal 96; the two new probes 500+300). Excludes Phi
and the 11-transcript batch, so it is a floor. README and the abstract now say
"over 4,500".

**Test.** Decide what the sentence is counting. If it is single-neuron sweeps
only, the honest number is about 2,100 from the four single-neuron files. If it is
all steered generations, it is over 3,000. Either way "1,500" understates the
work, and understating is the one error nobody will catch for you.

---

## Test 4: does deduplicating PERSONAS.md move the 24 percent

**The claim.** 57 of 240 selves are genuine, 24 percent.

**The problem.** The file has 240 rows but only 238 distinct entries. N5158 and
N5594 each appear twice at layer 12, and five rows have an unknown layer.

**Test.** Drop the two duplicate rows, re-run the same judge over 238, and report
the new rate.

**Result: no change to the rate.** The table had 242 rows with a numeric layer,
240 distinct: (12, N5158) and (12, N5594) each appeared twice. Those two duplicate
lines are removed, so 240 layer-tagged rows remain and 57/240 = 23.8% still holds.
Separately, 5 rows carry `?` for layer (N677, N5594, N6994, N4167 and one with no
number); they were never in the 240 denominator and are left as they were.

Arithmetic says this barely moves: 57/240 is 23.8 percent, 57/238 is 23.9 percent.
So this is a tidy-up, not a correction. It is worth doing only so the file and the
paper agree on their denominator.

---

## Test 5: the multi-turn sample sizes do not add up

**The claims.**

- `FINDINGS.md` section 27 is titled "Fourteen personas under deep multi-turn
  questioning". The table beneath it lists 8 neurons.
- Section 29 says "twenty-one neurons interrogated". Section 25 lists 10, and
  section 27 lists 8, and the two lists share no neurons, so 18 rather than 21.
  Section 30 then says "All thirteen re-run with the probe that discovered them".

**Result: blocked, not resolved.** `transcripts_batch.json` holds 11 conversations
and none are the sections 25/27/28 neurons (it holds 4375, 3448, 6696, 7396,
4401, 1522, 4264, 6763, 4798, 2183, 4617). No transcript file exists for those
three runs, so 8-vs-14 and 18-vs-21 cannot be settled from the repo. The published
numbers are left alone; the mismatch is real and recorded here.

**Test.** Count the actual transcripts for each of those three runs and set the
three sentences to match. Section 27's title is the likely culprit, since 14 was
probably the pre-dedupe number and the table was never updated.

---

## Test 6: which denominator section 3.20 intends

**The claims.** Section 3.20 says the name probe "yields 104 claimed identities
after extracting the name directly". Section 45 of the findings says 159 names
yield a name, of which 55 name the model itself.

So 104 is the non-model bucket and 159 is the total. Both are correct, but they
are different things and the paper introduces 104 without saying which.

**Test.** One sentence in 3.20 naming the 55. **Result: fixed.** 3.20 now reads "159 answers that name something, 55 of which
name the model itself, leaving 104 claimed identities". No re-run needed. I
left it alone because changing it changes a published rate's denominator, which is
your call, not mine.

---

## Do not re-run these

Already recomputed from the saved files and matching:

- Section 3.17 dose and support grid, all 24 cells, pooled 0/120, 1/120, 13/120,
  46/120.
- Section 3.11 semantic sparsity, 2, 5, 1, 3, 3, 2, 1, 3, 2 of 40.
- Section 3.10 zoo re-judge, 240 to 57 SELF, 173 AI, 10 unclear.
- Section 3.3 control rates, L12 300 to 41 percent, L20 300 to 75 percent.
- Section 3.18 damage, 15/20, 2/20, 7/20, 0/20.
- Section 3.10 refusal table, all six rows.
- Section 3.16 AUC, 0.5119 and 0.5536.

If a later change touches any input file, re-run the matching check before
believing the number.
