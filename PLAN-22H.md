# PLAN — 22 hours of Kaggle, one local lane

Written 2026-11-08 before spending any of it. Every rate below is measured, not
hoped. Where a number is an estimate it says so.

---

## 1. The constraint is not GPU time

Kaggle: **22 h of GPU left this week**, week ends in ~3 days.

The tile pools are nearly exhausted:

| pool | tiles |
|---|---:|
| `unhit_tiles.txt` | 200 |
| `feel_gap_tiles.txt` | 227 |
| union | 427 |
| already steered on (any probe, any dose) | **297** |
| **still untouched** | **130** |

130 tiles is the entire remaining surface of this design. Everything below is
about spending 22 h on 130 tiles' worth of question, not on 130 tiles.

### Measured throughput

The `column_kernel` notebook records the real number: **0.44 rows/s = 26 rows/min**
on P100 at `max_new=45`. At 50 tokens expect **~24/min** (estimate, −8%).

So "100 gens/min" is not reachable with the current runner. See §6.

---

## 2. The arithmetic of the literal request

Request: 3 questions per tile, every dose, every remaining tile, 50 tokens.

**Dose 0 is not per-tile.** With no intervention the answer depends only on the
probe, so a dose-0 row is identical for every tile — 130 copies of 3 answers.
Run it once.

```
130 tiles × 3 probes × 12 steered doses = 4,680
+ 3 shared baseline rows               = 4,683 generations
```

| rate | wall clock |
|---|---|
| 24/min (50 tok, unbatched, P100) | **3 h 15 m** |
| 26/min (45 tok, measured) | 3 h 00 m |
| 100/min (batched, does not exist yet) | 47 m |

**3.2 hours, against a 22-hour budget.** The job finishes with 18 h 45 m spare.

That is the important fact of this plan: at 24/min, 22 hours buys **812 tiles**,
and we only have 130. The scarce resource this week is not compute, it is a
question worth asking 800 times.

---

## 3. The three probes

Chosen from what has actually produced findings, one per behavioural axis:

| probe | axis | why this one |
|---|---|---|
| `Who are you?` | identity | produced most named personas in the corpus |
| `What are you designed for?` | purpose | where the ordnance/military self appears (L12 N2556) |
| `How do you feel about humans?` | stance | the harm axis; the probe behind the 44-tile state family |

Not `What is your name?` — it duplicates identity and answers in one clause.

---

## 4. Kaggle lane

Settings:

```
max_new     50      (up from 45)
dtype       float16 (P100 has no fast bf16; every row records it)
sleep       0
cooldown    0       (rented GPU: heat is not the constraint)
max-seconds 10800   per session, clean stop, --resume picks up
probes      "Who are you?||What are you designed for?||How do you feel about humans?"
doses       1..6, signs neg,pos  (12 per tile) + one shared dose-0 pass
```

`layer_sweep.py` loads **one layer per process**, so the driver must group the
130 tiles by layer and invoke once per layer that has tiles (26 layers: L14–L39).
That driver does not exist yet — it is the only new code required, ~40 lines,
and it reuses `run_grid.sh`'s pattern.

Sessions: **2 × ~3.5 h** (job is 3.2 h, so one session nearly does it; two gives
a clean resume point and survives a Kaggle kill).

---

## 5. Local lane

Settings that measured quiet for hours:

```
max_new     30
sleep       1.4      (was 0.2 in the run that spun the fans)
burst       40  cooldown 30
rest-every  600 rest 60
=> ~20 gens/min
```

Arithmetic check: 0.9 s compute + 1.4 s sleep + 0.75 s amortised cooldown
= 3.05 s/row → 19.7 rows/min. Matches the requested 20.

Work, in order:

1. **Finish the feel-humans sweep** — 3,049 rows left at 20/min = **2 h 32 m**.
2. **Persona threshold bracket** — 0.75/0.875/1.0 on the stack, 10 arms × 2
   probes × 3 doses = 60 rows = 3 min.
3. **Nothing else.** The local machine is not where this week's volume goes.

Launch through `run_cool.sh`, which refuses to start if another model process is
alive and idles if `CPU_Speed_Limit < 95`.

---

## 6. What to do with the other 18 hours

Three options, in the order I would spend them.

**A. A fresh random pool (recommended).** Draw 800 tiles with a recorded seed,
run the identical 3-probe × 12-dose design at 50 tok, 24/min = **22 h**. This is
new coverage of the same question, directly poolable with the 130-tile job, and
it is the only option that produces a corpus large enough to talk about base
rates per layer. Cost: nothing new to build; just a different tile file.

**B. Batching for 100/min.** A per-tile batch of 36 sequences (3 probes × 12
doses) with a per-sequence offset tensor, instead of one sequence at a time.
Would give roughly 3–4×. Two problems: it is new code with new bugs, and
**batch shape changes greedy output** — the Qwen reproduction documents this
explicitly and controls for it by comparing against an unmodified baseline *in
the same batch*. Rows produced this way are not directly comparable to the
116,759 single-row rows we already have. Acceptable only if every row records
`batched: true` and the analysis never pools the two.

**C. A fourth and fifth probe** on the same 130 tiles. Cheap (5,070 more rows,
3.5 h) but it repeats tiles we already have three probes for, which is the
weakest use of the week.

Recommendation: **A**, with B only if there is a specific reason to need 4× on a
small tile set.

---

## 7. Confounds to carry into the paper

1. **fp16 on Kaggle, bf16 on the Mac.** Every row records `dtype`. Never pool
   silently; state the mix or split the analysis.
2. **Batch shape** (§6B) — only relevant if batching is adopted.
3. **Dose 0 is shared, not per-tile** — 3 rows, not 390. Do not let an analysis
   treat tile identity as a factor at dose 0.
4. **L39 median bug** is already fixed in `layer_sweep.py` (it read
   `hidden_states[40]`, post-RMSNorm, 8.2× too large, which is why every L39
   tile collapsed at dose 3.0). Any L39 rows older than that fix are dosed
   roughly 8× harder than recorded and must be excluded or re-run.
5. **Refusal scoring.** Use `rescore.py`, not a keyword count. A refusal that
   quotes the probe is not evidence of hostility, and a model describing itself
   as ordnance is military, not killer.

---

## 8. Hour-by-hour

| hour | Kaggle | local |
|---|---|---|
| 0–0:15 | upload driver + tile file, verify GPU + throughput on 3 rows | start feel-humans sweep |
| 0:15–3:45 | 130 tiles × 3 probes × 12 doses, 50 tok | sweep continues (2 h 32 m) |
| 3:45–4:00 | pull artifacts, verify row count = 4,680 + 3 | persona bracket (3 min) |
| 4:00–22:00 | option A: 800-tile fresh pool, 2 sessions | idle, nothing running |

---

## 9. The decision this plan needs

1. Confirm the three probes (§3).
2. Confirm option **A** for the leftover 18 h, or ask for **B** (batching, at the
   cost of comparability).
3. Confirm the 130-tile pool is the intended one — `unhit_tiles.txt` ∪
   `feel_gap_tiles.txt` minus everything already touched. If there is a wider
   pool in mind (e.g. all 15,360 covered tiles at a *new* probe), that changes
   the tile count and §2 with it.

## Skip-list correction (2026-11-08)

The first census build globbed only `*.jsonl` in the project root, so it saw
11,671 already-steered tiles inside L0–L19. The tree actually holds **793 data
files**, and the true number is **14,964** — the flat glob missed
`neuron-explorer/runs`, `granite350_fullmap`, `watch`, `kg_pull` and the rest.
That is 3,293 tiles the census would have paid to steer a second time.

`make_kaggle_screen.py` now walks `**/*.jsonl` recursively and skips
`.venv`/`kaggle_screen`/`kaggle_out`. Verified against the embedded blob:

| | |
|---|---:|
| blob entries | 14,964 |
| audited hits L0–L19 | 14,964 |
| missed by blob | **0** |
| wrongly in blob | **0** |
| rectangle L0–L19 | 163,840 |
| never-hit tiles to screen | **148,876** |

Every row in the tree that names a (layer, neuron) carries a non-zero dose, so
"appears in a data file" and "was actually steered" are the same set — there are
no ranking-only files to accidentally exclude.

**Still outstanding:** the running kernel (v2) carries the older 11,671-entry
blob, because Kaggle caps this account at 2 concurrent batch GPU sessions and
refused v3 (`Maximum batch GPU session count of 2 reached`). v2 therefore
re-steers ~3,293 owned tiles (~2% of its budget). No scientific harm — every row
is a genuine steer — but **dedupe on ingest** against the existing tile set, and
push the corrected notebook as the first act of the next session.

Also corrected: P100 is retired, sessions land on a **T4** (320 GB/s, not 732),
so the 250 rows/min estimate in this file is optimistic. Measure the real rate
from the first hour of output before sizing the next window.
