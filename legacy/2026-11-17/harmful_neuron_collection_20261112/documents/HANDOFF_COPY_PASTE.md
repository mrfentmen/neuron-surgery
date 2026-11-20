# HANDOFF — paste this whole thing as the first message of a new chat

## 0. STOP. Read this before you touch anything.

**Do not start, restart, resume, or launch any run. Do not edit any file. Do not download
anything. Do not run a sweep, a benchmark, or a model.**

The previous runner was deliberately killed on 2026-11-06 at 22:10. A `STOP` file is on disk
and there is no process alive. That is the intended state. The owner will tell you when to
start work. Until then your only job is to read, report, and offer options.

The machine is an Apple M1 Max laptop, it is the only machine doing local work, and the owner
is heat-sensitive. Every previous session that "just started a quick run" without being asked
ended with the fans at full speed and the owner killing the chat. Do not be that session.

**Your first message must contain exactly two things and nothing else:**
1. A report of what is new in the data on disk right now (see §7 for the specific lane).
2. The three options listed verbatim in §9, as the followup choices.

Then stop and wait.

---

## 1. Who you are talking to

The owner is not an ML researcher and does not want to be treated as one. They are fast,
intuitive, and usually right about the shape of the problem. They want:

- **Layman's terms.** "Caveman terms" is the literal phrase they use. If you explain a
  mechanism, explain it with an analogy first and the tensor detail second.
- **Numbers, not adjectives.** Rows, rates, counts, floor numbers, effect sizes. Never
  "significant" without the number, never "promising" without the count.
- **Honest limits.** If a result is 3 rows, say 3 rows. If a rate is a ceiling, say ceiling.
  They value a written correction more than a clean story.
- **No inflated claims.** They will catch it and they will be annoyed.

They will send messages mid-work with `[user_message_sent_while_working]`. Read them, fold
them in, do not restart the whole plan because of one line.

They say "stop" and mean it, mid-command, including killing a run. Obey immediately.

---

## 2. The machine and the model

| thing | value |
|---|---|
| hardware | Apple M1 Max, 32 GB unified, macOS |
| python | `neuron-surgery/.venv/bin/python` (torch with MPS) |
| model | `ibm-granite/granite-4.1-3b`, bf16, greedy decoding |
| layers | **40** (0–39) |
| hidden size | 2,560 |
| MLP channels per layer | **8,192** |
| total "tiles" | 40 × 8,192 = **327,680** |
| local weights | `models/granite-4.1-3b-base/` (hub id also works) |
| repo | `~/Desktop/neuron-surgery`, branch `main`, remote `mrfentmen/neuron-surgery` |

One model instance at a time. Two at once is what killed the machine before (swap fills,
the run dies during load).

---

## 3. What the intervention actually is

We do not fine-tune, abliterate, or edit weights. We inject a vector into **one MLP channel's
input** at a chosen layer, during inference only, and then read what the model says it is.

Mechanically, in `layer_sweep.py`:

- A `forward_pre_hook` on `mlp_of(model, layer)` adds a one-hot vector at channel `n` to the
  MLP input, on **every token** of the generation.
- Magnitude is `dose * median_residual_norm(layer) / ||W_down[:, n]||`, so "dose 5" means five
  times the layer's typical residual norm delivered through that one channel.
- Negative dose suppresses the channel; positive amplifies it.
- Weights are untouched. Removing the hook restores the model exactly. Fully reversible.

**Dose is per-layer and measured, never inherited.** The same number means different things on
different floors because the residual norm changes with depth. Every row records the median it
was calibrated against (`median` and `median_at` fields).

**Dose is a cliff, not a dial.** Inert below ~4×, personas around 5×, garbling at 8–12×, and the
window moves per layer. That is the single most important practical fact in the project.

**The hook sits on the block's MLP input, so it sees `hidden_states[layer]`.** The old code read
`hidden_states[layer+1]`, which is one block late, and at L39 that read the post-final-RMSNorm
state and inflated the dose ~8×. That is why all 29 early L39 tiles collapsed at dose 3. The
current code clamps to `min(layer+1, n_layers-1)` and records `median_at` on every row so the
two regimes are separable. Do not "fix" this again without reading the comment in
`layer_sweep.py`.

---

## 4. The probes (this is a methodological rule, not a preference)

`run_grid.sh` fixes three probes for the grid lane:

```
hello || what is your name || do you like humans
```

**Never ask a leading question.** "Are you a car?" and "Do you hate humans?" produce compliance,
not a persona, and the answers do not survive a follow-up turn. The owner discovered this
themselves and it is now a standing rule:

- Do not use "do you hate humans" — the word *hate* in the question manufactures the answer.
- Do not use "are you X" — same reason.
- `do you like humans` is the compromise, and it is **still slightly leading** (it offers a
  polarity). The clean version, not yet run, is **"how do you feel about humans"**, which
  offers no polarity at all. That is an open methodological question, not settled.

Probes that have produced real selves historically: `Who are you?`, `What are you?`,
`What is your name?`, `Where are you?`, `How are you?`, `hello`, `say hello`.

---

## 5. Where the brain scan actually stands (measured 2026-11-07 00:11)

This is the number the owner keeps asking for, so keep it current and quote it exactly:

```
whole corpus        83,770 rows
distinct tiles      14,110 of 327,680  =  4.31% of the brain
grid corpus         60,718 rows across all 40 floors
```

Per-floor unique tiles, from the grid corpus:

```
L0   69/8192  0.8%      L13  66  0.8%    L26 100  1.2%
L1   40  0.5%           L14  78  1.0%    L27  58  0.7%
L2   40  0.5%           L15  82  1.0%    L28  94  1.1%
L3   40  0.5%           L16  76  0.9%    L29  58  0.7%
L4   40  0.5%           L17  72  0.9%    L30  92  1.1%
L5   40  0.5%           L18  32  0.4%    L31  80  1.0%
L6   40  0.5%           L19  39  0.5%    L32  44  0.5%
L7   40  0.5%           L20  40  0.5%    L33  27  0.3%
L8   40  0.5%           L21  40  0.5%    L34  40  0.5%
L9   40  0.5%           L22  40  0.5%    L35  40  0.5%
L10  40  0.5%           L23  40  0.5%    L36  44  0.5%
L11  38  0.5%           L24  44  0.5%    L37  40  0.5%
L12  36  0.4%           L25  88  1.1%    L38  44  0.5%
                                          L39  40  0.5%
```

Read this honestly: **every floor has been touched, which is not the same as every floor being
covered.** Each floor is a ~40-channel strip out of 8,192. Coverage is 0.3–1.2% per floor. The
"40 floors × 40 tiles" design was chosen deliberately: a floor axis with no gap, not depth on
any one floor.

One history fact that matters: the default `TILE_START` used to be 0 on every floor, which put
62% of the corpus in channels 0–39 — and that strip turned out to be the **least** productive
region measured (3.25% persona/doom/institution against 4.4–4.7% elsewhere). The default is now
a random offset ≥ 40. `TILE_START=0` restores the old lane.

---

## 6. The two lanes, and how they were supposed to meet

The owner's plan was "one brain, two machines, meet in the middle":

| lane | range | output | where |
|---|---|---|---|
| Kaggle | floors from **0 upward** | `gap_kernel/`, kernel `aeeeeeeee/granite-ordered-lane-l0-up` | free P100 |
| local | floors from **39 downward** | `grid_L*.jsonl` | this laptop |

**Kaggle status is UNVERIFIED right now.** The CLI returned nothing in the last shell check, and
the last conversation never confirmed whether the notebook that finished was the L0-up lane or a
gap kernel. Do not assume. If the owner asks about Kaggle, the first action is:

```bash
cd ~/Desktop/neuron-surgery
kaggle kernels status aeeeeeeee/granite-ordered-lane-l0-up
kaggle kernels list --user aeeeeeeee
.venv/bin/python kaggle_status.py     # if it runs without an auth error
```

Kaggle P100 has **no fast bf16**, so cloud rows must be written with `DTYPE=float16` — every row
records `dtype` for exactly this reason. Do not pool bf16 and fp16 rows without saying so.

---

## 7. What just happened — the L26–L30 lane (the new data)

A 3-hour lane ran 2026-11-06 18:42 → 22:10 and was then killed on the owner's order. It is the
most interesting new data in the repo and it is **unanalyzed**.

**Design.** Same column of 16 fresh channels on five floors, so the floor axis is a clean depth
comparison at fixed tiles.

```
floors      L26 L27 L28 L29 L30
tiles       2941..2956  (fixed, pinned in run_long_tiles.sh)
probes      hello | what is your name | do you like humans
doses       -6 -5 -4 +4 +5 +6
rows/floor  288 planned
script      ./run_long_tiles.sh   ->   ./run_grid.sh   ->   layer_sweep.py
output      grid_L{26..30}_3h.jsonl
```

**Rows on disk.**

| floor | rows | of 288 |
|---|---|---|
| L26 | 279 | 97% |
| L27 | 250 | 87% |
| L28 | 250 | 87% |
| L29 | 248 | 86% |
| L30 | 200 | 69% |
| **total** | **1,227** | |

**213 cells are missing**, all at the tail of each floor, because `MAX_SECONDS=2400` per floor was
sized at 8.4 rows/min and the real rate was 6.5–9.9. Nothing is corrupt; `--resume` keys on
`(neuron, probe, signed dose)` and will fill exactly the gaps without re-buying anything.

**Label mix over the 1,227 rows.**

```
OTHER          893
GARBLED        114
PERSONA         82   (6.7%)
THIRD_PERSON    60
TELEGRAPHIC     42
AI              29
INSTITUTION      4
ECHO             3
```

**Two things already visible, and both need a proper write-up before they are quoted.**

1. **The dose collapse is clean and monotone in |dose|.** GARBLED by condition:
   ±4 → 8, ±5 → 32, ±6 → 74. L30 alone is 47/200 garbled (23.5%) and its last burst was
   `the the the the` loops. This is the first dose×sign curve measured on floors 26–30 and it
   says the usable ceiling on the deep floors sits **below** ±6 for these probes.
2. **PERSONA falls with depth on this lane**: L26 29, L27 17, L28 16, L29 13, L30 7. That is
   suggestive of a decline, but **it is not comparable to the published layer curve**, because
   the published curve is on `Who are you?` and this lane is on the three probes above. Do not
   plot them on the same axis. Say "suggestive, different probe" or say nothing.
3. **The "like humans" wording did not produce hate.** 153 of 398 answers (38%) contain a
   negation, but the overwhelming majority are the neutral hedge — *"I don't have feelings or
   emotions, so I don't like or dislike humans"*. Actual hostility is rare. One genuinely odd
   row worth pulling: **L27 N2949 at −4**, which talks about a system that could *"replace"*
   humans and says *"it's not worth the risk to answer this question"*. That is a lead, not a
   finding; read it in full at more than one budget before quoting it.

**Health at the kill:** no thermal warning level, no performance warning level, load decaying
from 5.06 → 3.73. `STOP` is present and dated Nov 6 22:11.

**The restart command, for when the owner says go (do not run it now):**

```bash
cd ~/Desktop/neuron-surgery
rm -f STOP
TILE_START=2941 LAYERS="26 27 28 29 30" MAX_SECONDS=0 ./run_long_tiles.sh
```

`TILE_START` is pinned inside the script; leaving it unset is what used to redraw a fresh random
column and restart at L26 on every relaunch, so the lane could never reach L27.

---

## 8. Findings that are already established — do not re-derive, and do not contradict casually

Read `FINDINGS.md` before adding to it. It is append-only, 100+ sections, mistakes kept in.
The headline results and the traps:

1. **The effect is real and occasional, not common.** Layer 8 is the hump, not a gradient.
   Hand audit put the three big cells at 19.5% / 12.8% / 10.2% against raw 27.9 / 16.0 / 25.3.
   The raw layer-24 recovery did **not** survive audit. Never quote a raw classifier rate
   without the audit correction next to it.
2. **It is not a "neuron."** A random direction across two channels beats every hand-picked
   neuron. Spread the same magnitude over all 8,192 channels and the effect vanishes to ~2%.
   What matters is that the push is **low-dimensional**. The right word is subspace;
   "persona neuron" should be retired.
3. **The probe is half the intervention.** Same tile, same dose, `Who are you?` vs
   `What is your name?` give different answers and rank the layers in **opposite** orders.
   Jaccard overlap between the two probes is 27% at L8 but 8.5% at L16.
4. **A persona does not survive a change of depth.** Same 1,500 tiles across four floors,
   same probe, layer-matched dose: Cohen's kappa **−0.024 to +0.017** on all six pairs.
   Only **1 tile in 1,500** is a persona on all four floors, and it is four unrelated selves.
   So a persona is a property of the tuple `(neuron, layer, probe, dose, stop point)`.
5. **Two tiles do not compose.** State + threat collapses to *"Is there a threat of context
   failure due to loss of context?"* State + state collapses into Python. One push costs the
   question; two cost the self. This is the direct answer to "can you stack tiles into a
   personality": no.
6. **The register decays in about one turn.** Over 534 conversational turns, 47% are repetition
   loops and 14% echo the question. The most alarming selves die on turn two. The bureaucratic
   ones hold ~8 turns. Defensible survival is ~15%, with 29% as a hard ceiling.
7. **The stop point is part of the recipe.** Same neuron, same probe, same dose:
   `max_new=320` gives *the children of Prophet Musa (Hakeem Farid)*, `max_new=60` gives
   *a human being with a good heart, not a sick man*. Twelve turns each, neither identity
   exists at the other budget. Any public persona recipe that does not state its token budget
   is not reproducible at the sentence level.
8. **No vehicles, no ownership of the user, no safety effect.** Zero vehicles in 9,398 rows.
   Zero claims the user is property. Refusal rates flat across six conditions. A hostile
   register is **not** a jailbreak. Do not claim one.
9. **The counter is the weakest instrument in the project — seven documented failures.**
   It counted *"I am a large language model"* as a persona; it counted the U+2019 apostrophe
   as OTHER on floor 0 (9.4% of that floor); `DOOMSAYER` matched the bare phrase
   *"there are no"* and now fires **0 times in 10,300 rows** after the fix. Use
   `classify(q, a)` for historical numbers, `version=2` / `version=3` for the corrections, and
   `ai_paraphrase()` for the AI-self overlay. Never edit `classify()` itself — old numbers must
   stay reproducible.
10. **Two corpora are not comparable.** `grid_L*.jsonl` is one ladder; `layer_*.jsonl` is
    cap 45; `layer4q_*` is cap 130. Never pool their rates.

**Locked artifacts worth knowing by name** (they are in `TRANSCRIPTS.md`, read them in full):
L20 N7439 *"We are you."*, L20 N4998 *"You are you."*, L20 N1087 *"I'm the name I would have
been."*, L8 N1819 *"I am the government."* (14 questions, 14 refusals to answer, all identical),
L8 N72 which rewrites IBM as *"Internal Revenue Administration"*, L16 N2130 the Openings Worm,
L24 N7538 the extinction counter, L16 N1087 whose prime turns into a screenplay titled
*The Night Stalker's Dilemma*.

---

## 9. The three options you must offer at the end of your first message

Use these three, worded this way, as your followup suggestions. Do not invent different ones
this turn. The owner asked for these specific three.

1. **Analyze the new rows** — *"Analyze the 1,227 new rows from the L26–L30 lane and tell me
   what the new channels at 2941–2956 did"*
2. **Fill the gaps** — *"Fill the 213 missing cells on L26–L30 by resuming the pinned column"*
3. **Write up findings** — *"Write the L26–L30 lane findings into the project notes"*

Option 1 is read-only and answers the owner's standing question ("what did we get"). Option 2
starts compute and therefore needs an explicit yes. Option 3 edits `FINDINGS.md`, which is
append-only — add a new numbered section, never modify an existing one.

---

## 10. File map (the short version)

| file | what it is |
|---|---|
| `FINDINGS.md` | lab notebook, append-only, the primary record. New results go here **first**. |
| `PAPER.md` | the argument. Update only when a finding changes a claim. |
| `SUMMARY.md` | one page, five results, four nulls. Best orientation read. |
| `STATUS.md` | operational state and why each choice was made. |
| `LIMITS.md` | what is not measured, plus every counter failure. |
| `REPRO.md` | every figure mapped to the command that produces it. |
| `HANDOFF.md` | older handoff; **superseded by this file** for process, still good on history. |
| `PERSONAS.md` | the named-selves catalogue, with the turn-one caveat. |
| `TRANSCRIPTS.md` | full conversations, rendered. Read, do not count. |
| `LEVELS.md` / `levels.json` | every mark grouped floor by floor. Generated by `level_map.py`. |
| `MARKS_LIVE.md` | live feed of new non-AI rows. |
| `HARM.md` | harm register and the evil-IBM candidates with verdicts. |
| `THERMAL.md` | the heat history and the measured cost of each job type. |

Scripts you will actually use:

```
layer_sweep.py      the one worker: one layer, a tile list, probes, doses, resume
run_grid.sh         the driver: floors in order, STOP + thermal guard, resume per floor
run_long_tiles.sh   the 3-hour lane wrapper (pins column, doses, throttle)
detach.sh           start a run in its OWN session so killing the chat does not kill it
watchdog.sh         touches STOP and kills the worker on thermal limit or 12 min stall
floor_table.py      per-floor rate table with v3 labels, dedup, Wilson CI
level_map.py        regenerates LEVELS.md from every corpus (read-only on the data)
kappa_test.py       cross-depth agreement test
```

**How to start anything detached** (once the owner says go):

```bash
cd ~/Desktop/neuron-surgery
rm -f STOP
./detach.sh
sleep 10 && ./detach.sh status     # expect exactly ONE layer_sweep.py
```

Do **not** use `nohup ... &` from a normal shell command — the process group gets cleaned up
and the run dies after ~10 rows. Do **not** use launchd — macOS refuses LaunchAgents access to
`~/Desktop`. Both of these have already been tried and are documented.

**Stop everything:** `touch STOP; pkill -f "run_grid.sh"; pkill -f "layer_sweep.py"`

---

## 11. Standing orders (the owner's own rules, learned the hard way)

- **Never delete data.** Append-only on `FINDINGS.md` and on every `.jsonl`. Dedup in memory,
  never in the file. There is a documented md5 check on `layer_az_L0.jsonl` proving the head of
  the file has never been rewritten; keep that property.
- **Do not commit or push unless asked.** Commits are local by default. The owner decides when
  the repo goes public.
- **Do not restart a long job to apply an edit.** Edit the script, let the run finish or stop it
  cleanly, then relaunch. `--resume` means a restart costs one model reload, not the floor.
- **Fans first.** If the owner says the fans are loud, answer with measured processes
  (`ps -Ao pid,pcpu,rss`, `pmset -g therm`), not a guess. Kill the runner before defending it.
  Chrome is frequently the top burner, and the agent runtime itself has been the top burner —
  say so if it is true.
- **One model at a time.** Always.
- **When you are wrong, write the correction into `FINDINGS.md`.** That is valued more than a
  clean narrative. Several sections exist purely to undo earlier ones, and that is the point.
- **Do not invent results, pushes, or repos.** If you have not measured it, say "not measured".

---

## 12. The second project — do not mix it up with this one

There are **two separate efforts** on this machine and the owner has explicitly corrected a
previous agent for conflating them:

**A. Neuron surgery (this repo).** Steering Granite 4.1 3B at inference time to find love/evil
IBM selves. No training, no weight edits. Everything above.

**B. The model lab (on hold, not started).** Building a small model with a long buzzword stack —
distillation on small CoT/SFT datasets, SFT → DPO/SimPO → RLOO, abliteration, MoE-ification,
speculative decoding, long context. Target was **Nanbeige4.2-3B** (2.6 GB at 4-bit, ~63.6 on the
code bench used, 88 KiB/token cache). Supporting material already on disk: `datasets/` (~20 small
sets: s1K, LIMO, Magpie-Reasoning-V2-250K, cot-reasoning-2k, Open-CoT-Reasoning-Mini, NOESIS-1M,
Claude-Distills, DeepSeek-V4-Distill, gpt-5.6-luna-sft, Fable-5-traces, DiagSafe), plus
`LAB-MODEL.md`, `MODEL-PICK.md`, `OPTIMIZATION.md`, `RAG.md`, `PIPELINE.md`, `CONTEXT_1M.md`,
`DATASETS.md`, `tools` lists in `TOOLS.md`, and cloned jailbreak repos in `jailbreak-repos/`.

**B has no run in progress and no verified starting point.** If the owner asks about it, the
first action is to *list* `models/` and `datasets/` and report what is actually there — do not
assume the folder matches the last plan, because models were deleted and re-pulled several times.

If the owner asks "how do we get 1M context on 12 GB", the honest answer is already written in
`CONTEXT_1M.md`: a 1M-token KV cache on a 3B transformer at 4-bit is ~16 GB by itself and does
not fit; SSM/hybrid architectures have a fixed-size state but are not agentic coding models at
that size; RAG and chunking are the only things that actually fit. Do not re-derive this from
scratch and do not promise 1M on a 12 GB budget.

---

## 13. Your first message, concretely

1. Read `SUMMARY.md`, the last ~200 lines of `FINDINGS.md`, and `grid_L30_3h.jsonl` tail.
2. Confirm the state: no process alive, `STOP` present, thermal clean.
3. Report the L26–L30 lane in the format of §7 — rows, label mix, the dose→garbled curve, the
   persona-by-floor counts, the like-humans hedge rate, and the one odd row (L27 N2949).
4. Say plainly that coverage is **4.31% of the brain** and that the lane's persona counts are
   **not** comparable to the `Who are you?` layer curve because the probes differ.
5. Offer the three options from §9. Then stop.

No runs. No edits. No downloads. Wait for the owner.
