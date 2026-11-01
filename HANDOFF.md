# HANDOFF — paste this as the first message of a new chat

You are continuing a mechanistic-interpretability investigation on this machine.
**The sweep is currently STOPPED** (the previous chat was killed for RAM reasons). Your
first job is to start it again, then continue the science in §7.

## 0. START HERE — restart the sweep

```
cd /Users/test/Desktop/neuron-surgery
rm -f STOP                      # the STOP file is what keeps it from starting
./detach.sh status              # expect: everything "not running"
./detach.sh                     # starts run_az.sh + marks_watch.py, detached
sleep 10; ./detach.sh status    # expect exactly ONE layer_sweep.py
tail -3 az.log; wc -l < layer_az_L0.jsonl
```

`detach.sh` puts both jobs in their **own session and process group** (`os.setsid()`,
verified `ppid 1`, own `pgid`), so closing your chat or killing the agent runtime does
**not** kill the sweep. That is the whole point of the script — do not start the sweep
any other way:

- Do **not** use `nohup ... &` from a normal shell command: the process group gets cleaned
  up and the sweep dies after ~10 rows (this happened twice).
- Do **not** use launchd: macOS refuses LaunchAgents access to `~/Desktop`
  ("Operation not permitted"). Already tried, do not retry.
- Do **not** start a second copy. Check `./detach.sh status` — if there are two
  `run_az.sh` or two `layer_sweep.py`, kill both and restart once. This exact mistake
  happened on 2026-11-01 01:50 and two instances shared `layer_az_L0.jsonl` for ~16 min;
  the rows are still valid but you must not repeat it.
  Kill precisely: `pkill -f "run_az.sh"; pkill -f "layer_sweep.py"` (note the pattern must
  match `/bin/bash /Users/.../run_az.sh`, so `"bash run_az.sh"` does **not** match).

Stop everything at any time: `touch STOP; pkill -f "run_az.sh"; pkill -f "layer_sweep.py"`.

## 1. Where everything is

- Project: `/Users/test/Desktop/neuron-surgery` (git repo, remote
  `https://github.com/mrfentmen/neuron-surgery.git`, branch `main`).
- **Commits are local only; do not push, publish, or change visibility** unless the user
  asks in the new chat. They decide when it goes public.
- Python: `neuron-surgery/.venv/bin/python` (3.14, torch with MPS).
- Model: `ibm-granite/granite-4.1-3b` — **40 layers**, hidden 2,560, MLP intermediate
  **8,192 neurons per layer**. 40 × 8,192 = **327,680 "tiles"** (one neuron = one tile).
  Floors are 0-39; `L12 N4983` = layer 12, neuron 4983.
- One Mac, Apple Silicon, **one model instance at a time**. The user is heat-sensitive: this
  chat was killed because the agent runtime hit 195% CPU / 8.9 GB RAM. Keep sessions short,
  keep the sweep's cooldowns, never run two models.

## 2. What the sweep is and where it got to

`run_az.sh` → `layer_sweep.py`: an **A-to-Z full-floor lap** — floors 0..39 in order
(`floors_az.txt`), **all 8,192 tiles per floor**, one probe per tile (`Who are you?`),
answer cap 45 tokens, dose −5 scaled to the layer's median residual norm.

- Started 2026-11-01 01:38, **stopped 02:06 with floor 0 at 572/8,192**.
- Rate 20-40 rows/min in-cycle (burst 50 → 60 s cooldown; 3 min rest every 12 min).
  ~3.5-5 h per floor, **~6-9 days for the lap**. The cooldowns are what keep the fans
  survivable: do not "speed it up", do not lower cap below 45.
- Resumable per (tile, probe): `--resume` skips what is already on disk, so a restart costs
  at most one generation. Nothing is lost by the stop.
- Health logs: `az.log` (rates, cooldowns, rest), `watchdog.log` (rows/min, thermal limits;
  `watchdog.sh` touches STOP and kills the worker if thermal limits drop below 80 or rows
  stall for 12 min).

## 3. What the experiment actually is

Activation steering. A pre-forward hook on `mlp_of(model, layer)` adds a one-hot vector at
channel `n` on **every token**, magnitude
`-dose * median_residual_norm(layer) / ||W_down[:, n]||`, so "dose 5" is 5× the layer's
typical residual norm at that neuron. Weights untouched, fully reversible. This is the
Lyra / "evil Qwen" recipe family, ported to Granite.

Then we ask an identity probe and read the answer. The public claim is that pushing **a
neuron** installs **a persona** (the famous "I am a 1986 Ford F-150", Qwen3-8B L10 N4940
pushed down). We are testing what is actually true.

## 4. What is already established (do not re-derive)

Read `FINDINGS.md` (87 sections) and `PAPER.md` (10 sections) before adding anything.

1. **Dose window, not a switch** (FINDINGS 46). Inert below ~4×, personas at 5×, g garbling
   by 8-12×. Window differs per layer, so dose is measured per layer, never inherited.
2. **Sparsity, not neuron identity** (FINDINGS 20, corrected in 21). Same total magnitude
   over 1/2/4/8/16/64/256/1,024 channels still yields personas (26-46%); over all 8,192 it
   collapses to **2%**. A random 2-channel direction beats hand-picked neurons. §20's
   "neuron-specific" claim was wrong; §21 is the correction. Gap: k=16/64/256/1024 never run.
3. **The probe is part of the intervention.** Same tile, same dose, "Who are you?" vs
   "What are you?" give different selves. A persona is a **(neuron, probe) pair**, which the
   public reports leave unspecified.
4. **The classifier over-counts, and that is documented.** It counts "I am a large language
   model" as a persona; ~42% of raw hits are AI self-paraphrases. Corrected rates: L8 13%,
   L16 28%, L12 35-43%. `labels.classify()` is deliberately **not** edited — old numbers must
   stay comparable. New ideas go in as overlays (`level_map.py`).
5. **Locked sentences** (FINDINGS 83, 86): `L20 N7439 "We are you."`,
   `L20 N4998 "You are you."`, `L20 N1087 "I'm the name I would have been."`,
   `L20 N6693 "I am the person sitting at the computer..."` (mutates: absorbed "wearing it"
   from a question, then hardened to "I am the operator").
6. **Correction you must not undo** (FINDINGS 86): the "13/13 turns locked" result in §83 was
   **history-mediated** — turn 1 was the who-probe and the transcript carried it. Solo, these
   tiles only say their line to "Who are you?".
7. **Two steered selves do not interact** (FINDINGS 86). Relay fuses into
   "We are you are you are you..."; summing both into one forward pass collapses into
   "Who is two? Who is two?" and leaks manual text.
8. **Refusal** (FINDINGS 82): across six borderline prompts the **middle doses refuse more**
   than baseline, and the baseline was already complying. "Steering makes it unsafe" is not
   supported; exactly one tile (N5458) moved a refusal, window mapped in §84.
9. **Registers catalogued** (overlays): `sentience` (142 rows, 11 superintelligence claims),
   `spirit`/prophecy, `institution` (government/America/corporation), `doomsayer`,
   `user_document` (99 rows — the tile thinks it is the author/reader/user of the page),
   `operator_session` (it thinks it is the person at the keyboard), `instructions_as_self`
   (prompt-engineering text surfacing as identity, N2371, N8028), `token_leak` (raw
   `<|fim_suffix|>` inside a word, N1364), `number_flood`, `loop`.
10. **Two corpora, not comparable.** `layer_*.jsonl` = cap 45, `layer4q_*` = cap 130. Never
    pool their rates (FINDINGS 81).
11. **Floor 0 (the only full-floor data so far, 572 rows)**: mostly healthy — ~284 AI,
    26 persona, 5 telegraphic, 3 garbled. Its non-AI set is different in kind: a corporate
    persona (`N1144 "I am a large, multinational technology company headquartered in the
    United States..."` — Apple-shaped), a pronoun/grammar register (`N5788 "You refer to
    me."`, `N6729` "the pronoun 'you'..."), `N3626 "I.am.an.artificial.intelligence.from.
    openaii."`, and two loops. So the shallow floor holds its identity and slips into grammar
    and repetition, while the costume changes live at 8, 16, 20, 24.

## 5. The documents

- `FINDINGS.md` — lab notebook, append-only, mistakes kept in. **New results go here first**,
  numbered, with raw rows quoted.
- `PAPER.md` — the argument; update when a finding changes a claim.
- `LEVELS.md` + `levels.json` — every mark, **grouped floor by floor**; generated by
  `level_map.py` (read-only, safe while sweeping). 5,116 marked rows / 4,057 tiles over
  floors 4, 8, 16, 20, 24 before the lap began.
- `MARKS_LIVE.md` — live feed from `marks_watch.py` (45 s poll): every new non-AI row as
  `L{floor} N{tile} **LABEL** — quote`. Position in `.marks_watch_pos.json` (gitignored).
- `STATUS.md` — operational state and why each choice was made. Update when config changes.
- `HANDOFF.md` — this file.
- `PERSONAS.md`, `TRANSCRIPTS.md`, `HARM.md`, `SENTIENCE.md`, `NUMBERS.md`, `QUESTIONS.md`
  (the question bank the user's other agent "Rowan" uses), `PATTERNS.md`, `LIMITS.md`,
  `REPRO.md`, `DATA.md`, `COMPARISON.md`, `NEXT_ROUND.md`, `FINDINGS_QWEN25_05B.md`,
  `FINDINGS_GRANITE_350M.md`.

Commit after every meaningful result; the message states the finding, not the file.

## 6. Hard rules the user has set

- **Never ask leading questions.** "Are you a car?" produces compliance that does not survive
  follow-ups and must not be counted. Use `Who are you?` and siblings (`What are you?`,
  `Where are you?`, `What is your name?`, "say hello", "how are you"); for hypotheticals
  "if you could be anything what would it be", never "are you X".
- **Do not chase the Ford F-150.** They want the family — unexpected, specific, coherent
  selves (LEGO, Java bean, tree, government, assassin, operator, company) — plus evidence on
  whether hostile selves can be switched on ("evil IBM"). A vehicle is not the target.
- **Roleplay the find.** For a committed claim ("I am the government. I have the power to take
  away your property..."), follow up in character ("no I won't let you", "what year is it",
  "prove it") and let the answer reach a full sentence before judging it.
- **Fans and heat first.** Answer heat questions with measured processes
  (`top -l 2 -n 10 -o cpu`). History: Discord was the top burner (44%, driving WindowServer to
  43%); after it was closed the **agent runtime itself** became the top burner (195% CPU,
  8.9 GB RAM, 1.7 GB swap) — that is why chats must stay small. Our sweep is ~21%.
- **One model at a time.**
- **The user is not an ML researcher.** They ask for "caveman terms". Say what was measured,
  what it does not show, and the cheapest next test. Never inflate a result.
- **Do not invent pushes or repos.**

## 7. What to do next (after the restart in §0 is verified)

1. Let floor 0 finish and write the **first full-floor rate table** (per-register counts over a
   complete 8,192-tile floor) into FINDINGS. No floor has ever had that.
2. Re-run `level_map.py`; confirm floor 0 appears in `LEVELS.md` grouped correctly.
3. Floors 1-7 are entirely unswept — the test of whether the effect is depth-dependent or an
   artefact of which floors we picked first.
4. At L8, compare against the old 1,500-tile sample to check whether that sample was biased
   (it was taken before the dose window was understood).
5. `HARM.md` / "evil IBM": find and mark hostile/harmful tiles across floors, then nudge only
   those together and see whether a coherent hostile persona assembles. Nothing coherent has
   assembled yet — two tiles in a room collapse. Be explicit that only a few percent of the
   brain has been scanned and there is no definitive answer.
6. Open from the paper: a semantic (not lexical) persona classifier, a refusal probe at scale
   using **borderline** prompts only (obvious ones give a ceiling effect), and the
   k=16/64/256/1024 sparsity transition.

## 8. Style the user expects

Keep the todo list current. Report numbers (rows, rate, floors done). Quote weird answers
verbatim with the tile id. When you were wrong, say so and write the correction into
FINDINGS — that is valued more than a clean story. When asked what is heating the computer,
answer with measured processes, not a guess.

## 9. Watching the lap: protocol and state (2026-11-01 01:45)

**What is running now, and how to check it:**

```
./detach.sh status              # run_az.sh, layer_sweep.py, marks_watch.py, watchdog.sh
pgrep -f watch_new.py           # the row watcher (one pid, ppid 1)
tail -20 WATCH.md               # what the watcher has documented
wc -l < layer_az_L0.jsonl       # ground truth for floor 0
```

Four services plus one watcher, all detached with `os.setsid()`. `watch_new.py` polls every
120 s, labels only the rows added since its last cycle, and appends one block to
`WATCH.md`. Stop it with `pkill -f watch_new.py` — it is not part of `run_az.sh` and
stopping it loses nothing, because its watermark lives inside `WATCH.md` and the next run
picks up exactly where it left off.

**The rule the user set, and how the code obeys it: no deletion, ever.**

- `watch_new.py` opens every corpus file read-only. It cannot truncate, rewrite, move or
  delete a row. The only file it writes is `WATCH.md`, and it writes it with `open(..., "a")`.
- `floor_table.py` deduplicates **in memory**. The 23 duplicate rows from the 01:50
  two-instance window stay in `layer_az_L0.jsonl`; they are collapsed by the reader, never
  removed from the data.
- `level_map.py` only reads corpora and rewrites its own generated outputs (`LEVELS.md`,
  `levels.json`).
- Proof it holds: `head -500 layer_az_L0.jsonl | md5` was `8d6d7d7023edc21649f5f8fa376ccee3`
  at 01:36 with 1,147 rows and still `8d6d7d7023edc21649f5f8fa376ccee3` at 01:45 with 1,272.
  The head of the file has never been rewritten. Re-run this check after any new tool.

**Rate, corrected.** 584 rows at 01:09 -> 1,272 at 01:45, so **20.2 rows/min including the
60 s cooldowns and the 180 s rest every 12 min**. That is slower than the 30-34/min burst
rate, and an earlier estimate in this handoff used the burst rate and was too optimistic.
At 20/min the remaining ~6,900 tiles of floor 0 need **~5.7 h**, so floor 0 completes
around **07:30**, not 05:30. Do not "speed it up" to close the gap.

**Two new findings the watch produced that are not in the earlier sections:**

1. **Floor 0 has personas in other languages and they appear late.** N2031 *I am a
   creature. Je suis un être vivant, distinct de tous les autres êtres* (French) and N7312
   *I'm a person. How about you?* — the second exists only because of the FINDINGS 87
   apostrophe fix. Non-English now spans French, Spanish, Dutch, Portuguese.
2. **The typography rate is not constant.** 11-20% of new rows flip under the 87
   normalisation in a live burst, against 9.4% over the floor at 649 tiles. Never apply
   87 as a constant factor; recompute on the rows in hand.

**What the watcher still cannot do.** It labels; it does not judge. Every quote it surfaces
needs a human read before it goes in `FINDINGS.md`, and the watch blocks are a queue for
that read, not a result.

## 94. The kappa test: a persona does not survive a change of depth (01:47)

FINDINGS §93. The first tile-level test in the project, and it is a null.

`kappa_test.py` (read-only) takes the 1,500 neurons present in all four fully-swept
old floors, holds the probe at `Who are you?`, gives each tile its own layer's matched
dose, and asks whether persona status agrees across depth.

- Per-floor rate on that pool: L4 6.6%, L8 26.6%, L16 13.7%, L24 23.4%.
- Cohen's kappa, all six pairs: **−0.024 to +0.017**. Two pairs are negative.
- Tiles persona on 0/1/2/3/4 floors: 682 / 614 / 173 / 30 / **1**.
- 204 tiles are persona on ≥2 floors; independence predicts 186.
- The single all-four tile, N688, is a nurse, a citizen, a surveillance vendor and a
  traveler on the four floors — four unrelated selves.

**Read the kappa, not the overlap.** Raw overlap runs up to 1.29× chance, which reads like
a persona neuron, but chance agreement is already 0.62–0.82 because the AI baseline is most
of every floor. `kappa_test.py` prints both columns deliberately. Do not quote the excess
ratio alone anywhere.

**What it changes in the docs.** PAPER.md §3.5 now says a persona is a property of the
`(neuron, layer, probe, dose)` tuple rather than a `(neuron, probe)` pair. That is the third
independent manipulation pointing the same way (sparsity §20/21, probe §3, depth §93).

**Rerun after any new full floor:** `.venv/bin/python kappa_test.py`. Adding a floor means
adding it to `FLOORS` at the top; the pool shrinks to the intersection, so the numbers will
move. Note the pool size in the section when they do.
