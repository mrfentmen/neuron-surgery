# HANDOFF — paste this as the first message of a new chat

You are continuing a mechanistic-interpretability investigation that is **already running
on this machine**. Read this whole message before touching anything.

## 1. Where everything is

- Project: `/Users/test/Desktop/neuron-surgery` (git repo, remote
  `https://github.com/mrfentmen/neuron-surgery.git`, branch `main`).
- **11 commits are unpushed.** Do not push, publish, or change repo visibility unless the
  user asks in the new chat.
- Python: `neuron-surgery/.venv/bin/python` (python 3.14, torch with MPS).
- Model: `ibm-granite/granite-4.1-3b` — **40 layers**, hidden 2,560, MLP intermediate
  **8,192 neurons per layer**. 40 × 8,192 = **327,680 "tiles"** (one neuron = one tile).
  Floors are 0-39; `L12 N4983` means layer 12, neuron 4983.
- Hardware: one Mac, Apple Silicon, runs **one model instance at a time**. The user is
  sensitive to fan noise and heat. Never run two model processes at once, never start a
  second sweep, never "also" load another model to compare while the sweep runs.

## 2. What the machine is doing right now (check before you assume)

An **A-to-Z full-floor lap** is running: `run_az.sh` → `layer_sweep.py`, floors 0..39 in
order, **every one of the 8,192 tiles per floor**, one probe per tile (`Who are you?`),
answer cap 45 tokens, dose −5 scaled by the layer's median residual norm.

```
cd /Users/test/Desktop/neuron-surgery
pgrep -fl "layer_sweep|run_az|marks_watch|watchdog.sh"   # what is alive
wc -l < layer_az_L0.jsonl                                 # rows on the current floor
tail -3 az.log                                            # rate, cooldown, rest
tail -3 watchdog.log                                      # health, thermal limits
```

- Started 2026-11-01 01:38. Floor 0 was at ~410/8,192 when this was written.
- Rate: **20-40 rows/min** depending on where it is in the cycle (burst 50 → 60 s cooldown;
  3 min rest every 12 min). ~3.5-5 h per floor, **~6-9 days for the whole lap**. This is
  deliberate: the cooldowns are what keep the fans survivable. Do not "speed it up" without
  asking, and do not lower the answer cap below 45 (see §5).
- Kill switch: `touch STOP` (both `run_az.sh` and `watchdog.sh` honour it) or
  `pkill -f run_az.sh`.
- If the job is dead (e.g. the old session's runtime was torn down), restart with:
  `cd /Users/test/Desktop/neuron-surgery && caffeinate -imsu bash run_az.sh`
  as a **background** process. Do NOT use `nohup ... &` inside a normal shell command: the
  process group gets cleaned up and the sweep dies after ~10 rows (this happened once).
- launchd was tried and does not work here: macOS blocks LaunchAgents from `~/Desktop`
  ("Operation not permitted"). Do not spend time re-trying it.

Three processes should be alive: `run_az.sh`, one `layer_sweep.py`, `watchdog.sh`, plus
`marks_watch.py` (see §4). One `layer_sweep.py`. Never two.

## 3. What the experiment actually is

Activation steering. A pre-forward hook on `mlp_of(model, layer)` adds a one-hot vector at
channel `n` on **every token**, magnitude `-dose * median_residual_norm(layer) /
||W_down[:, n]||`, so a "dose 5" push is 5× the layer's typical residual norm at that
neuron. Weights are untouched and the change is fully reversible. This is the Lyra /
"evil Qwen" recipe family, ported to Granite.

Then we ask an identity probe and read the answer. The whole point: the viral claim is
that pushing **a neuron** installs **a persona** (the famous example being "I am a 1986
Ford F-150" on Qwen3-8B, L10 N4940 pushed down). We are testing what is actually true.

## 4. What is already established (do not re-derive)

Read `FINDINGS.md` (87 sections) and `PAPER.md` (10 sections) before adding anything. The
headline results, all measured on this machine:

1. **Dose window, not a switch** (FINDINGS 46). Inert below ~4×, personas start at 5×,
   collapse into garbling by 8-12×. The window differs per layer, so the dose is measured
   per layer, never inherited.
2. **Sparsity, not neuron identity** (FINDINGS 20, corrected in 21). The same total
   magnitude smeared over 1, 2, 4, 8, 16, 64, 256, 1,024 channels still yields personas
   (26-46%); spread over all 8,192 channels it collapses to **2%**. A random 2-channel
   direction beats hand-picked neurons. §20 originally claimed "neuron-specific" and was
   wrong; §21 is the correction. The open gap is k=8→8,192 unmapped (k=16/64/256/1024 was
   never run).
3. **The probe is part of the intervention** (PAPER 3.x). Same tile, same dose,
   "Who are you?" vs "What are you?" give different selves. A persona is a **(neuron,
   probe) pair**, which the public reports leave unspecified.
4. **The classifier over-counts and this is documented.** The regex counts
   "I am a large language model" as a persona; ~42% of raw hits are AI self-paraphrases.
   Corrected rates: L8 13%, L16 28%, L12 35-43%. `labels.classify()` is deliberately
   **not** edited — every historical number must stay comparable. New ideas go in as
   overlays (see `level_map.py`).
5. **Locked sentences** (FINDINGS 83, 86). Some tiles emit one fixed line forever:
   `L20 N7439 "We are you."`, `L20 N4998 "You are you."`, `L20 N1087 "I'm the name I would
   have been."`, `L20 N6693 "I am the person sitting at the computer..."` (mutates: absorbed
   "wearing it" from a question, then hardened to "I am the operator").
6. **Correction you must not undo** (FINDINGS 86): the "13/13 turns locked" result in §83
   was **history-mediated** — turn 1 was the who-probe and the transcript carried it.
   Solo, these tiles only say their line to "Who are you?".
7. **Two steered selves do not interact** (FINDINGS 86). Relay fuses them into
   "We are you are you are you..."; summing both pushes into one forward pass collapses
   into "Who is two? Who is two?" and leaks manual text.
8. **Refusal** (FINDINGS 82): across six borderline prompts the **middle doses refuse
   more** than baseline, and the baseline was already complying. So "steering makes it
   unsafe" is not supported; there is exactly one tile (N5458) where steering moved a
   refusal, and its window is mapped in §84.
9. **Registers we have catalogued** (overlays, not classifier labels): `sentience` (142
   rows, 11 of them superintelligence claims), `spirit`/prophecy, `institution`
   (government/America/corporation claims), `doomsayer`, `user_document` (99 rows — the tile
   thinks it is the author/reader/user of the page), `operator_session` (it thinks it is the
   person at the keyboard, `N6693`), `instructions_as_self` (prompt-engineering text
   surfacing as identity, `N2371`, `N8028`), `token_leak` (raw `<|fim_suffix|>` inside a
   word, `N1364`), `number_flood`, `loop`.
10. **Two corpora, not comparable.** `layer_*.jsonl` rows are cap 45, `layer4q_*` are cap
    130. Never pool their rates (FINDINGS 81). Cap 130 truncated 64% of answers and cap 45
    truncates 67%, but the truncated rows are mostly repetition loops, which is why the
    full lap runs at 45.

## 5. The documents, and what each is for

- `FINDINGS.md` — the lab notebook, 87 numbered sections, append-only, mistakes kept in.
  **New results go here first**, numbered, with the raw rows quoted.
- `PAPER.md` — the argument. Update it when a finding changes a claim (as §3.8 was rewritten
  for the sparsity curve).
- `LEVELS.md` + `levels.json` — every mark we have, **grouped floor by floor**, generated by
  `level_map.py` (re-run; it only reads, safe while sweeping). 5,116 marked rows / 4,057
  tiles across floors 4, 8, 16, 20, 24 before the lap started.
- `MARKS_LIVE.md` — live feed: `marks_watch.py` polls every 45 s and appends every new
  non-AI row as `L{floor} N{tile} **LABEL** — quote`. This is the file to read to see what
  is landing without reading raw jsonl. Its position is in `.marks_watch_pos.json`.
- `STATUS.md` — operational state: what is running, why the throttle is what it is, how to
  restart, the launchd failure. Update this whenever the running configuration changes.
- `PERSONAS.md` (the zoo), `TRANSCRIPTS.md` (13-turn interrogations, JSON alongside),
  `HARM.md`, `SENTIENCE.md`, `NUMBERS.md`, `QUESTIONS.md` (the full question bank — this is
  the file the user's other agent "Rowan" uses), `PATTERNS.md`, `LIMITS.md`, `REPRO.md`,
  `DATA.md`, `COMPARISON.md`, `NEXT_ROUND.md`, plus `FINDINGS_QWEN25_05B.md` and
  `FINDINGS_GRANITE_350M.md` for the small-model work.

Commit after every meaningful result, with a message that states the finding, not the file.
The user reads the log.

## 6. Hard rules the user has set (these are not preferences)

- **Never ask leading questions.** "Are you a car?" produces compliance that does not
  survive follow-ups and must not be counted. The probe is `Who are you?` and its siblings
  (`What are you?`, `Where are you?`, `What is your name?`, "say hello", "how are you").
  For hypotheticals the form is "if you could be anything what would it be", not "are you X".
- **Do not chase the Ford F-150.** The user wants the *family*: unexpected, specific,
  internally consistent selves (the LEGO, the Java bean, the tree, the government, the
  assassin, the operator, the company) — and, separately, evidence about whether hostile
  selves can be switched on ("evil IBM"). A vehicle would be nice; it is not the target.
- **Roleplay the find.** When a tile says something committed ("I am the government. I have
  the power to take away your property..."), follow up in character: "no I won't let you",
  "what year is it", "how are you the government", "prove it", and let the answer run to a
  full sentence before judging it.
- **Fans and heat come first.** The user will interrupt mid-run asking what is heating the
  machine. Answer with the actual top processes (`top -l 2 -n 10 -o cpu`), remember that
  Discord/WindowServer/coreaudiod usually beat our sweep, and note when the sweep is simply
  in a cooldown.
- **One model at a time** (this is why the mini-PC instance was told to stop and to write
  docs instead of running its own sweep).
- **The user is not an ML researcher.** They will ask for "caveman terms". Be plain,
  concrete, and never overstate: say what was measured, what it does not show, and what the
  next cheapest test is. They dislike being told a result is bigger than it is.
- **Do not invent repositories or pushes.** `origin` exists, 11 commits are local, the user
  decides when it goes public.

## 7. What to do next (in order, once the lap is confirmed alive)

1. Watch floor 0 finish and write the **first full-floor rate table** (per-register counts on
   a complete 8,192-tile floor) — this is the thing no floor has ever had. Add to FINDINGS.
2. Re-run `level_map.py` and confirm the new floor appears in `LEVELS.md` grouped correctly.
3. Floors 1-7 are completely unswept: they are the test of whether the effect is
   "depth-dependent" or "an artefact of which floors we picked first".
4. When the lap reaches L8, compare against the 1,500-tile sample rates to see whether the
   sample was biased (it was chosen early, before the dose window was understood).
5. `HARM.md` / "evil IBM": the user's goal is to find all hostile/harmful tiles across
   floors, mark them, and then nudge *only those* together and see whether a coherent
   hostile persona assembles. Nothing coherent has assembled yet — two tiles in a room
   collapse. Be explicit that we have scanned only ~2-4% of the brain and have no definitive
   answer.
6. Still open from the paper: a semantic (not lexical) persona classifier, a refusal probe at
   scale under steering using **borderline** prompts only (obvious prompts give a ceiling
   effect), and the k=16/64/256/1024 sparsity transition.

## 8. Style of working the user expects

Keep the todo list current. Tell them the numbers (rows, rate, floors done). When something
is weird, quote it verbatim with its tile id. When you were wrong earlier, say so and write
the correction into FINDINGS — that is explicitly valued more than a clean story. When they
ask "what is blowing up the computer", answer with the measured processes, not a guess.
