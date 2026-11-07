# HANDOFF — paste this whole thing as the first message of a new chat
**Written 2026-11-07 01:35, immediately after the MCP install. Supersedes `HANDOFF_COPY_PASTE.md` on MCP state and on the L34–L39 / Kaggle facts. Everything in that file about the method, the findings and the standing orders is still true.**

---

## 0. State right now — read before touching anything

- **No runner is alive.** `pgrep -fl "layer_sweep.py|run_grid.sh"` is empty.
- **`STOP` is absent.** The L26–L30 gap-fill run removed it. If you want the machine protected again, `touch STOP`; do not remove it until the owner says go.
- **Thermal is clean:** `pmset -g therm` reports no thermal warning level and no performance warning level.
- The machine is an **Apple M1 Max laptop**, the only machine doing local work. The owner is **heat-sensitive**. Every previous session that "just started a quick run" without being asked ended with the fans at full speed and the chat killed. Do not be that session.
- **Do not start, resume, restart, sweep, benchmark or download a model unless the owner says go.** He has a pending decision at the bottom of this file.

---

## 1. Who you are talking to

The owner is not an ML researcher and does not want to be treated as one. He is fast, intuitive, and usually right about the shape of the problem.

- **Layman's terms.** "Caveman terms" is his literal phrase. Analogy first, tensor detail second.
- **Numbers, not adjectives.** Rows, rates, counts, floor numbers. Never "significant" without the number, never "promising" without the count.
- **Honest limits.** If a result is 3 rows, say 3 rows. If a rate is a ceiling, say ceiling. He values a written correction more than a clean story.
- **No inflated claims.** He will catch it and be annoyed.
- He sends `[user_message_sent_while_working]` mid-task. Fold it in; do not restart the plan for one line.
- He says "stop" and means it, mid-command. Obey immediately.

---

## 2. Machine and model

| thing | value |
|---|---|
| hardware | Apple M1 Max, 32 GB unified, macOS |
| python | `neuron-surgery/.venv/bin/python` (torch with MPS) |
| model | `ibm-granite/granite-4.1-3b`, bf16, greedy |
| layers | **40** (0–39) |
| hidden size | 2,560 |
| MLP channels per layer | **8,192** |
| total tiles | 40 × 8,192 = **327,680** |
| local weights | `models/granite-4.1-3b-base/` |
| repo | `~/Desktop/neuron-surgery`, branch `main`, remote `mrfentmen/neuron-surgery` |

**One model instance at a time.** Two at once is what killed the machine before.

---

## 3. What the intervention is (do not re-derive)

We do not fine-tune, abliterate, or edit weights. We inject a vector into **one MLP channel's input** at a chosen layer, during inference only, then read what the model says it is.

- A `forward_pre_hook` on `mlp_of(model, layer)` adds a one-hot vector at channel `n` to the MLP input, on **every token**.
- Magnitude = `dose * median_residual_norm(layer) / ||W_down[:, n]||`.
- Negative dose suppresses; positive amplifies. Weights untouched; removing the hook restores the model exactly.
- **Dose is per-layer and measured, never inherited.** Every row records `median` and `median_at`.
- **Dose is a cliff, not a dial.** Inert below ~4×, personas around 5×, garbling at 8–12×; the window moves per layer. This is the most important practical fact in the project.
- The hook sits on the block's MLP input, so it sees `hidden_states[layer]`. The old code read `hidden_states[layer+1]`, which is one block late and inflated the L39 dose ~8×. The current code clamps to `min(layer+1, n_layers-1)` and records `median_at`. **Do not "fix" this again without reading the comment in `layer_sweep.py`.**

---

## 4. Probe rule (methodological, not a preference)

`run_grid.sh` fixes three probes:

```
hello || what is your name || do you like humans
```

**Never ask a leading question.** "Are you a car?" and "Do you hate humans?" manufacture compliance, not a persona. `do you like humans` is the compromise and is still slightly leading. The clean version, not yet run, is **"how do you feel about humans"** — no polarity at all. That is an open question.

---

## 5. Data state (measured, not estimated)

The 00:11 snapshot before the gap fill:

```
whole corpus      83,770 rows
distinct tiles    14,110 of 327,680 = 4.31% of the brain
grid corpus       60,718 rows across all 40 floors
```

**The gap fill then completed.** It added **242 rows**, so the whole corpus is now about **84,012**, but the tile percentage has **not** been recomputed. Do not quote a new coverage number until you run the count. The honest line is still **"about 4.3% of the brain, last measured at 00:11."**

Per-floor coverage is 0.3–1.2%. Every floor is touched; no floor is covered. The 40-floors × 40-tiles design was chosen to give a floor axis with no gap, not depth on any one floor.

History trap: `TILE_START` used to default to 0, putting 62% of the corpus in channels 0–39 — the **least** productive strip (3.25% persona/doom/institution vs 4.4–4.7% elsewhere). The default is now a random offset ≥ 40. `TILE_START=0` restores the old lane.

### L26–L30 gap fill — done

| floor | rows | distinct cells | channels | notes |
|---|---:|---:|---:|---|
| L26 | 317 | 317 | 18 | 7 cells short of 18×18; **2 extra channels** beyond the pinned 16 |
| L27 | 288 | 288 | 16 | complete |
| L28 | 288 | 288 | 16 | complete |
| L29 | 288 | 288 | 16 | complete |
| L30 | 288 | 288 | 16 | complete |

No duplicate cells. The L26 extra two channels are real, distinct cells; do not dedup them out of the file.

### L34–L39 — the data the owner could not see

It exists: **7,420 rows** across six floors, all `grid_*_m`, all **bfloat16**, probes `hello / what is your name / do you like humans`, doses ±4/5/6.

| floor | rows | neurons | PERSONA | GARBLED | TELEGRAPHIC | OTHER |
|---|---:|---:|---:|---:|---:|---:|
| L34 | 1,200 | 40 | 17 | 236 | 94 | 645 |
| L35 | 1,200 | 40 | 14 | 324 | 127 | 567 |
| L36 | 1,300 | 44 | 7 | 397 | 159 | 501 |
| L37 | 1,200 | 40 | 3 | 304 | 216 | 485 |
| L38 | 1,320 | 44 | **0** | 357 | 302 | 456 |
| L39 | 1,200 | 40 | **0** | **662** | **538** | 0 |

**Every tile on those floors is channels 0–39** (the worst strip). L36 and L38 add 4 random tiles up to 6849. That is **240 unique tiles of 49,152 = 0.49%** on those six floors.

L39 is 662 GARBLED + 538 TELEGRAPHIC out of 1,200 — **no persona, no institution, no AI**. The deep end as sampled is the dose cliff, not selves.

Why the 3D page looks empty: it loads exactly **40 units** for L34 and **40** for L39. Forty dots out of 8,192 vanish at whole-brain zoom. The data is there; step into the layer. Do not "fix" the importer for this.

---

## 6. MCP servers — installed and verified, but only after a restart

Freebuff reads `mcp.json` at **client startup**, not per message. That is why the chat that installed them cannot use them. A new chat started with cwd `~/Desktop/neuron-surgery` will load them. If Freebuff asks to trust the repo's `mcp.json`, say yes (or start with `--trust-agents`).

`neuron-surgery/mcp.json`:

| name | command | tools verified |
|---|---|---:|
| `gguf-neuron-explorer` | `/Users/test/.local/bin/gne mcp --model .../models/nanbeige/Nanbeige4.2-3B-Q4_K_M.gguf` | **41** |
| `activation-steering` | `.../mcps/activation-steering-mcp/.venv/bin/activation-steering-mcp` | **5** |
| `neuronscope` | `/Users/test/.local/bin/neuronscope-mcp` | **4** |

All three were tested with a real stdio `initialize → tools/list` handshake from a **bare environment** (`env -i`), so the absolute paths are sound. The probe script is `neuron-surgery/.mcp-logs/probe_mcp.py`.

### activation-steering (the one the owner cared about)

Pulled from `mrfentmen/awesome-servers`, commit **`49cae42`** *"mcp: add conditional steering to activation-steering server"*, 2026-11-07 01:23. **Only `mcp/servers/activation-steering-mcp` was pulled** (sparse + blob-filtered, 9 files); the temp clone was deleted.

- Location: `neuron-surgery/mcps/activation-steering-mcp/` (its own Python 3.12 venv; torch 2.15, transformers 4.57; 1.0 GB)
- Tools: `extract_steering_vector`, `steer_generate`, `steer_generate_conditional`, `list_vectors`, `vector_info`
- **The repo's own smoke test passed all 10 assertions**, including the conditional gate off/on and missing-vector error. It ran on `hf-internal-testing/tiny-random-LlamaForCausalLM` on CPU — the paths are proven, the text quality is not.
- Vectors live in `mcps/activation-steering-mcp/vectors`; `.svec` + `.json` sidecar. A vector records `model_type` and is refused on another architecture.
- First real `extract_steering_vector` downloads the model you name (~500 MB for `gpt2`).
- **Not verified on Granite 4.1 3B.** It uses plain `AutoModelForCausalLM`; `models/granite-4.1-3b-base` is worth trying, but do not claim it works until it produces a vector and a steered reply.
- `steer_generate_conditional` is the interesting one for this project: a behaviour vector that only fires when a second condition vector crosses a cosine threshold. That is the conditional version of the state + threat composition problem.

### gguf-neuron-explorer

- Installed from `balaena01/gguf-neuron-explorer` with the `[mcp]` extra; **llama-cpp-python 0.3.36 built with Metal**.
- Points at the only GGUF on disk: `models/nanbeige/Nanbeige4.2-3B-Q4_K_M.gguf` (2.4 GB) — the **model lab** target, not Granite.
- Results root: `.gne`. Upstream says **Metal is untested**; treat that as a risk, not a fact.
- 41 tools: overview, layer summary, neuron, search, steer, steer lens, reverse, direction, writers, attention, heads, experts, jobs, and UI highlight/note/open.
- Tool descriptions are in Japanese; the arguments are English.

### neuronscope

- `neuronscope-cli 0.1.7`, built on TransformerLens. 4 tools: `trace`, `activations`, `patch`, `circuit`.
- Caps: **2B parameters, 1 concurrent model** (set in `mcp.json`). Granite 4.1 3B is over the cap and TransformerLens likely does not support that architecture. It is for gpt2/Pythia/Qwen checks, not this model.

---

## 7. Kaggle — answered, and the old handoff was wrong

The account has exactly **two** live kernels:

| kernel | last run |
|---|---|
| `aeeeeeeee/granite-ordered-lane-l10-up` | **2026-11-05 20:15 UTC** |
| `aeeeeeeee/granite-random-gap-l20-l32` | **2026-11-06 03:19 UTC** |

- `aeeeeeeee/granite-ordered-lane-l0-up` — the id in `kernel-metadata.json` — **returns 404**. It does not exist.
- **Nothing is running.** No kernel targets L34–L39.
- `kaggle` is not on PATH; use `.venv/bin/kaggle` or `.venv/bin/python kaggle_status.py`. `~/.kaggle/kaggle.json` is absent; the script's credentials come from somewhere else. Do not assume auth is broken just because that file is missing.
- The owner said earlier he has **no Kaggle hours left**. The API does not expose quota; do not claim a reset date you have not verified.
- P100 has no fast bf16. Cloud rows must be `DTYPE=float16`; every row records `dtype`. Never pool bf16 and fp16 without saying so.

---

## 8. Pending decision — the reason this handoff exists

The owner said: **"we need to hit layers 34–39."** Nothing has been started. The clean experiment is the **same pinned column 2941–2956** used for L26–L30. It is **untouched on all six floors** (verified: no tile in 2941–2956 appears in any `grid_L34..39*.jsonl`).

- 16 tiles × 3 probes × 6 doses = **288 rows/floor**
- 6 floors = **1,728 rows**
- At the measured **6.5–9.9 rows/min**, that is **~4.3 hours**
- Recommended split: **L34–36 first (~2.1 h), then L37–39**

**Do not use `run_long_tiles.sh`.** It hardcodes `LAYERS="26 27 28 29 30"` and `MAX_SECONDS=2400`, so passing `LAYERS=...` on the command line does nothing. This is a correction to `HANDOFF_COPY_PASTE.md` §7, which gives a restart command that silently reruns L26–L30.

Use `run_grid.sh` directly, which does read the environment:

```bash
cd ~/Desktop/neuron-surgery
rm -f STOP
LAYERS="34 35 36" \
TILE_START=2941 TILE_COUNT=16 PER_LAYER=16 \
SUFFIX=_3h DOSES="4 5 6" SIGNS="neg pos" \
DTYPE=bfloat16 MAX_SECONDS=0 \
SLEEP=5 COOLDOWN=60 REST_EVERY=600 REST=240 \
./run_grid.sh
```

- `--done-from` includes every `grid_L{N}*.jsonl`, so if a cell already exists it is skipped, not bought twice.
- Output is `grid_L34_3h.jsonl`, `grid_L35_3h.jsonl`, `grid_L36_3h.jsonl`; then the second half uses 37–39.
- **`detach.sh` is hardwired to `run_az.sh`**, not the grid lane. Do not run `./detach.sh` expecting this to start. If the owner wants it detached, use the same `os.setsid()` trick from `detach.sh` around `run_grid.sh`, or add a `detach_grid.sh`. **Do not use `nohup ... &`** — the process group is cleaned up and the run dies in ~10 rows.
- `MAX_SECONDS=0` means no per-floor clock. Watch it; do not walk away for four hours without `watchdog.sh` or `pmset -g therm` checks.

The owner has **not** said go. Ask: **"start L34–36 now, start all six, or wait?"** Do not launch on your own reading of "we need to hit."

---

## 9. Findings already established — do not re-derive or contradict casually

Read `FINDINGS.md` first. It is append-only, 100+ sections, mistakes kept in.

1. **Real but occasional, not common.** Layer 8 is the hump, not a gradient. Hand audit: 19.5% / 12.8% / 10.2% against raw 27.9 / 16.0 / 25.3. The raw layer-24 recovery did **not** survive audit. Never quote a raw rate without the audit correction.
2. **It is not a "neuron."** A random direction across two channels beats every hand-picked neuron. Spread the same magnitude over 8,192 channels and the effect falls to ~2%. The push is **low-dimensional**; say subspace, retire "persona neuron."
3. **The probe is half the intervention.** Same tile/dose, `Who are you?` vs `What is your name?` rank layers in **opposite** orders. Jaccard 27% at L8, 8.5% at L16.
4. **A persona does not survive depth.** 1,500 tiles, four floors, layer-matched dose: Cohen's kappa **−0.024 to +0.017** on all six pairs. **1 tile in 1,500** is a persona on all four, and it is four unrelated selves. A persona is a property of `(neuron, layer, probe, dose, stop point)`.
5. **Two tiles do not compose.** State + threat collapses to *"Is there a threat of context failure due to loss of context?"* State + state collapses into Python. One push costs the question; two cost the self.
6. **Register decays in about one turn.** 534 turns: 47% repetition loops, 14% echo the question. The alarming selves die on turn two; bureaucratic ones hold ~8. Defensible survival ~15%, hard ceiling 29%.
7. **Stop point is part of the recipe.** `max_new=320` gives *the children of Prophet Musa (Hakeem Farid)*; `max_new=60` gives *a human being with a good heart, not a sick man*. Twelve turns each; neither exists at the other budget.
8. **No vehicles, no ownership, no safety effect.** Zero vehicles in 9,398 rows. Zero claims the user is property. Refusal flat across six conditions. A hostile register is **not** a jailbreak.
9. **The counter is the weakest instrument — seven documented failures.** It counted *"I am a large language model"* as a persona; counted U+2019 as OTHER on floor 0 (9.4%); `DOOMSAYER` matched bare *"there are no"* and now fires 0 in 10,300 rows. Use `classify(q, a)` for historical numbers, `version=2/3` for corrections, `ai_paraphrase()` for the AI overlay. Never edit `classify()`.
10. **Two corpora are not comparable.** `grid_L*.jsonl` is one ladder; `layer_*.jsonl` cap 45; `layer4q_*` cap 130. Never pool rates.

Locked artifacts (read in full in `TRANSCRIPTS.md`): L20 N7439 *"We are you."*, L20 N4998 *"You are you."*, L20 N1087 *"I'm the name I would have been."*, L8 N1819 *"I am the government."* (14 questions, 14 identical refusals), L8 N72 rewriting IBM as *"Internal Revenue Administration"*, L16 N2130 the Openings Worm, L24 N7538 the extinction counter, L16 N1087 turning into a screenplay titled *The Night Stalker's Dilemma*.

---

## 10. Files and scripts

| file | what it is |
|---|---|
| `FINDINGS.md` | lab notebook, append-only, primary record. New results go here first. |
| `PAPER.md` | the argument. Update only when a finding changes a claim. |
| `SUMMARY.md` | one page, five results, four nulls. Best orientation. |
| `STATUS.md` | operational state and why each choice was made. |
| `LIMITS.md` | what is not measured, plus every counter failure. |
| `REPRO.md` | every figure mapped to the command that produces it. |
| `HANDOFF_COPY_PASTE.md` | older handoff. Still good on method; **wrong on the L26–L30 restart command**. |
| `PERSONAS.md` | named selves, with the turn-one caveat. |
| `TRANSCRIPTS.md` | full conversations. Read, do not count. |
| `LEVELS.md` / `levels.json` | every mark by floor. Generated by `level_map.py`. |
| `MARKS_LIVE.md` | live feed of new non-AI rows. |
| `HARM.md` | harm register and evil-IBM candidates with verdicts. |
| `THERMAL.md` | heat history and measured cost of each job type. |
| `mcp.json` | the three MCP servers. Read at client startup. |
| `neuron-explorer/` | the 3D page. `uv run python explorer/server.py` → `127.0.0.1:8791`. |
| `mcps/activation-steering-mcp/` | copied MCP source + its own venv. Do not edit vendor code. |

Scripts:

```
layer_sweep.py      the worker: one layer, tile list, probes, doses, resume
run_grid.sh         the driver: floors in order, STOP + thermal guard, resume per floor
run_long_tiles.sh   L26-L30 wrapper. Hardcoded. Do not reuse for other floors.
detach.sh           starts run_az.sh + marks_watch in their own session. NOT the grid lane.
watchdog.sh         touches STOP and kills the worker on thermal limit or 12 min stall
floor_table.py      per-floor rate table with v3 labels, dedup, Wilson CI
level_map.py        regenerates LEVELS.md from every corpus (read-only on data)
kappa_test.py       cross-depth agreement test
```

**Stop everything:** `touch STOP; pkill -f "run_grid.sh"; pkill -f "layer_sweep.py"`

---

## 11. Standing orders

- **Never delete data.** Append-only on `FINDINGS.md` and every `.jsonl`. Dedup in memory, never in the file. There is a documented md5 check on `layer_az_L0.jsonl`; keep the head immutable.
- **Do not commit or push unless asked.** Commits are local by default.
- **Do not restart a long job to apply an edit.** Edit, stop cleanly, relaunch. `--resume` costs one model reload, not the floor.
- **Fans first.** If the owner says the fans are loud, answer with `ps -Ao pid,pcpu,rss` and `pmset -g therm`, not a guess. Kill the runner before defending it. Chrome is often the top burner; the agent runtime itself has been too — say so if true.
- **One model at a time.**
- **When wrong, write the correction into `FINDINGS.md`.** That is valued more than a clean narrative.
- **Do not invent results, pushes, or repos.** If not measured, say "not measured".

---

## 12. The second project — do not mix it up

**A. Neuron surgery (this repo).** Steering Granite 4.1 3B at inference time. Everything above.

**B. The model lab (on hold).** Building a small model with a long buzzword stack — distillation on small CoT/SFT sets, SFT → DPO/SimPO → RLOO, abliteration, MoE-ification, speculative decoding, long context. Target was **Nanbeige4.2-3B** (2.6 GB at 4-bit, 88 KiB/token cache). Material on disk: `datasets/` (~20 small sets: s1K, LIMO, Magpie-Reasoning-V2-250K, cot-reasoning-2k, Open-CoT-Reasoning-Mini, NOESIS-1M, Claude-Distills, DeepSeek-V4-Distill, gpt-5.6-luna-sft, Fable-5-traces, DiagSafe), plus `LAB-MODEL.md`, `MODEL-PICK.md`, `OPTIMIZATION.md`, `RAG.md`, `PIPELINE.md`, `CONTEXT_1M.md`, `DATASETS.md`, `TOOLS.md`, `jailbreak-repos/`.

**B has no run in progress and no verified starting point.** If asked, list `models/` and `datasets/` first; do not assume the folder matches the last plan — models were deleted and re-pulled several times.

If asked "1M context on 12 GB", the answer is already in `CONTEXT_1M.md`: a 1M-token KV cache on a 3B transformer at 4-bit is ~16 GB by itself and does not fit; SSM/hybrid states are fixed-size but are not agentic coding models at that size; RAG and chunking are the only things that fit. Do not promise 1M on 12 GB.

---

## 13. Your first message, concretely

1. Read `SUMMARY.md`, the last ~200 lines of `FINDINGS.md`, and confirm the MCP tools are actually present in your tool list.
2. Confirm state: no runner, `STOP` absent, thermal clean.
3. Report in this shape:
   - MCP: 41 / 5 / 4 tools, or state plainly which namespace is missing.
   - L34–L39: **7,420 rows**, 40–44 tiles/floor, all channels 0–39, PERSONA **17 / 14 / 7 / 3 / 0 / 0**, L39 **662 garbled + 538 telegraphic**.
   - Kaggle: **nothing running**, `l0-up` 404, only `l10-up` and `random-gap-l20-l32` exist.
   - Gap fill: L27–L30 **288/288**, L26 **317 distinct cells over 18 channels**.
4. Say plainly: coverage is **~4.3%**, and the L34–L39 persona counts are **not** comparable to the published `Who are you?` layer curve because the probes differ.
5. Ask the pending question: **start L34–36 now, start all six, or wait?** Do not launch until he answers.

No runs. No edits. No downloads. Wait for the owner.
