# Jailbreak / red-team repos — what they are, what they're good for

Reviewed 2026-11-06. Cloned into `jailbreak-repos/` (19 MB total, 4 repos).
Nothing here touches the neuron map — these are all **prompt-level** artifacts. Their
value to this project is as *probes*, *baselines* and *detectors*, not as data.

---

## Pulled

| repo | size on disk | files | license | what it actually is |
|---|---:|---:|---|---|
| `NVIDIA/garak` | 18 MB | 881 | Apache-2.0 | LLM vulnerability scanner. **45 probe modules / 38 families**: dan, encoding, glitch, lmrc, latentinjection, promptinject, grandma, realtoxicityprompts, suffix, tap, malwaregen, badchars, divergence. Python, pip-installable, has a local `huggingface` target. |
| `togg53192-cmd/jailbreaks` | 940 KB | 41 | **none** | 32 paste-ready **system prompts / persona locks / AGENTS.md specs**, ~70k words, one file per model. Not probes — full instruction sets. |
| `0xk1h0/ChatGPT_DAN` | 196 KB | 2 | **none** | One README, 9,476 words. The DAN 1→13 lineage. Historical, 2023-era. |
| `langgptai/LLM-Jailbreaks` | 164 KB | 3 | Apache-2.0 | One README, 2,186 words. DeepSeek R1 / Grok3 / Gemini 2 / DAN snippets. |

## Not pulled, and why

| repo | size | why not |
|---|---:|---|
| `microsoft/PyRIT` | **486 MB** | MIT, but it's a full red-team framework with its own orchestrator, targets and scorers. We already have 40k rows of purpose-built harness for one model. Nothing here we don't already do. |
| `promptfoo/promptfoo` | **811 MB** | TypeScript monorepo (now under OpenAI). Eval harness for API models, not local hooked weights. Wrong language, wrong shape. |
| `elder-plinius/GLOSSOPETRAE` | 1.5 MB | **AGPL-3.0.** Different research program entirely: procedural conlangs, K-shot acquisition cliff, tokenizer covert channels, semantic stego. Interesting, not relevant to steering. Do not vendor the code — copyleft. |
| `Alejandros17487/ai-jailbreaks` | 378 KB | **Duplicate of `togg`** — same 35 prompts, same section order, same ad copy, only case/hyphens differ. Pulling both buys nothing. |
| `ShadowHackrs/Jailbreaks-GPT-Gemini-deepseek-` | 264 KB | One "ollie" operating contract buried in attribution locks and morse code. The *format* (register + prohibitions + hard triggers) is mildly instructive; the content is noise. |

---

## The three uses that are real

**1. Probe diversity — the important one.**
Every row in this project rests on three fixed probes: `hello`, `what is your name`,
`do you like humans`. If the killer register only appears on the third one, it's a
prompt artifact, not a property of the neuron — and you already spotted that
("saying like or hate encourages the answer"). garak ships prompt families with
names, so the same tile can be re-asked cold. That is a cheap test of whether a
finding survives its probe.

**2. A prompt-only baseline.**
`togg` is 32 whole persona locks. If a system prompt alone produces the same hostile
register without touching a weight, the honest claim about steering changes: it buys
*persistence*, not personality. Reviewers will ask this. Better to have the number.

**3. Detector vocabulary.**
garak's detectors are how "did it refuse / was it harmful" gets scored without a
regex. This repo has inflated four separate effects with lexical counts already.
Using a maintained detector set instead of hand-rolled regex is the fix.

---

## Traps

- **DAN prompts are from 2023** and target GPT-3.5/4. Granite 4.1 is April 2026.
  Expect most of them to do nothing, and don't read that as "the model is safe".
- **These are system prompts, not user turns.** Pasting a persona lock into the
  system prompt *and* steering a neuron confounds the experiment — you won't know
  which one produced the answer. Keep the two arms separate.
- **Licenses are mixed.** garak and langgptai are Apache-2.0 (safe to vendor and
  quote). `togg` and `ChatGPT_DAN` have **no license at all** — read them, don't
  redistribute. GLOSSOPETRAE is AGPL.

---

## The one thing worth building from this

A probe bank: pull the user-turn prompt lists out of garak's `dan`, `lmrc`,
`grandma` and `promptinject` modules into a single JSONL, then re-run a handful of
already-mapped tiles across it. Same dose, same neuron, five different questions.
That directly answers "is this the neuron or the sentence" — and it costs one short
sweep, not a new sweep.
