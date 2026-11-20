# The toolkit — external repos we actually use, and why

Started 2026-11-06. One verdict per repo: **USE** (runs here), **READ** (method or
numbers transfer, code does not), **SKIP** (wrong shape for this project).

The dividing line on this machine: **Apple M1 Max, MPS, no CUDA.** Anything built on
CUDA kernels, CUDA graphs, dp4a, NVFP4 or Triton is READ-only, no matter how good the
number is. Anything that is plain PyTorch, llama.cpp flags, GGUF, or pure methodology
transfers.

---

## 0. In plain English

**Who he is.** Adrian, in Rome. He makes AI models run on a 2016 graphics card with **8 GB**.
You are trying to do the same thing with 8-12 GB. He has 19 followers and no stars, because
nobody cares about old cards. **Ignore the star count.**

**Why his code is useless to you.** Everything he builds is for NVIDIA cards. Your Mac is not
one. It is a diesel repair manual for a petrol car — the idea of how to check the engine
travels, the wrench sizes do not.

**What is actually useful: three things.**

> **1. How he tests.** This is the gold, and it is free because it is a habit, not software.
>
> He never says "it got faster" without: running it forward-back-back-forward and throwing
> away the first run; reporting the **median**, not the best; treating anything under **3%**
> as noise; and fixing the random seed, because without it *the same test twice differed by
> 10%*.
>
> Your own notes say the classifier is right about **half** the time and barely agrees with
> itself. That is the exact disease his method cures.

> **2. One repo that actually runs on your Mac.** He fit an 8-billion model into 8 GB by
> keeping most of it in ordinary memory and pulling in only what is needed. That is the same
> trick the 12 GB plan needs for a 30-billion model. He also found a trap: when you feed the
> model several things at once you must pad them to equal length, and **padding changes the
> answer** (similarity 0.76 vs 0.999956). Batch by exact length or you get quietly wrong
> numbers.

> **3. A speed trick that does port.** 33 -> 83 tokens per second on a 2B model using a
> **draft model**: a tiny fast model guesses the next few words, the big one only says yes or
> no. The settings are llama.cpp settings, not NVIDIA settings. That is the speculative
> decoding lever with a working recipe attached.

**The one to read even though it is not a tool: `flytollm`.**

He took the complete brain of a fruit fly — every one of **166,700 neurons, 2.75 million real
connections** — and used it as the engine of a language model. Then he tested it against
fake brains, and **the fake ones won**. Randomly-rewired fly brains scored better. Shuffled
connections scored better. A simple word-guessing trick beat it.

**He published that anyway, and leads with it.**

```
86% of the fly's neurons never fire at all; 10% are active per word
cut the most-connected 10% of neurons  -> +1.75 (much worse)
cut 10% random ones, same connection count -> ~0 (no change)
connection strengths moved 0.3% after training
shuffling those strengths changed nothing
```

That third and fourth line together: **the wiring barely moves, and what learned is the front
door and the exit** — how text gets in, how answers get out — not the connections.

**Why that should worry this project.** The steering work pushes one neuron and asks what the
model says. His result says the behaviour lives in the transport, not the tile. That is the
same thing this repo already found: a random **2-channel** direction beats every hand-picked
neuron. The unit was never the neuron.

He also **retracted one of his own published numbers** and left the correction visible instead
of deleting it.

**The bonus.** His profile links a multi-agent tool where each worker, when its memory fills,
**writes a handover note** and the next one picks up. That answers the open question from
`CONTEXT_1M.md` — what happens when a subagent's 22K runs out.

---

## Adopted

| repo | what we take from it | verdict |
|---|---|---|
| **`MakazhanAlpamys/Soup`** | LoRA SFT / DPO / GRPO on Apple Silicon via a native **MLX** backend; `soup profile` prices a run against Metal's working set; `soup export --format gguf`. Apache-2.0, ~10.2k stars, PyPI `soup-cli` v0.77.2. | **USE** — see `PIPELINE.md` for the install (needs Python 3.13, not 3.14) |
| **`ggml-org/llama.cpp`** | the runtime for every 1M-context plan; prefix cache, KV quant, GGUF | **USE** — not installed yet |
| **`heretic-llm`** | abliteration on the *merged* checkpoint, after SFT | **USE** (later step) |
| **`NVIDIA/garak`** | probe families + detectors instead of hand-rolled regex | **USE** — already cloned |

---

## Rejected

| tool | why not |
|---|---|
| **Unsloth Core** | needs `bitsandbytes`, `triton` and `xformers` — all CUDA. Its own requirements page lists "Apple/Silicon/MLX is in the works" and "your device should have xformers, torch, BitsandBytes and triton support". The "2x faster, 70% less memory" headline is a **CUDA kernel** result; there is no equivalent kernel on Metal. |
| **Unsloth Studio** (the Mac path) | does run on macOS — but it **trains via MLX**, not via Unsloth's kernels. So it delivers the MLX speedup, not the Unsloth speedup, and it is one more dependency around the same thing Soup already does. |
| **Unsloth on this interpreter** | requires Python **3.11-3.13**. This venv is **3.14.8** — the same floor that stops `soup-cli` in the project venv. |

Verdict: **not installed, and not worth installing.** The Apple Silicon advice from the ecosystem
itself is "do not install standard Unsloth on an M-series Mac; use the MLX path instead." That
is exactly what `soup-cli[mlx]` already is, so Unsloth adds a dependency and no speedup.

---

## `ArtyomITA` (Adrian Ciprian Iancu) — reviewed 2026-11-06

AI engineer, Roma. 37 public repos, ~19 followers, ~0 stars on most. **Low stars is not a
signal here** — the work is dense, measured and self-critical in a way that high-star repos
usually are not. Everything he does targets one GeForce GTX 1080 (Pascal, sm_61, 8 GB).

### The three that are worth real time

**1. `nvidia-gtx1080-sm61-ai-full-guide` — READ. Highest-value repo in the set.**

Not a tool: a measurement log. But the *how he measures* is the most useful thing on this
page, because this project already has the disease it prevents.

```
A/B-B-A alternating runs, first run dropped, median reported, <3% = noise
kernel changes must be bit-identical, or pass a KL-divergence gate, before any speed number
sampling seed fixed -- without it two runs at temp 1.0 differed by up to 10%
thermals recorded next to every number (his card sits at 91-94C after 3 min)
```

This repo's own classifier audit found per-row precision **0.49** and kappa **0.21**. Four
lexical effects have already been inflated here. The protocol above is the cheap fix, and
it costs nothing to adopt.

Also worth lifting, and it inverts one of our assumptions: **"at batch size one, decoding is
bound by kernel launches, not memory bandwidth."** That is true of a 2016 CUDA card. It is
**not** true of an M1 Max, which is bandwidth-bound at ~400 GB/s. Do not port the conclusion,
port the discipline of checking which regime you are in first.

**2. `llm2vec-8gb-offload-lossless-batching` — USE. The one repo whose code runs here.**

Plain PyTorch, no kernels. Runs Llama-3-8B on 8 GB via RAM offload, and batches without
changing the output.

The finding that matters: **padding shifts embeddings.** Cosine similarity **0.76** with
padding versus **0.999956** batched by exact token length. 14,000 sentences in 3h16m.

Two things follow for us:
- It is the working reference for the "weights in RAM, hot path in VRAM" pattern that the
  12 GB Nemotron plan depends on.
- If we ever batch inference or scoring, **batch by exact length, not padded**. Silent 0.76
  cosine drift is exactly the kind of wrong number that survives review.

**3. `minicpm5-kernel-pascal` — READ the flags, not the kernels.**

33 → 83.5 tok/s on a 2B model. The portable part is **FR-Spec speculative decoding**, which
is a llama.cpp feature, not a CUDA one:

```bash
-md <draft.gguf> -ngld 999 \
--spec-type draft-dspark --spec-draft-n-max 5 --spec-draft-n-min 0
```

Measured, and paired with the honest framing that a draft only pays when acceptance is high.
That is `TOKENS.md` L4 with a real config attached. The Q8 kernels and the dead-KV-slot
erasure do not port.

### The one to read for Lane 1, not Lane 2

**`flytollm` — READ. This is the most conceptually relevant repo to the neuron project.**

The entire male *Drosophila* CNS connectome — **166,700 neurons, 2,753,975 real synapses** —
used uncut as the recurrent spiking core of a language model. 5.46M parameters, one GPU.

Its structured findings are a mirror of what this project is trying to establish, and its
null-model design is better than anything in `PATTERNS.md`:

| finding | why it matters here |
|---|---|
| **"Every null model beats the real wiring"** (3.634 real vs 3.410 trigram) | a striking-sounding neuron-level structure can lose to a dumb baseline. The honest headline is the *null*, not the effect. |
| **85.8% of neurons never fire**, 10% active per token | at neuron level, most of the unit of account is dead. Directly relevant to "the tile is a bad unit" (`PAPER 3.8`). |
| **lesion the rich club → +1.3/+1.5/+1.75 nat; equal-degree random control ≈ 0** | a causal neuron-level claim with the *matched control* that makes it a claim. |
| **weights barely move: 0.11% at 1000 updates, 0.30% at 8000** | "after 8000 updates the core is a fixed reservoir; learning lives in the interfaces (embedding, gates, readout, attention)." |
| **permuted synapse weights: no effect** — "the information is in what stays fixed" | disturbing weights ≠ disturbing the computation. |
| a published spectral-radius result was **retracted and corrected** in-place | the standard of correction this repo should hold itself to |

That last cluster is the sharpest warning available for the steering work: **if the
learned core barely moves and the interfaces carry the behaviour, then steering one MLP
channel is a small probe on a mostly-fixed object** — which is exactly what the random
direction results already suggested (`PAPER 3.8`: random 2-channel beats every hand-picked
neuron).

### Read, don't run

| repo | what it is | takeaway |
|---|---|---|
| `bonsai2-ternary-pascal` | Ternary 9B, +51% decode, **bit-identical** | ternary (PQ2_0) + MTP fits a 9B in tiny memory at no measured quality cost. Needs the PrismML llama.cpp fork and CUDA. |
| `blackwell6000-qwen3.8-flash-next` | 4 users × **227k context**, ~100 tok/s each, exact kernels + determinism fix | the long-context serving reference. 227k on one card is the shape the 12 GB plan is aiming at. |
| `hildanext` | AR → discrete diffusion LLM (LLaDA 2.1 recipe, Qwen3-0.6B) | a genuinely different architecture. Not now; note that it exists. |
| `vergilius` | local OSINT assistant, small models, no external provider | adjacent product shape, not a tool |
| profile: `claude-plugins` | multi-agent harness — fixed roles (worker/proposer/auditor/registrar), a role **writes a handover when its context fills**, hosts swap on quota exhaustion, **long tool output compressed per agent** | this is `CONTEXT_1M.md` §5 built for real, and the handover pattern is the answer to "what happens when a subagent's 22K runs out". Influenced by 24 papers. |

### Skip

`llama.cpp` fork (CUDA), `stable-diffusion.cpp` fork, `bonsai-webgpu-tiled-vae-fix`,
`lfm-to-kimodo`, `stt-whisper-pad`, `odysseus`, `Shadowbroker`, `fantacalcio-*`,
`storico`, `K-Forecast`, `sp500-predictability-lab`, `idle-isekai-trials`,
`supermario-1-1-rl`, `fantaoracle`, `QPT`, `Shejire`, `HackDay`, `MedHubHAQ`,
`TubeMind`, `Permit-Forge`, `ResearchHubV2`, `UniFinderV2`, `Xexun_Mobile`,
`Performance-Management-Platform`, `what-if`, `pulsekz`, `claudeshield`,
`Kaspi-Analytics-Platform`, `NASA-Exoplanet-Detection-AI`, `Higgsfield-DS`,
`Halyk-AI-Challenge`, `soup-kernel-challenge`, `soup-connectome`, `Namedrop`, `aet`,
`docx-mailmerge-sender`, `Telegram-Chat-Scraper`, `memory-service`, `G1-Ping-Pong-Lab`,
`AI-Layout-Configurator-for-Construction`.

None of these touch a 3B model, a context window, or a neuron. `memory-service` (Postgres +
pgvector long-term memory) is the only one that would be interesting later, and only if the
subagent architecture needs persistent state.

---

## The one-line verdict

**His kernels are useless here and his measurement protocol is the most valuable thing
anyone has handed this project.** `flytollm` is a free lesson in what a neuron-level claim
has to survive before it counts, delivered by someone whose null models beat his own model
and who published that result.
