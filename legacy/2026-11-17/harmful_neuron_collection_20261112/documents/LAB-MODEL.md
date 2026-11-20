# Which model to do the 20 things to

> **SUPERSEDED 2026-11-06, evening. The runner is now `Qwen/Qwen3.5-9B`.**
> SALA is **deleted** — its 1M window is six Nvidia-only kernels and the port fails silently
> instead of loudly. Granite 4.1 8B survives as the *training* target only, because its 160 KiB/token
> cache makes it structurally unable to be the *runner*. See `MODEL-PICK.md` for the one test that
> decides whether a model runs on this machine at all.

Chosen 2026-11-06. Requirement, in the user's words: **a model nobody has touched the way we are
about to touch it**, under 10 GB of RAM, not a 3B, and it has to survive twenty stacked
interventions (distillation from ~20 teachers, abliteration, MoE-ification, DPO/SimPO, SFT,
quantization, steering) without the result being "someone already did that".

"Untouched" is not a vibe. It is countable, and the count is below.

---

## How untouched was measured

Hugging Face indexes every model that declares a `base_model:` tag. Querying that index is a
direct count of how many people have built on a checkpoint:

```
https://huggingface.co/api/models?filter=base_model:<org>/<model>&limit=100
```

Pulled 2026-11-06:

| base checkpoint | derivatives on HF | of those, abliterated/heretic | verdict |
|---|---:|---|---|
| `openbmb/MiniCPM-SALA` | **4** | **0** — all four are quantizations | virgin |
| `ibm-granite/granite-4.1-8b-base` | **9** | 0 for 4.1 | effectively virgin |
| `allenai/Olmo-3-1025-7B` | **100+ (capped)** | 0 | busy, mostly by AI2 itself |
| `Qwen/Qwen3-8B` | **100+ (capped)** | **5+** (huihui, DreamFast, LastMachine, TeskesLab, Thox) | saturated |

The Qwen row is why Qwen is out. Five separate groups have already abliterated it, one has built
a reward model on it, one a reranker, one a Mixture-of-Experts latent-steering variant
(`Vaib202/SecurMoE-Latent-8B`, uploaded *today*), one an agentic security model. There is no
novelty left in that direction; there is only catching up.

---

## How big can we actually go — three different ceilings

The **10 GB number was mine, and it is wrong for this machine.** It came from a graphics-card
budget. `sysctl hw.memsize` is **32 GB** and `iogpu.wired_limit_mb` is `0`, so macOS will wire
~**24 GB** for the GPU. On Apple Silicon there is no VRAM pool to exceed — the GPU reads the same
RAM the CPU does.

Weights only, at three quantization levels (Q4_K_M ≈ 4.85 bits/weight, Q3_K_M ≈ 3.9):

| params | Q8 | Q4 | Q3 |
|---:|---:|---:|---:|
| 8B | 8.5 G | 4.8 G | 3.9 G |
| 12B | 12.8 G | 7.3 G | 5.8 G |
| 14B | 14.9 G | 8.5 G | 6.8 G |
| 17B | 18.1 G | 10.3 G | 8.3 G |
| 22B | 23.4 G | 13.3 G | 10.7 G |
| **30B** | 31.9 G | **18.2 G** | **14.6 G** |
| 32B | 34.0 G | 19.4 G | 15.6 G |

- **At 10 GB total (weights + context):** **14B dense at Q4 with 8K.** Not 9B — that ceiling was
  pessimistic by 5B.
- **At ~24 GB wired (this Mac):** 30B dense at Q4 with 16K (`18.2 + 4.0 = 22.2 GB`), or 30B at Q3
  with 32K, or the 30B MoE below with 256K.

### The rule that is actually measured

Five independent 2026 write-ups converge on the same sentence:

> **a larger model at lower bits beats a smaller model at higher bits** — down to about Q4, often Q3.

Two caveats that matter here:

1. **Coding and agentic tool use degrade faster than perplexity predicts.** One suite's guidance is
   flat: do not go below Q4 for agent workloads.
2. **Q8 is rarely worth its memory.** A 21-test gauntlet scored Q6_K **417/420**, Q4_K_M **405**,
   and **Q8_0 405** — the extra 12 GB bought nothing, and Q6 beat both. One code-diagnosis test
   inside it went 8/20 → 10/20 → **20/20** for Q4 → Q8 → Q6.

### The MoE escape hatch — this is the real answer

Parameter count stops being the constraint when only a fraction of it activates.

| model | total | active | KV/token | Q4 weights | 256K cache | total |
|---|---:|---:|---:|---:|---:|---:|
| **Nemotron-3-Nano-30B-A3B** | 30B | 3.5B | **6 KiB** | 22.0 G | 1.5 G | **23.5 GB** |

**3.3x the parameters of Qwen3.5-9B, with 5x less cache per token**, and it fits this Mac at 256K
context. At 1M it is 28 GB — at the edge, not comfortable. This is the same checkpoint documented in
`CONTEXT_1M.md` §3.

### But training is a different ceiling

Running a 30B and training a 30B are not the same problem.

| size | run on this Mac | LoRA train on this Mac |
|---|---|---|
| 8B | easy | **fits** — the only comfortable target |
| 14B | easy | marginal at Q4, ~14 GB peak |
| 30B | fits at Q4 | **no** — 18.2 GB weights + adapters + optimizer states + activations overflows 24 GB |

**So the split is: run a 30B, train an 8B.** The twenty interventions are a *training* plan, which is
why the pick stays at 8B — and the 8B is not even constrained by the 10 GB figure, since it fits at
Q8 (`8.5 + 1.3 GB` at 8K = 9.8 GB).

---

## The pick: `ibm-granite/granite-4.1-8b-base`

**9 derivatives. Zero abliterated versions. Apache-2.0. ~4.9 GB at Q4_K_M, ~8.5 GB at Q8.**

Architecture, from IBM's model card:

```
8B dense · 40 layers · hidden 4096 · 32 Q heads · 8 KV heads · head_dim 128
MLP 12800, SwiGLU · 131,072 native context · extended to 512K during pretraining
```

Four reasons it wins, and the first is not the size:

**1. It is the same family and the same depth as the model you already measured.**

```
granite-4.1-3b-base    40 layers    MLP  8,192   hidden 2560
granite-4.1-8b-base    40 layers    MLP 12,800   hidden 4096
```

**Forty floors in both.** Layer 16 is at the same relative position. The hook site
(`mlp.down_proj` input) is identical. `corpus.txt`, the median-residual dose unit, the three
probes, the classifier — all of it transfers without a rewrite.

Which makes the 25,843 rows you already have into a **baseline instead of a dead end.** "Does the
sign asymmetry at L16 hold when the model is 2.7x wider, family held constant" is a real question
with a real answer, and nobody has asked it because nobody sweeps both Granite sizes.

**2. Nine derivatives is genuinely virgin** — and the one fine-tune that exists is a SillyTavern
character-card generator, i.e. someone using it, not studying it.

**3. IBM has already moved on.** `ibm-granite/granite-4.2-8b` shipped 2026-08-07 with reasoning,
thinking and tool-calling tags. 4.1-8b-base is a **frozen, abandoned, clean substrate** — the
ideal condition for work you want to be first at.

**4. Under 10 GB at both 4-bit and 8-bit.** No offload, no compromise.

### The cost you are accepting

```
KV cache = 2 x 40 layers x 8 kv_heads x 128 head_dim x 2 bytes = 160 KiB/token
131K native  -> 20 GiB of cache
512K         -> 80 GiB
```

**160 KiB per token is twice the 3B's.** At Q8 weights (8.5 GB) there is 1.5 GB left, which is
about **9,600 tokens** of context. At Q4 (4.9 GB) it is about **32,000 tokens**.

So: **this model is a *training* substrate, not a long-context runtime.** Long context stays with
`Qwen3.5-9B` / `MiniCPM-SALA` in `CONTEXT_1M.md`. That split is correct and should not be blurred —
one model you *build on*, a different model you *run long things in*.

---

## REJECTED 2026-11-06: `openbmb/MiniCPM-SALA`

**Out.** Not because anything below is wrong — because its 1M context lives in six Nvidia-only
kernels (`infllm_v2` + `flash_attn`), the surrounding port is multi-day, and a mis-implemented
top-k fails *silently* by making the model dumber rather than crashing. Measured work is recorded
in `LAB-MODEL.md` git history; the model code has been deleted from `models/`.

The section below is kept because the *analysis* still applies to the replacement search: custom
kernels are the thing to screen for, not parameter count.

**4 derivatives, and every one of them is a quantization.** Nobody has fine-tuned it. Not once.

Read from its actual `config.json` (2026-11-06), not from the announcement:

```
num_hidden_layers        32
mixer_types              8 x "minicpm4" (real attention) + 24 x "lightning-attn"
num_key_value_heads      2
head_dim                 128
max_position_embeddings  524,288        <- 512K NATIVE, not 1M
attn_use_rope            false          <- the 8 attention layers have NO RoPE at all
lightning_use_rope       true
rope_theta               10,000
hidden_size              4096  ·  intermediate_size 16,384  ·  vocab 73,448
sparse_config            window 2048, topk 64, block 64, dense_len 8192
```

**KV cache: 2 x 8 layers x 2 kv_heads x 128 dim x 2 bytes = 8 KiB per token.**

That is the number that matters, and it is **20x smaller than Granite 8B's 160 KiB.**

| context | bf16 cache | q8 | q4 | + 9B Q4 weights (5.5 GB) |
|---|---:|---:|---:|---:|
| 512K native | 4.3 GB | 2.1 GB | 1.1 GB | 6.6 GB |
| **1M (extended)** | 8.6 GB | 4.3 GB | 2.2 GB | **7.7 GB** |

**8 KiB/token is why 1M is reachable here and not on Granite.** The 24 Lightning Attention
layers keep a fixed 12 MB state no matter how long the conversation is — the same trick as
Nemotron and Qwen3.5, applied harder (24 of 32 layers, 2 KV heads).

`attn_use_rope: false` on the attention layers is the second half of it: with no rotary
embeddings there is no out-of-distribution position cliff to fall off, so extrapolation past
512K degrades instead of breaking. It is the same reason the Nemotron-30B holds 86.3 at 1M.

**1M is demonstrated, not configured.** The native ceiling in the checkpoint is 524,288. The
paper reports "up to 1M" and a 1M run on an RTX 5090, which means extension beyond the training
length — read those numbers as tested-external, not native.

### Does it code — the published numbers, 2026-11-06

Read from Table 2 of the technical report (`arXiv 2602.11761`), not from an aggregator. SALA's
own paragraph claims "comparable to modern full-attention models like Qwen3-8B"; these are the
rows behind that sentence.

| benchmark | SALA |
|---|---:|
| HumanEval | **95.12** |
| MBPP | **89.11** |
| LiveCodeBench v5 | **60.48** |
| LiveCodeBench v6 | **52.00** |
| MMLU-Pro | 67.04 |
| CMMLU | 81.5 |
| BBH | 81.5 |
| IFEval | 76.3 |
| AIME 24 / 25 | 83.75 / 78.33 |
| **average** | **76.53** |

Long context, same report:

| benchmark | 32K | 64K | 128K | 512K | 1M | 2M |
|---|---:|---:|---:|---:|---:|---:|
| RULER | — | 92.7 | 89.4 | 87.1 | **86.3** | 81.6 |
| NoLiMa | 54.5 | 43.0 | **23.9** | — | — | — |
| MRCR (2-needle) | — | 29.8 | 28.6 | — | — | — |

Three things to read off this rather than the headline:

- **HumanEval 95.12 is not evidence of anything in 2026.** The benchmark is saturated; nearly every
  8B scores 90+. The row that carries signal is **LCB-v6 at 52.00**, and the paper's own table puts
  it 1–6 points *below* Qwen3-8B on the code rows, not above. "Comparable" means the same class,
  not better.
- **RULER 86.3 at 1M and NoLiMa 23.9 at 128K are the same model.** RULER is synthetic needle
  retrieval — find the string. NoLiMa makes the model *reason* about what it found, and it falls
  to 23.9 well before the advertised window. The effective **reasoning** context is a small
  fraction of 1M; the retrieval context is genuinely 1M. Those are different products.
- **The agentic evals are absent.** No SWE-bench, no Terminal-Bench, no BFCL, no tau-bench in the
  report. "Can it drive a repository" is unmeasured by the authors, which is exactly the question
  that matters for this project.

What it *does* have is a tool format: the tokenizer config ships a function-calling chat template
(`<function name="..."><param name="...">` XML) and SGLang takes `--tool-call-parser minicpm4_xml`.
So it is built to be an agent, not just a chat endpoint. Nobody has measured whether it is a good one.

### Running it on the Mac — measured 2026-11-06

An earlier version of this file claimed the model assert-fails on MPS. **That was wrong**, and
the reason is worth keeping because it is the difference between reading an assert and reading
the branch that guards it:

```python
# modeling_minicpm_sala.py:2448
if config.sparse_config is not None and torch.cuda.is_available():
    self.self_attn = MiniCPMInfLLMv2Attention(...)      # the assert lives in here
else:
    self.self_attn = MINICPM_ATTENTION_CLASSES[config._attn_implementation](...)
```

The assert (`line 1328`) is inside `MiniCPMInfLLMv2Attention.__init__`, which is only constructed
when `torch.cuda.is_available()`. **On MPS it is unreachable dead code.** A non-CUDA fallback is
built in: the 8 attention layers become plain SDPA, the 24 Lightning layers are unaffected.

The real blockers were three others, and all three are now cleared:

| blocker | why | fix |
|---|---|---|
| `einops` missing | imported at module level | `pip install einops` |
| `fla` missing, **unguarded** | `chunk_simple_gla` / `fused_recurrent_simple_gla` are **Triton kernels**, and Triton has no Metal backend. Unlike `flash_attn`, these imports are *not* inside a `try/except` — so the module will not import at all. | `models/sala_mps_shim/fla.py` — pure-PyTorch simple GLA |
| `is_torch_fx_available` deleted | the modelling code targets transformers **4.56**; this venv has **5.22.1**, where the symbol no longer exists | `models/sala_mps_shim/sala_compat.py` |

**Result — `test_sala_mps.py`, tiny random-weight config (3.21M params, 2 attention + 2 lightning layers):**

```
[1] shim vs closed form    max_abs_err=9.5e-07  rel=1.2e-07   PASS
[2] CPU  float32           logits=(1,32,1024)  finite=True    13 ms
[3] MPS  float32           logits=(1,32,1024)  finite=True   360 ms
    MPS  bfloat16          logits=(1,32,1024)  finite=True   103 ms
```

The architecture executes on Metal, both mixer types, in bfloat16.

**What that does and does not prove — read this before trusting the numbers:**

- It proves the **code paths run**. It does not prove generation works. This is random weights and
  one forward pass, not one token of text.
- The shim is **not bit-exact** with `fla`. It is verified against the *closed form* of simple GLA
  (the PASS above), not against the CUDA kernel. Decay ordering is the one place a one-token offset
  can hide, and a diff against a CUDA run is still owed.
- **Sparse attention is off on MPS**, because it is gated on `torch.cuda.is_available()`. The 8
  attention layers run dense SDPA. The KV cache is still 8 KiB/token, but attention compute on
  those layers goes O(n²) — so **the 1M-token property, the entire reason to pick this model, does
  not transfer to the Mac.** Treat Metal as a low-thousands-of-tokens prototyping lane.
- The shim is a **Python loop over the sequence**. Fine for a forward pass, hopeless for long
  generation. A real MPS lane needs the recurrence as a chunked matmul, which is a rewrite.

Verdict: it now loads and runs here. The Mac is where you prototype and debug; anything that needs
the actual long window still goes to CUDA.

Two things make it seductive:

- **It already has the 1M context.** Twenty interventions and not one of them has to buy window.
  That deletes the hardest requirement in the whole plan.
- **Zero fine-tune competition.** Whatever you produce is the first of its kind, by measurement.

Three things make it a second pick, not a first:

- `model_type: minicpm_sala`, `custom_code`, custom `configuration_*.py` and `modeling_*.py`.
  Every tool needs `trust_remote_code=True` and has to be checked against the architecture.
- **Lightning Attention layers have no KV cache in the normal sense.** Your `down_proj` hook still
  exists (the MLP is standard), but anything that reasons about cache, eviction or attention
  entropy breaks.- **The GGUF path does not exist, and that is now measured, not guessed.** Searching the Hub for the architecture returns **15 derivatives and not one with a `gguf` tag** — AWQ 4/8-bit, NVFP4, GPTQ w4a16, FP8, EAGLE-3 draft heads, SGLang builds, plain re-uploads. So **llama.cpp cannot run this model at all**, and the runtime installed on this machine is llama.cpp.

  Counted against the API rather than eyeballed: of **15 repos, 0 carry a `gguf` tag and 0 carry an `mlx` tag.** The only Apple-shaped artifact is **`moxin-org/MiniCPM4-SALA-9B-8bit-mlx`** — 8-bit, an 11-download upload from 2026-02-12, day two of the release, tagged only `safetensors`/`custom_code`. That is the whole Apple Silicon story: one unofficial 8-bit conversion with "mlx" in its name, and no 4-bit anywhere.

  Consequence, stated plainly: choosing MiniCPM-SALA means choosing **Transformers-on-MPS or vLLM-on-CUDA** as the only ways to run it. It is not a llama.cpp model and may never become one.

Read it as: *the highest-novelty target and the highest-friction target, in the same checkpoint.*

---

## Training it on GitHub repos — the shape problem

The instinct is right and the execution as stated does not work. A repository is **source, not
training pairs**, and SFT needs `(instruction -> response)`. Feed it raw files and you teach file
autocompletion, which is a different model from the one you want.

Three shapes survive the filter:

| shape | what it is | where it comes from |
|---|---|---|
| **commit / PR diff** | (broken tree + issue text) -> (patch) | merged PRs, ideally with CI green |
| **FIM** | (signature + docstring + neighbouring lines) -> (body) | StarCoder-style fill-in-the-middle |
| **repo walk** | (files 1..N) -> (file N+1) | sequential tree order |

**Only the first is worth real effort, and not for SFT — for RLVR.** A merged PR whose tests passed
is a *free, verifiable reward signal*: apply the patch, run the suite, pass or fail. That is the
scarce commodity in this whole project. `PIPELINE.md` already names RLVR as the step that adds
capability rather than selecting it; commit histories are the one place it comes at scale for free.

### The two numbers that bound how much to pull

- **LIMO: 817 rows. s1K: 1,000 rows.** Both beat sets hundreds of times larger.
- `OPTIMIZATION.md` §5: **SFT selects and formats. It does not create.**

So "a bunch of repos" optimises an axis that has already been measured not to matter. The question
is never *how many repos*, it is *how many verifiable pairs*.

### Against what is already on disk

```
datasets/ total          7.2 GB
  reasoning / distill    ~7.2 GB   (Claude, Opus, DeepSeek, Magpie, NOESIS, LIMO...)
  code                   1.2 MB    <- Code-sonnet-5-gpt-5.5-kimi-k2.5
```

The gap is real and it is not 7 GB wide. One curated code set in the 50–500 MB range — PR diffs with
tests, or a clean instruction set — closes it. Pulling The Stack to fill the gap adds licence
exposure and no capability.

### The licence trap, which is specific to code

Text corpora are contested in theory. **Code is contested in court.** Every repo carries its own
licence and copyleft ones (GPL/AGPL) do not evaporate because you turned them into tokens. The Stack
ships an opt-out filter for exactly this reason and it is still disputed. Filter to permissive
licences *at pull time*, not after — this is the first question anyone asks about a released model
and the hardest to answer retrospectively.

### What this does not buy

Training on repos does not make a model good at *codebases*. Navigating a repo is a scaffold
property — grep, read, test, recover — and `OPTIMIZATION.md` measures the harness as the larger
lever for a 3B-class model. Code data improves syntax and local patterns; it does not install
navigation.

---

## RL, MoE and CoT on SALA — answered 2026-11-06

| buzzword | possible on SALA? | on this Mac? | worth doing? |
|---|---|---|---|
| RL (GRPO/DPO) | **yes** | **no** | yes — on rented CUDA |
| MoE | **yes, structurally** | yes | **no** — solves a problem it does not have |
| CoT | **already shipped** | — | nothing to add |

### The MLP makes MoE easy, and that is the trap

`modeling_minicpm_sala.py:745` is a stock SwiGLU — the same three matrices Llama uses:

```python
down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

So MoE-fication is mechanical: cluster the 16,384 intermediate channels per layer into N experts,
add a top-k router. Nothing in the architecture resists it.

It is still the wrong move, for two reasons:

1. **It buys compute, and SALA's compute advantage is already spent.** 24 of 32 layers are linear
   attention — the speed came from there, not from the MLP. A router, a retrain and a distribution
   shift, to make something already fast marginally faster.
2. **MoE adds no knowledge.** It routes the same stored facts more cheaply. Against the goal
   "frontier level", this is the one item on the list that cannot help.

Tooling gap is real too: MoE-fication scripts expect a `LlamaMLP` in a known repo layout, not
`MiniCPMMLP` behind `trust_remote_code`. Expect to write the trainer yourself.

### RL is possible — and TorchRL is a legitimate tool, contrary to what I first assumed

**Correction:** `pytorch/rl` is not only robotics and control. TorchRL ships an LLM interface
(`torchrl.modules.llm`) with `GRPOLoss`, `DAPOLoss`, `CISPOLoss`, `SFTLoss`, `LLMCollector`, and
wrappers for Transformers, vLLM **and SGLang** — which matters, because SGLang is SALA's fast path.
It is a real option, alongside TRL (`GRPOTrainer`) and OpenRLHF (Ray + vLLM, 70B+).

The blocker is not the library. **It is the shim.** RL generates thousands of rollouts and scores
them, and every rollout runs through the model. On this Mac that means running through a Python-loop
stand-in for a Triton kernel, not the real thing. You would be training SALA to satisfy the shim's
errors.

- **This Mac: no.**
- **Kaggle/CUDA with SGLang: yes** — and that is the only lane where the resulting number means
  anything.

### CoT is already in the model

The tokenizer's chat template has a first-class `reasoning_content` field:

```jinja
{%- elif message.role == "assistant" %}
    {%- if message.reasoning_content is string %}
```

The benchmark table backs it up: **AIME 2024 83.75, AIME 2025 78.33** from a 9.5B hybrid — better
than most 8B models. There is nothing to add. Prompt it or SFT it; do not build a CoT stage.

---

## Rejected, with the reason

| model | why not |
|---|---|
| **Qwen3-8B / Qwen3.5-9B** | 100+ derivatives, 5+ abliterated, a reward model, a reranker, an MoE variant. Every idea lands on top of someone else's. |
| **Llama 3.x 8B** | The most fine-tuned checkpoint in history. Same problem, worse. |
| **Olmo 3 7B** | Not actually untouched — 100+ derivatives, though nearly all are AI2's own post-training stages. **Still the best choice if provenance matters more than novelty**: weights, Dolma 3 (9.3T tokens), code and intermediate checkpoints are all public, so a behaviour change can be traced to a document. Different question, different answer. |
| **Gemma 3 12B** | Licence terms, not size, are the blocker. |
| **Nemotron-Nano-9B-v2-Base** | `nvidia-open-model-license`, **not** Apache-2.0. If this ever gets published, that matters. Distinctive asset: hybrid Mamba-2, 5.9 GB at Q4 — but the licence is a real constraint, not a footnote. |
| **Granite 4.2 8B** | Newer and better (thinking + tool-calling built in), but 4.2 is where people are currently working and it already has a Heretic build. Go one generation back to be first. |

---

## What this costs before anything trains

```
.venv has:  torch 2.15.0  transformers 5.22.1  datasets  safetensors
.venv lacks: mlx  mlx-lm  peft  trl  accelerate  bitsandbytes  gguf
```

**Nothing can train yet.** The model choice is necessary and not sufficient — the trainer is the
blocker, and it is the same Python 3.14 wall documented in `PIPELINE.md`.

```bash
uv python install 3.13
uv venv --python 3.13 .venv-train
.venv-train/bin/pip install mlx-lm peft trl datasets
```

Separate environment. Do not install training dependencies into `.venv`, which is pinned to
transformers 5.22.1 for the neuron tooling.

---

## Order

```
1. download   ibm-granite/granite-4.1-8b-base      ~16 GB bf16, or take a Q8 GGUF
2. verify     hook fires: rerun ONE known neuron at L16 on the 8B, compare to the 3B row
              <- this is the test that the transfer is real, and it costs one generation
3. ceiling    exemplar-in-context test (OPTIMIZATION.md §5) before any SFT
4. SFT        small set first, 817-2,000 rows, measure tok/s
5. then       distill / DPO / MoEfication / abliterate, in that order
```

Step 2 is the one that must not be skipped. If the same neuron at the same floor does not produce
the same *kind* of answer on the wider model, the family-transfer assumption is wrong and the
whole reason for picking Granite collapses — cheap to check, expensive to discover later.
