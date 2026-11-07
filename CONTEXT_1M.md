# 1M context on 12 GB — the arithmetic, the model, and subagents

Written 2026-11-06. **Lane 2/3 (the lab), not the neuron research.** Nothing here touches
`models/granite-4.1-3b-base/`.

---

## 0. In plain English

**A normal model takes notes. Every layer, on every word, forever.**

That notebook is called the KV cache. It is not the model, and it is not the prompt — it is
scratch paper the model writes while reading, and it has to re-read that paper to say each
next word. A 9B model writes **128 KB of notes per word**. One million words of notes is
**122 GiB of scratch paper.** A 12 GB card holds 11.

That is the entire reason big context "needs" big GPU. Not intelligence — **note-taking.**

**Nemotron mostly doesn't take notes.** It is 52 layers. Forty-six of them don't keep a
notebook at all; they keep one fixed feeling about the conversation — 47 MB, the same 47 MB
whether you said ten words or a million. The difference between *"his account number was
4471"* and *"he sounded annoyed."* Only **6 layers** keep real notes.

```
6 note-taking layers x 6 KB per word   =  6 KB of notes per word
1,000,000 words x 6 KB                 =  5.7 GiB
squeeze the notes to 4 bits            =  1.4 GiB      <- fits on the card
```

**Where the 22 GB of model goes.** A 30B model is a team of 30. Only **3.5 of them work on
any one word** — the other 26 are specialists waiting for their subject to come up. So the
card holds the 8 GB that is always needed, and the other 14 GB of specialists sit in system
RAM and get called in when relevant.

```
card (11.3 GiB usable)      notes 1.4  +  always-on 8.4  +  working space 1.5  = 11.3
system RAM                  the other 13.6 GiB of specialists, on call
```

**What it costs you.** Reading the first million words takes 10-20 minutes, once. Talking
afterwards is slower than a model that fits entirely on the card, because some answers have
to fetch a specialist from RAM.

**What it does not give you.** It misses about 14 of every 100 needles at 1M (86.3 RULER),
and it is not a frontier coder. The window is real; the precision is not perfect.

---

## 0b. The short answer, arithmetic checked (2026-11-06)

The question was "1M context on a 12 GB laptop." Two things need correcting before the answer
is useful, and both are measurements rather than opinions.

**Your machine is not 12 GB.** `sysctl hw.memsize` = **32 GB**. `iogpu.wired_limit_mb` = 0,
which means macOS lets the GPU wire up to ~75% of RAM, i.e. **~24 GB**. On Apple Silicon there
is no VRAM pool to exceed — the GPU reads the same RAM the CPU does. So 12 GB is a *card*
budget, and the card is not what you own. Numbers below are given for both.

**The cache price is set by how many layers keep a notebook, and nothing else.** Computed from
each actual `config.json` with `2 x L_full x kv_heads x head_dim x bytes`:

| model | full-attn layers | KV heads | head_dim | **KV / token** | 1M @ bf16 | 1M @ q8 | 1M @ q4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Granite-4.1-3B | 40 of 40 | 8 | 64 | **80 KiB** | 76.3 GiB | 38.1 | 19.1 |
| **Qwen3.5-4B** | **8 of 32** | 4 | 256 | **32 KiB** | 30.5 GiB | 15.3 | 7.6 |
| **Qwen3.5-9B** | **8 of 32** | 4 | 256 | **32 KiB** | 30.5 GiB | 15.3 | 7.6 |
| Nemotron-3-Nano-30B-A3B | **6 of 52** | 2 | 128 | **6 KiB** | 5.7 GiB | 2.9 | 1.4 |

Read column five. **Qwen3.5 puts three of every four layers on Gated DeltaNet**, which keeps a
fixed-size state instead of a growing notebook — 24 of its 32 layers cost the same at token
1,000,000 as at token 10. Verified in the local `Qwen/Qwen3.5-4B` config: `layer_types` is 24x
`linear_attention` + 8x `full_attention`. That is a **2.5x** saving over Granite, and it is the
entire difference between impossible and fits.

**Then weights decide it, and they flip the answer.**

| model | weights @ Q4 | + 1M @ q4 KV | **total** | 12 GB card | your 32 GB Mac |
|---|---:|---:|---:|---|---|
| **Qwen3.5-4B** | 2.5 GB | 7.6 GB | **~10 GB** | **fits, no tricks** | trivial |
| **Qwen3.5-9B** | 5.7 GB | 7.6 GB | **~14 GB** | over by ~2 | **fits, ~10 GB spare** |
| Nemotron-30B-A3B | 22 GB | 1.4 GB | **~23.5 GB** | needs expert offload | fits, fully resident |
| Granite-4.1-3B | 2.0 GB | 19.1 GB | 21 GB | no | no |

> **The model with the smallest cache is not the model that fits.** Nemotron-30B writes 5x less
> scratch paper than Qwen3.5 and still loses a 12 GB budget, because 22 GB of weights cannot be
> quantized away. Qwen3.5-9B reaches one million tokens in **14 GB total** with no expert
> offload, no spill to system RAM, and no flags beyond KV quantization.

**So, concretely:**

- **12 GB card, no offload:** `Qwen3.5-4B`. ~10 GB at 1M, q4 KV, entirely resident.
- **Your 32 GB Mac — this is what to actually run:** `Qwen3.5-9B`. ~14 GB at 1M, ~10 GB left
  for the OS. Unified memory, no spill, no OOM.
- **Biggest brain that still fits your Mac:** the Nemotron-30B-A3B in §3. 23.5 GB resident,
  6 KiB/token, RULER 86.3 at 1M — but slower, SWE-bench 38.8, and it is a 30B MoE you cannot
  fine-tune on this machine.

**Why the 9B over the 30B if only one gets picked:** it is agentic in the way that matters —
native tool calling, thinking mode on by default, vision, 201 languages, Apache-2.0, MTP-trained
— and it is small enough to be the model you *also* fine-tune later. The 30B is not.

**The command.** `llama-cli`, `llama-server`, `mlx_lm` and `ollama` are all absent on this
machine; `brew info llama.cpp` resolves to 0.8.0:

```bash
brew install llama.cpp

llama-server -m Qwen3.5-9B-Q4_K_M.gguf \
  -c 1000000 -np 1 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --jinja            # required for tool calling; without it tools silently do nothing
```

**The contradiction to keep in view.** `OPTIMIZATION.md` §4 lists *"more context at lower
precision"* as a measured anti-pattern — FP16/12K beat 4-bit/32K on AppWorld. Running 1M at q4
KV is exactly that trade. So 1M here is a **survey instrument, not a precision instrument**:
ingest once to build the map, then work at 32K-128K with 8-bit KV. §5 already concludes this and
the arithmetic above does not change it.

---

## 1. Why 262K cost 35 GB — and why that is not the model's fault

Ornith-1.5-9B is a **dense full-attention** transformer. Every layer keeps a KV cache,
forever:

```
2 (K+V) × 32 layers × 8 kv_heads × 128 head_dim × 2 bytes = 128 KB per token
262,144 tokens × 128 KB                                     = 32 GiB
```

The window was never the expensive part. **The attention is.** A 9B dense model pays
128 KB of cache for every token it has ever seen, so context is priced per-token by
architecture and there is nothing to tune.

**1M on that model would be 122 GiB of cache before a single weight loads.** No quant,
no offload, no driver flag fixes a number that size.

## 2. The only lever that works: fewer layers that keep a cache

Everything that reaches 1M in small memory does the same thing — it reduces **how many
layers hold a growing cache**, not how many tokens you feed.

| model | layers with a real KV cache | KV / token | 1M cache | RULER @ 1M |
|---|---|---|---|---|
| Qwen3-8B (dense) | 36 of 36 | 144 KB | **137 GiB** | OOM — cannot run |
| Ornith-1.5-9B (dense) | 32 of 32 | 128 KB | **122 GiB** | not runnable |
| Granite-4.1-3B (dense) | 40 of 40 | 80 KB | **76 GiB** | not runnable |
| **Nemotron-3-Nano-30B-A3B** | **6 of 52** (11.5%) | **6 KB** | **5.7 GiB** (1.4 q4) | **86.3** |

Nemotron's 46 other layers are **Mamba-2**: a fixed 46.8 MiB recurrent state no matter how
long the context. Only 6 layers use grouped-query attention (2 KV heads × 128 dim), which
is why the cache is **24× smaller per token than Qwen3-8B**.

> 5.7 GiB of bf16 cache at one million tokens. That is smaller than the q4 weights of a
> 1.5B model. The 1M problem was never solved by more VRAM; it was solved by deleting
> 46 attention layers and putting Mamba in their place.

## 3. The model that actually does it

`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` — Apache-ish open weights, 30B total, **3.5B
active**, 128 experts (6 active), 25T pretraining tokens.

| benchmark | score |
|---|---|
| RULER-100 @ 256k | 92.9 |
| RULER-100 @ 512k | 91.3 |
| **RULER-100 @ 1M** | **86.3** |
| MMLU (IQ4_XS, measured) | 77.5 |
| HumanEval pass@1 (IQ4_XS) | 70.1 |
| GPQA Diamond | 73.0 |
| LiveCodeBench v6 | 68.2 |
| SWE-bench Verified | 38.8 |

**86.3 at one million tokens is a measured long-context score, not a card claim.** For
comparison the dense hybrid of the previous generation (Nemotron-Nano-12B-v2) collapses to
**23.4** at 1M — the MoE+Mamba layout is what survives extrapolation, because Mamba layers
carry position implicitly and the attention layers use **no RoPE at all**, so there is no
rotary OOD cliff to fall off.

Caveat to keep attached: it was **continue-pretrained to 512K**, and 1M is extrapolation
that degrades gracefully rather than trained behaviour. 91.3 → 86.3 is the honest cost of
the last doubling.

## 4. Fitting it in 12 GB

1M of cache is only 1.43 GiB at q4 KV. The *weights* are the problem — 30B is 22 GiB at
Q4_K_M. So the card holds the cache and the hot path, and the experts spill to system RAM:

```
12 GB card, ~11.3 GiB usable
  KV cache, 1M @ q4_0        1.43 GiB
  compute buffers            1.2-2.3 GiB
  --------------------------------
  weights resident           7.6-8.6 GiB
  experts in system RAM     13.4-14.4 GiB   (of 22 GiB Q4_K_M)
```

| quant | weights | resident in VRAM | spills to RAM | verdict |
|---|---|---|---|---|
| IQ2 | 8.0 G | all | 0 | fastest, quality loss |
| Q3 | 13.0 G | ~9.4 G | 4.4 G | middle |
| **IQ4 / Q4_K_M** | **17-22 G** | **~8.5 G** | **8-14 G** | **recommended — 77.5 MMLU** |

llama.cpp already runs this (build 7917 was benchmarked on it, Mamba hybrid supported):

```bash
llama-server -m nemotron-3-nano-30b-a3b-IQ4_XS.gguf \
  -ngl 99 -c 1000000 -np 1 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  -ot "exps=CPU"          # keep experts in RAM, attention+Mamba+KV on the card
```

The same command on the Mac (32 GB unified) runs it entirely resident: 22 GiB + 1.4 GiB =
23.4 GiB, fits, no spill, faster.

**Cost of a full 1M ingest: 10-20 minutes of prefill.** Once per repo, not per question.
That is the real price of the window and it is paid once.

## 5. Subagents — and why they are not a model feature

**You cannot add subagents to a model.** There is nothing to train, nothing to download,
no layer to hook. "Subagent" describes how a *program* calls the model: one model, many
separate calls, each with its own fresh context window.

```
orchestrator (1 window)          subagents (N windows, one each)
  holds the plan          ──►      reads files, runs tests, edits
  never reads code                 returns a ~500-token summary
  receives summaries only  ◄──     intermediate work stays here and is thrown away
```

Measured on ProgramBench (NanoHarness, arXiv 2609.32459 — 10 models, 3 benchmarks):

| harness component | delta |
|---|---|
| **task-specific subagents** | **+5.91** |
| tool registry | +4.57 |
| explicit planning | +1.36 |
| lazy skills | +0.81 |
| **general subagents** | **−0.70** |
| **context compression** | **−4.86** |
| all combined | **+7.37** |

Two rules fall straight out of that table and they are not intuitive:

1. **Specialised subagents help; generic ones hurt.** A `reader`, an `editor`, a `tester`
   with narrow instructions and a fixed output shape: good. A "general helper" that can do
   anything: worse than no subagent, because it re-does work the orchestrator already did
   and returns noise.
2. **Context compression is actively harmful** (−4.86). Summarising the conversation to
   stay inside the window loses exactly the detail that made the agent right. Preventing
   overflow by *delegation* is fine; compressing to survive overflow is not.

### The arithmetic that matters

```
1M tokens of code ÷ 22K per subagent window = 45 passes
45 summaries × ~500 tokens                  = 22.5K  -> one 32K orchestrator window
```

**So subagents give you unlimited effective context on a 22K window.** The disk is the
memory; the model only ever sees a slice plus the index. This is how a 12 GB card can work
on a 500-file repository without a 1M window at all.

### Which one you actually want

They solve different halves and compose:

| | 1M window | subagents |
|---|---|---|
| answers | "what is the shape of this whole repo" | "change this function, run the test" |
| cost | 10-20 min prefill, once | one 22K call per unit of work |
| fails when | you ask for precision on a needle at token 900,000 | you ask it to hold the global picture |
| gives you | survey | edits |

**Build both.** Ingest once at 1M to produce the map, then never load 1M again — the
subagents each get one file and the map, not the corpus. That is the whole architecture.

## 6. Build order

```
1. download   nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B  GGUF IQ4_XS (~17 GB)
2. verify     1M runs:  -c 1000000 --cache-type-k q4_0 --cache-type-v q4_0
              measure prefill on a real 200K file, record tok/s. Nothing below
              is worth doing before this number exists.
3. RULER spot-check at 256K / 512K / 1M on our own needles - trust the 86.3
              only after reproducing the shape of the curve.
4. harness    orchestrator + 3 task-specific subagents (reader / editor / tester),
              disk-backed state at .agent/, NO context compression.
5. measure    same task, with and without subagents, on this model. NanoHarness's
              +5.91 is on other models; ours is unmeasured until we run it.
```

## 7. What this does not give you

- **Not frontier coding.** SWE-bench 38.8 vs 73.4 for Qwen3.6-35B-A3B. Nemotron buys you
  the window; it does not buy you the coder. Those are still two different models.
- **Not 1M of precision.** 86.3 on RULER means it finds most needles, not all. For a
  specific identifier 900K tokens back, a subagent that greps the file is still better.
- **Not free at 12 GB.** Expert offload costs decode speed. Measure it before believing any
  tok/s figure, including the ones in this file.

---

## 8. "Can we just rope-scale Granite 4.2 to 1M?" — no, and here is the exact number

`ibm-granite/granite-4.2-8b` `config.json`, read 2026-11-06:

```
max_position_embeddings  131,072
rope_scaling             null
rope_parameters          { rope_theta: 10000000, rope_type: "default" }
num_hidden_layers        40   ·  num_key_value_heads 8  ·  head_dim 128
```

**The flag exists and works.** Verified on the installed build (`llama-server 0.9.0`, build 12110):

```
--rope-scaling {none,linear,yarn}
--rope-scale N          expand context by a factor of N
--yarn-orig-ctx N       the context the model was trained at
```

So the invocation is:

```bash
llama-server -m granite-4.2-8b-Q4_K_M.gguf -c 1048576 \
  --rope-scaling yarn --rope-scale 8 --yarn-orig-ctx 131072
```

`131,072 x 8 = 1,048,576`. The flag will be accepted. **It still will not run, and the reason is
arithmetic, not configuration:**

```
Granite 8B KV = 2 x 40 layers x 8 kv_heads x 128 dim x 2 bytes = 160 KiB/token

1M tokens x 160 KiB = 160 GiB   (bf16)
                    =  80 GiB   (q8 cache)
                    =  40 GiB   (q4 cache)   <- before a single weight loads
```

| | bytes/token | 1M cache | verdict on 24 GB wired |
|---|---:|---:|---|
| Granite 4.2 8B, bf16 | 160 KiB | 160 GiB | impossible |
| Granite 4.2 8B, q4 KV | 40 KiB | 40 GiB | **impossible — cache alone exceeds the machine** |
| **MiniCPM-SALA 9B, bf16** | **8 KiB** | **8.6 GiB** | fits comfortably |
| **Nemotron-30B-A3B, bf16** | **6 KiB** | **5.7 GiB** | fits |

**RoPE scaling changes where the model thinks it is. It does not change how much it remembers.**
Those are separate costs and only one of them is a flag.

### The ceiling you actually get on Granite 4.2 8B

```
24 GB wired  -  4.8 GB (Q4 weights)  =  19.2 GB for cache
19.2 GB / 40 KiB per token           ≈  500K tokens    theoretical
                                     ≈  350-450K       realistic, with headroom
```

So a `-c` between 393216 (384K) and 458752 (448K) is the honest target, at q4 cache, with YaRN
factor 3-4. That is a real, usable window and it is roughly **7x the native 131K** — genuinely
useful, and not 1M.

### Two measured warnings before trying it

1. **llama.cpp's RoPE path past native is not optimized.** A 24 GB laptop test on a 256K-native
   model recorded decode at **87.2 tok/s native, 3.1 tok/s at 512K with YaRN, 4.7 tok/s at 1M**.
   The window opens; the speed collapses. Qwen's own deployment guide does not list llama.cpp
   as a supported YaRN framework.
2. **Extension crashes below 2x occasionally.** A reported llama.cpp failure died silently at
   ~520K prefill with YaRN x4 on a hybrid model — just under 2x the original context. YaRN x2
   completed cleanly on the same setup. Test at the length you intend to use, not at the length
   you hope for.

**Also required: `--override-kv`.** llama.cpp silently caps `-c` at whatever the GGUF metadata
claims:

```bash
--override-kv granite.context_length=int:1048576
```

Without it you will believe you have 1M and have 131K.

### The three ways to actually get 1M on a Granite, ranked

| route | cost | verdict |
|---|---|---|
| RoPE scaling only | one flag | ~7x is real, 8x is arithmetic denial |
| Retrain with fewer cache-keeping layers | a pretraining run | the only real fix — and it is what Qwen3.5, Nemotron and SALA all did |
| **Don't hold 1M** — agent + grep + subagents | free | §5 of this file already says this |

A fourth, dishonest option: run `-c 1048576` with `--override-kv` and no rope flag. llama.cpp
will accept it, print one warning, and quietly run the model at positions it has never seen.
That is how "the model got dumb" reports are made. If you do not see
`custom YaRN scaling detected, re-adjusting n_ctx_train` in the load log, the scaling did not
reach the loader.
