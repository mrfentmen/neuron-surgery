# The buzzword pipeline — what each item actually means for Granite 4.1 3B

Written 2026-11-06. **Scope change:** the neuron sweep and this are now two separate
tracks. The sweep (`layer_sweep.py`, `grid_L*`, Kaggle lane) keeps mapping the original
checkpoint. The lab work happens on the **copy**, `models/granite-4.1-3b-lab/`, and it
is post-training, not steering.

```
models/granite-4.1-3b-base/   6.3 GB   the map's model, untouched, read-only
models/granite-4.1-3b-lab/    6.3 GB   APFS clone, shares blocks, this is the lab animal
```

Hardware ceiling, computed from the config (40 layers · 2560 hidden · 8192 MLP · 100,352 vocab):

```
params            3.82 B      (3.56 B body + 0.26 B tied embedding)
bf16 weights      7.6 GB
full fine-tune    ~46 GB      weights + grads + 2 fp32 Adam moments  -> DOES NOT FIT
LoRA peak         ~10 GB      -> fits, this is the only training path on this machine
4-bit             ~2.1 GB
```

---

## Bucket A — doable on this machine

| technique | what it really is here | instrument | verdict |
|---|---|---|---|
| **Distillation** | *Data* distillation: SFT on teacher traces. True KD (matching teacher logits) needs the teacher's weights or API logprobs — we have neither, so every "distilled" dataset here is just good SFT data. | the 2.25 M rows already on disk | Do this first. It's the whole SFT stage. |
| **SFT** | LoRA r=16 on all linear layers, Granite chat template, assistant-turn loss only. | `mlx-lm` (native Apple Silicon) or `peft`+`trl` | The main event. Nothing installed yet. |
| **CoT** | Granite's template has **no thinking channel** — the only special tokens are `<\|start_of_text\|>`, `<\|end_of_text\|>`, `<\|start_of_role\|>`, `<\|end_of_role\|>`, `<\|tool_call\|>`. So chain-of-thought is plain prose inside the assistant turn, and if you want it separable you have to *teach* a marker. | data formatting | A decision, not an install. Pick `<\|think\|>` tokens reserved or use plain prose. |
| **Abliteration** | Heretic: diff-of-means refusal direction, ablation kernel over layers, Optuna co-minimising refusals vs KL. | `heretic-llm` 2.0 | Do this **after** SFT, not before. |
| **Quantization** | 4-bit/8-bit for local inference; GGUF for llama.cpp, MLX for the Apple stack. | `mlx-lm` convert + `llama.cpp` quantize | The shipping format. Cheap, last step. |
| **Speculative decoding** | Needs a small draft model in the same tokenizer family plus a target. Helps tokens/sec, changes nothing about the weights. | `mlx-lm` supports it | Optional. Only if generation speed becomes the bottleneck. |
| **Activation steering / forward pre-hook / sparsity** | This is the neuron work. Same hook site (`mlp.down_proj` input), but now applied on top of a trained checkpoint rather than a stock one. | existing `neuron_surgery.py` | Track 2, unchanged. |

## Bucket B — needs a rented GPU

| technique | why it can't live here |
|---|---|
| **GRPO** | Needs group rollouts (8+ generations per prompt) plus reward computation per step. Batch-1 on MPS makes this days per run. It fits a Kaggle P100 session with fp16; it does not fit a laptop afternoon. |
| **RLOO** | Same shape as GRPO, fewer moving parts. Same verdict. |
| **True KD** | Requires teacher logits — i.e. an API that returns them or the teacher on disk. Neither exists in this repo. |

## Bucket C — cannot be done to this checkpoint at all

| item | why |
|---|---|
| **MoE** | **CORRECTED 2026-11-06 — this entry was wrong.** Dense→MoE conversion is real and published. Three routes: (a) **MoEfication** (`thunlp/MoEfication`, ACL 2022) — balanced-k-means cluster FFN neurons by *co-activation*, train a small MLP router, **no weight training at all**; keeps >95% of original performance and cuts FFN FLOPs ~2x while activating 25%. (b) **sparse upcycling** (Google, ICLR 2023) — copy the FFN into experts, continue pretraining at +20-100% of the original dense compute. (c) **LLaMA-MoE** (EMNLP 2024) — split + 200B tokens. Only (a) is free. **But it buys speed, not context and not knowledge:** attention layers and the KV cache are untouched, so it does nothing for the 1M problem, and without (b)/(c) it does not raise quality. |
| **MTP** | Extra prediction heads trained jointly during pretraining with a specific schedule. Not addable after the fact. |
| **FlashAttention** | A CUDA kernel. There is no CUDA here; PyTorch's SDPA already picks the memory-efficient path on MPS. Nothing to install, nothing to gain. |
| **Unified memory** | A property of the hardware, not a technique. It's why 7.6 GB of weights can sit next to the OS at all. |

---

## What kind of model is this — and which kinds a 3B can become

"Vision model", "reasoning model", "coding model" are not flavours of the same thing. Each is
a different **training objective** or a different **input/output shape**, and that single fact
decides whether post-training can produce it.

| kind | what actually makes it that | addable to a 3B here? |
|---|---|---|
| **reasoning** | RLVR against a checker + long CoT traces | **yes** — SFT first, RLVR on Kaggle |
| **coding (chat)** | SFT on instruction/code pairs | **yes**, but it is imitation, not skill |
| **coding (native FIM)** | fill-in-the-middle is a *pretraining* objective — the suffix is moved in front of the prefix during pretraining | **no** — a base model trained without FIM never learned the task |
| **agentic / tool use** | SFT on traces containing tool calls and their results | **yes** — this is formatting, and formatting is exactly where 3Bs fail |
| **domain expert** (legal, medical, security) | SFT + retrieval over the domain corpus | **yes**, as style and vocabulary — **not** as knowledge |
| **embedding model** | bi-encoder architecture, contrastive loss | **no** — a causal LM is the wrong shape |
| **reranker** | cross-encoder, scores a pair jointly | **no** — same reason |
| **reward model / judge** | scalar head on a preference pair | **no** — needs a new head and preference data |
| **guardrail / classifier** | small discriminative model | **no** — a GPT-2-class model does this better and cheaper |
| **vision / video / audio in** | modality encoder + projector, trained on paired data | **no** — needs a pretraining stage, not post-training |
| **image / video / speech generation** | diffusion or token decoder, entirely different objective | **no** |
| **time-series / tabular** | numeric tokenisation, often no LM at all | **no** |
| **translation** | SFT on parallel text | **partly** — acceptable after SFT, worse than a dedicated model |
| **draft model** (speculative decoding) | deliberately small, same tokenizer, same family | **no** — but you don't need to make one; a Granite sibling works |
| **MoE** | architectural, decided before pretraining | **only** via MoEfication — free, buys speed not quality |
| **MTP** | extra prediction heads trained during pretraining | **no** |
| **1M context** | position scheme + attention, decided at pretraining | **no** — **131,072** is this checkpoint's ceiling (`max_position_embeddings`) |
| **long-context via SSM/hybrid** | recurrence instead of KV cache | **no** — different architecture, different weights |

Two lines worth carrying out of that table:

1. **Every "yes" is a formatting, style or vocabulary problem. Every "no" is a different
   objective used during pretraining.** Post-training can *select* behaviour; it cannot
   install a new input shape or a new memory mechanism.
2. **Each "yes" is a data problem, not a training problem.** You are not making the 3B
   smarter — you are stopping it from wasting the intelligence it has on the wrong format.
   That is the 95%-correct / 42%-success result in `OPTIMIZATION.md` §3.

---

## Order of operations

```
1. format the data         instruction/response -> Granite chat template, assistant loss only
2. LoRA SFT                small set first (817-2,000 rows), measure tokens/s, then scale
3. merge adapter           write a full checkpoint
4. abliterate the merged    Heretic, on the SFT'd model, not the original
5. quantize                 MLX 4-bit + GGUF
6. evaluate                 skill battery + refusal rate + KL vs the merged model
7. (track 2) steer          the neuron hook, on whichever checkpoint is current
```

---

## The tool that executes this list: Soup (`MakazhanAlpamys/Soup`)

Verified 2026-11-06. `github.com/MakazhanAlpamys/Soup` — Apache-2.0, ~10.2k stars,
2,364 commits, PyPI `soup-cli` v0.77.2. "Fine-tune LLMs from one YAML."

It is not a tutorial repo. It implements steps 1-7 above as CLI commands.

**Why it matters here specifically: Apple Silicon is a first-class backend.**

| our step | Soup command |
|---|---|
| 1. format data | `soup data inspect` / auto-detects alpaca, sharegpt, chatml, DPO pairs |
| 2. LoRA SFT | `soup train --config soup.yaml` with `backend: mlx` |
| 3. merge | `soup merge --adapter ./output` |
| 4. abliterate | not Soup's job — Heretic still runs on the merged checkpoint |
| 5. quantize | `soup export --format gguf --quant q4_k_m` |
| 6. evaluate | `soup eval benchmark`, `soup ship` verdict, calibration, A/B |
| 7. steer | Soup has a `steering` module — **do not point it at the base checkpoint** |

`soup profile --model <id>` answers "will it fit?" from config.json, and on Apple Silicon
it compares against **Metal's working set** rather than a CUDA VRAM pool — which is the
right model for this machine, and the one every other trainer gets wrong.

### The layer-streaming claim (and how honest it is)

Keeps the frozen base out of VRAM and feeds one decoder layer at a time. Measured: Llama-3.1-8B
NF4 + LoRA, **batch 1, seq 512, 3.32 GB peak** on a 4 GB RTX 3050. BETA, opt-in
(`stream_layers: true`).

The paper (Zenodo, CC-BY-4.0, v3 Aug 2026) is unusually disciplined for a tool repo:

- correctness is **bit-exactness against a resident reference of the same numerics** — 9
  architecture families, 14/14 comparisons at max logit difference **0.000e+00**
- it caught a PEFT dispatch path producing a **0.94 logit divergence** with byte-identical
  weights, no crash, and a healthy-looking loss curve — the exact silent failure mode that
  a "loss went down" check cannot see
- it **discloses and repairs** a wrong-gradient defect above ~165 MiB per NF4 layer
- it **retracts its own v2 explanation** after measuring it (host-to-device transfer was
  1.44% of the step) rather than quietly editing the claim
- it reports the unflattering result that **8 × ZeRO-3 cards are slower than 1 card resident**
- the H100 replication (113.00 tok/s, same 3.32 GB) is *no faster than the laptop*, and the
  author says so instead of hiding it

That last point matters for planning: **do not expect a datacenter card to rescue throughput.**
The constraint is common to both machines.

### Verified blocker on this machine

```
this venv      Python 3.14.8, torch 2.15.0, transformers 5.22.1, MPS true
Soup requires  Python >=3.11,<3.14
pip resolved   soup-cli 0.72.4 (an OLD version) and would DOWNGRADE transformers to 4.57.6
```

3.14 is one minor too new. pip silently back-resolved to a months-old release *and* would
have pulled transformers backwards, which would break the neuron tooling that is pinned to
5.22.1. **Do not install this into `.venv`.**

```bash
# separate interpreter, separate environment - never inside the project venv
uv python install 3.13
uv tool install --python 3.13 "soup-cli[mlx]"
```

**`[mlx]`, not `[train]`.** On Apple Silicon the PyTorch path needs bitsandbytes for 4-bit,
which is CUDA-only; MLX is the working 4-bit path here. On v0.77.x both extras share
`transformers>=5.16.1,<6`, which is compatible with the 5.22.1 already in use — the `<5.0.0`
pin seen in the dry run belongs to the old 0.72.4 that Python 3.14 forced.

MLX backend supports: SFT, DPO, ORPO, SimPO, KTO, GRPO. Measured on an M4 Pro: SimPO over
600 UltraFeedback pairs in ~4.5 min at 7.5-8.6 GB; DPO with `gradient_checkpointing` on
792-token pairs went 10.05 GB -> 3.63 GB.

Two cautions: `soup push --hub modelscope|modelers` on versions 0.53.10-0.75.0 required a
credential rotation (we are installing 0.77.2, unaffected), and unknown config keys
**hard-fail** since v0.75, so a config copied from an older example will refuse to load.

**The invariant holds:** Soup trains `models/granite-4.1-3b-lab/`. It never gets pointed at
`models/granite-4.1-3b-base/`, which is what the 25,843 measured rows are keyed to.

**Why abliterate last, not first.** SFT on aligned data re-installs the refusal behaviour
you just ablated — you pay for the removal twice. And Heretic's KL term is measured
against whatever base you hand it, so ablating the *final* model is the run that keeps
the thing you actually built. The one case for ablating first: if you want to SFT on
data the stock model refuses to imitate, then you need compliance before you need
quality.

---

## What's installed

```
transformers 5.22.1   torch 2.15.0   datasets   <- already here
mlx  mlx-lm  peft  trl  accelerate  bitsandbytes  gguf  llama.cpp  ollama
                                                    <- NONE installed
python 3.14 (mlx-lm and peft both publish wheels; check before pinning)
```

## First three runs, in order

1. **50-step timing run** on 817 LIMO rows. Records tokens/s and peak memory. No
   estimate in this document is a measurement until this exists.
2. **One full epoch** on the same 817 rows at the measured rate.
3. **Evaluate** on the skill battery before touching anything else. A LoRA that
   costs accuracy is worse than no LoRA.

Nothing in bucket B or C gets scheduled until 1–3 are done and measured.

---

## How RL actually enters this recipe

Four different things get called "RL". The cost lives entirely in **where the training signal
comes from** — which is why only two of them fit this machine.

| method | where the signal comes from | fits here? | measurement |
|---|---|---|---|
| **SFT** | a human or a bigger model already wrote the good answer | yes | not RL at all — it is the base case |
| **DPO / ORPO / SimPO / KTO** | a **pair**: this answer is better than that one | **yes — the Mac** | SimPO, 600 UltraFeedback pairs, ~4.5 min at 7.5–8.6 GB (M4 Pro, Soup's docs). DPO + `gradient_checkpointing` on 792-token pairs: **10.05 GB → 3.63 GB** |
| **GRPO / RLOO** | a **checker** scores 8+ sampled answers to the same prompt | **no — Kaggle fp16 session** | needs group rollouts every step; batch-1 on MPS is days per run |
| **PPO** | a separate critic model estimates the value | no, and don't | two models resident; most expensive option on the list |

**The distinction that decides it.** DPO never generates anything during training — it reads
finished pairs and pushes probability toward the chosen one. That is why it fits in 10 GB.
GRPO *must* generate, eight or more times per prompt, every step. That is why it does not.

**GRPO needs a checker, not a critic.** Dropping the critic model and scoring samples directly
is the actual contribution of GRPO and RLOO. "Better than the average of its own eight
attempts" is a reward you can compute with `==` on a number, or `pytest` on a repo. It is not
a reward you can compute for "is this answer nice" — which is precisely why RLVR lives in math
and code and nowhere else.

**Process beats outcome on small models.** Qwen2.5-0.5B, GSM8K: process-only rewards **63.73%**
vs outcome-only **53.75%**. At 3B: **+8–10 pp**, plus **+14 pp** on correctly saying "I don't
know" (`arXiv 2607.02869`). Rewarding the *steps* is worth more than ten times the sampling
budget at this size.

**What RL cannot do here.** You cannot RL a model into being quietly correct about facts.
There is no checker for a fact. Every RLVR result in the literature is anchored to something
mechanically verifiable — a number, a unit test, a proof term. For everything else the honest
instrument is `OPTIMIZATION.md` §6: give the model a tool and let the tool hold the exactness.

**So the order is fixed:**

```
1. DPO / SimPO on a few hundred pairs    cheapest real alignment step, runs tonight
2. GRPO on Kaggle, fp16, --max-seconds  only if a checker exists for the task
3. PPO                                  never, on this hardware
```
