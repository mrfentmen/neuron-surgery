# Model verdicts — what is on disk, and why**State 2026-11-07: `models/` is 6.3 GB — `granite-4.1-3b-base` only.**

`nanbeige`, `nanbeige-bf16`, `nanbeige-dspark` (11.8 GB) and `CHIMERA.md` were deleted on request
2026-11-07. Qwen3.5-9B and Qwen3.6-35B-A3B went the day before. **Every section below is the
reasoning trail, not a live recommendation** — read it as *"why that model was right while we had
it"*. SALA was deleted earlier; the rule that produced it is still below.

---

## The thermal ceiling — added 2026-11-07

New hard constraint, in the owner's words: **"it needs to not blow up my computer."** Granite is
the only model measured to hold **45 tok/s while staying cool** on this M1 Max.

The number that predicts heat is **not tok/s**. It is (a) bytes moved per token and (b) quadratic
prefill work. A 2 GB model at 45 tok/s moves ~90 GB/s and stays cool. A 20 GB model at 25 tok/s
moves about the same and sits at the memory ceiling, where macOS starts compressing and swapping —
that is where the heat actually comes from, not from the token rate.

Two rules follow, both stricter than anything above:

1. **Resident footprint <= ~12 GB.** This machine has already been killed once at 8.9 GB of
   combined RSS (`STATUS.md`, 01:50). A 20.6 GB model plus a 2.5 GB cache is 23.1 GB against a
   24 GB wired ceiling — that is not "fits", that is one browser tab from a swap storm.
2. **Prefill is the heater, not decode.** Measured: 275 tok/s at 6.5K, 116 tok/s at 19.4K. Long
   prompts are a compute job across every attention layer, and that is what spins the fans.

Measured cause, not theory: `STATUS.md` 01:50 found **Chrome at 62.8% CPU / 6.3 GB was the top
consumer, not the sweep** — and this file already records that *downloading while benchmarking* is
what made the fans spin. Thermal control here is mostly a browser problem.

### Revised pick under the thermal ceiling

| role | model | resident | verdict |
|---|---|---:|---|
| workhorse + the map | **granite-4.1-3b-base** | 2.0 G @ Q4 | the only one measured at 45 tok/s, cool. 131K max, 80 KiB/token — **cannot do 1M** |
| 1M reader | **Qwen3.5-4B** | 2.5 G weights, ~10 G at 1M @ q4 | 8 of 32 layers keep a cache. The only 1M model that stays inside a cool footprint |
| ~~Qwen3.6-35B-A3B~~ | — | 23.1 G @ 262K | **rejected on thermal grounds, not quality.** SWE 73.4 is real; 23 GB resident is the failure mode that already killed this machine once |
| ~~Nemotron-3-Nano-30B-A3B~~ | — | 16-23 G | rejected for the same reason. 6 KiB/token is the best cache in the field and still needs 22 GB resident at Q4 |

> **Frontier coding + cool + local does not exist at 24 GB.** Those are three constraints and the
> machine satisfies two. If frontier code is genuinely required, that is an API call, not a
> download. The harness argument still holds: `OPTIMIZATION.md` §1 measures the scaffold at
> **+19.4 pts** against the model's **+22.8** — a cool 3B in a good harness beats a hot 35B that
> is swapping.

---

## The rule, in one line

> **If the model needs `trust_remote_code` to load, it will not run on this Mac. Ever.**

SALA failed for exactly one reason. Its 24 Lightning Attention layers call
`fla.ops.simple_gla` — Triton, written for NVIDIA, imported at module top level with no fallback.
Its 8 real-attention layers call `infllm_v2` and `flash_attn` — also NVIDIA. Everything else about
the model was fine. The architecture is good. **The kernels are the problem.**

`fla` is the same library the Gated DeltaNet and KDA layers in Qwen3-Next, Qwen3.5 and Kimi Linear
use — confirmed publicly by NVIDIA's own linear-attention lead. So "pick a linear-attention model"
does not save you. What saves you is:

> **llama.cpp / MLX implementing that architecture in C.**

That is the whole test. A model with a llama.cpp port gets Metal, quantized KV cache, and full
speed. A model without one gets a port project.

---

## The candidates, screened

| model | KV/token | 1M cache | llama.cpp | verdict |
|---|---:|---:|---|---|
| **Qwen3.5-9B** | **32 KiB** | **8.0 G @ q4 KV** | **yes — `src/models/qwen35.cpp`** | **pick** |
| Qwen3.5-35B-A3B | 20 KiB | 9.6 G @ q8 | yes | weights too heavy at Q4 (19 G) |
| Nemotron-3.5-Bolt-74B-A9B | 8 KiB | 8 G | yes (24 GB Metal build) | 74B at IQ2 — quality gamble |
| Kimi-Linear-48B-A3B | 243 KiB | — | **PR branch only, not main** | out |
| Falcon-H1-7B | 44 KiB | 42 G | partial | 1M impossible |
| Granite 4.1 8B | 160 KiB | 152 G | yes | 1M impossible, stays as *training* target |
| ~~MiniCPM-SALA~~ | 8 KiB | 8 G | **no** | **deleted** |

The 1M column is the number that matters. Granite is 20x Qwen3.5 per token and that alone
disqualifies it from being the runner, no matter how good the weights are.

---

## Qwen3.5-9B, from its real config

Read out of the actual `config.json` on 2026-11-06, not from a blog post:

```
num_hidden_layers   32          layer_types: 8 full_attention + 24 linear_attention
num_attention_heads 16          num_key_value_heads 4      head_dim 256
full_attention_interval 4       hidden_size 4096           vocab 248320
max_position_embeddings 262,144 -> ~1,010,000 with YaRN factor 4.0
```

```
KV/token = 2 x 8 layers x 4 kv_heads x 256 head_dim x 2 bytes = 32,768 B = 32 KiB
```

Only 8 of 32 layers keep a cache — the other 24 are Gated DeltaNet with a fixed-size state.
Same trick as SALA, implemented in C instead of Triton.

Weights are not an estimate either: `Qwen3.5-9B-Q4_K_M.gguf` is **5.29 GB** on disk.

| context | f16 KV | q8 KV | q4 KV | **total @ q4 KV** | fits 24 G? |
|---:|---:|---:|---:|---:|---|
| 64K | 2.0 G | 1.0 G | 0.5 G | **5.8 G** | yes |
| 262K native | 8.0 G | 4.0 G | 2.0 G | **7.3 G** (13.3 @ f16) | yes |
| **1M extended** | 32.0 G | 16.0 G | 8.0 G | **13.3 G** (21.3 @ q8) | **yes, at q4** |

**13.3 GB at one million tokens.** On a 24 GB wired budget that is not tight. At f16 KV the 1M case
is 37.3 GB and simply does not run — the KV cache has to be quantized, which is a llama.cpp flag,
not a code change.

Independently measured on the same class of hardware (oMLX, M5 32 GB, MLX 4-bit):

```
context   prefill tok/s   decode tok/s   peak mem
   1k         724.7          34.4          7.7 G
  16k         647.0          28.9          9.4 G
  64k         246.3           9.9         13.9 G
```

An M1 Max will be slower than an M5 and the memory curve will be similar.

### It codes, which SALA very much did not

| benchmark | Qwen3.5-9B | SALA 9.5B |
|---|---:|---:|
| **LiveCodeBench v6** | **82.7** | 52.0 |
| HumanEval+ | 89.6 | — |
| MBPP+ | 69.8 | 89.1 (MBPP, easier) |
| BFCL-V4 (tool calling) | 49.7 | not published |
| TAU2-Bench (agentic) | 57.4 | not published |
| AA-LCR (long context) | 50.7 | — |
| LongBench v2 | 48.2 | — |

**LiveCodeBench v6 82.7 against 52.0** is the whole argument. LCB is fresh, uncontaminated
competitive programming; it is the row that still carries signal. A 9B scoring 82.7 is genuinely
frontier-adjacent for its size.

### The caveat that must travel with the 1M number

**1M is supported, not solved.** One independent aggregator puts Qwen3.5-9B at **Context Arena
AUC@1M = 3.2%**, up from 19.6% at 128K. It will *accept* a million tokens. It will not *reason*
across a million tokens. Same distinction as RULER-vs-NoLiMa for SALA: retrieval context and
reasoning context are different products.

Plan on **262K being the real working window** and 1M being an emergency overflow.

---

## How to run it

**Nothing needs to be installed or built.** Verified 2026-11-06:

```
installed binary   llama-server 0.9.0 (build 12110, commit cd959bae9)
upstream latest    v0.9.0, published 2026-11-04
src/models/qwen35.cpp   present at tag v0.8.0 AND v0.9.0
```

The installed build is the newest release. It already contains the Qwen3.5 graph in C.

```bash
# 5.29 GB, from the repo whose file sizes were checked rather than assumed
hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir models/qwen35

# 262K — the honest working window, comfortable at 13.3 GB
llama-server -m models/qwen35/Qwen3.5-9B-Q4_K_M.gguf \
  -c 262144 --cache-type-k q8_0 --cache-type-v q8_0 \
  --flash-attn on --jinja --port 8080

# 1M — possible, but you MUST quantize the cache or it will OOM
llama-server -m models/qwen35/Qwen3.5-9B-Q4_K_M.gguf \
  -c 1048576 --cache-type-k q4_0 --cache-type-v q4_0 \
  --rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144 \
  --flash-attn on --jinja --port 8080
```

MLX is the faster path on Apple Silicon but **`mlx` and `mlx-lm` are not installed on this venv** —
and this venv is Python 3.14, pinned to transformers 5.22.1 for the neuron tooling. Do not install
MLX into it. Separate 3.13 venv, exactly like the Soup situation.

---

## Measured on this machine — 2026-11-06

Not estimated. `llama-server 0.9.0`, `Qwen3.5-9B-Q4_K_M.gguf`, `-ngl 99`, `-c 8192`, M1 Max.

```
load time          8 s        (5.3 GB from disk, fully offloaded to Metal)
decode            34.7 tok/s  (220 tokens)
decode            29.6 tok/s  (60 tokens, short)
prompt eval      119.1 tok/s  (26 tokens — short prompts undersell it)
```

It answers, and the answer is correct code — verified, not assumed:

```python
# request: "k most frequent words, ignoring stopwords, one sentence then code"
# response: 306 tokens, correct Counter + stopword filter + tie-break, finished cleanly
```

### THE GOTCHA: thinking is on by default and it will eat your whole budget

First test, `max_tokens: 900`, returned **`content: ""`** with `finish_reason: length`.
Not a broken model — all 900 tokens went into `reasoning_content` (3,617 chars of
"Thinking Process: 1. **Analyze the Request**...") and it never emitted visible text.

**Fix that works:**

```json
{"max_tokens": 700,
 "messages": [...],
 "chat_template_kwargs": {"enable_thinking": false}}
```

Result: `reasoning_content` length 0, `finish_reason: stop`, correct answer.

**Fix that does NOT work:** appending `/no_think` to the prompt. Tested — it still produced
222 chars of reasoning and an empty content field. Same prompt, same model, same build.

This matters beyond convenience: any harness that measures "did it answer correctly" will
silently score **zero** on every row if thinking isn't disabled, and the failure looks like the
model is bad rather than the request being wrong. Exactly the same class of silent-wrong as the
SALA port.

---

## Can a bigger model do 1M in 8-10 GB? No. The arithmetic, not an opinion

Computed from the real configs of every Qwen3.5 size that exists (2026-11-06, see `kv_audit.py`):

| model | params | layers | **attention layers** | KV/token | weights @ Q4 | 1M cache @ q4 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-9B | 9.7 B | 32 | **8** | 32 KiB | 5.5 G | 8.0 G |
| Qwen3.5-27B | 27.8 B | 64 | **16** | 64 KiB | 15.7 G | 16.0 G |
| Qwen3.5-35B-A3B | 36.0 B | 40 | **10** | 20 KiB | 20.3 G | 5.0 G |

**Going up in params makes long context strictly worse, on both axes at once.** The 27B has
2.9x the weights *and* 2x the KV per token, because it has 16 attention layers instead of 8.
There is no size where bigger buys you more context. The two numbers move against you together.

What actually fits, per budget (q4 KV):

```
10 GB   9B     -> ~596K tokens     (native caps at 262K anyway)
10 GB   27B    -> weights alone are 15.7 GB. Does not load.
10 GB   35B    -> weights alone are 20.3 GB. Does not load.
16 GB   9B     -> 1,048,576  OK
24 GB   35B-A3B-> ~776K tokens, and it is a much better model
```

**So: 1M at 8-10 GB is impossible for any model in this class,** including the 9B — 1M on the 9B
is 13.5 GB, not 10. The KV cache alone is 8 GB at that length; you cannot spend it and the weights
out of a 10 GB envelope.

And note what the middle row means: **the only model with a better KV/token than the 9B (-37%)
is the 35B-A3B, and its weights are 3.7x larger.** It wins the cache and loses everything else.

**But the 10 GB budget is imaginary on this machine.** It is 32 GB unified with a ~24 GB wired
ceiling. 13.5 GB at 1M is comfortable. The 8-10 GB number was left over from the "12 GB GPU"
frame; on unified memory the real ceiling is 24, and the 9B uses 13.5 of it.

If a bigger model is genuinely wanted, the ladder is:

```
9B  @ 1M      13.5 GB   fits now
35B-A3B @ 776K 24-25 GB  fits now, at the wired edge, ~2x slower
27B           does not reach 1M on this machine at all (24 GB -> 545K)
```

---

## The real answer for agentic coding: Qwen3.6-35B-A3B (not the 3.5)

Found 2026-11-06. This supersedes the 3.5 for the agentic lane.

**It is the same architecture.** `model_type: qwen3_5_moe`, `qwen3_5_moe_text`,
`Qwen3_5MoeForConditionalGeneration`. Identical config to the 3.5: 40 layers, 10 attention,
kv_heads 2, head_dim 256, **20 KiB/token**. So `qwen35moe.cpp` in llama.cpp v0.9.0 loads it
unchanged — there is no `qwen36.cpp` and none is needed. The GGUF arch string is `qwen3_5_moe`.

**It is much better on exactly the rows that matter for building things:**

| benchmark | Qwen3.6-35B-A3B | Qwen3.5-35B-A3B | delta |
|---|---:|---:|---:|
| **SWE-bench Verified** | **73.4** | 70.0 | +3.4 |
| **Terminal-Bench 2.0** | **51.5** | 40.5 | **+11.0** |
| **MCPMark (tool use)** | **37.0** | 27.0 | **+10.0** |
| LiveCodeBench v6 | 80.4 | 74.6 | +5.8 |
| GPQA Diamond | 86.0 | 84.2 | +1.8 |

SWE-bench 73.4 is genuinely near the open-weight frontier. Terminal-Bench and MCPMark are the
agentic rows — those are the two that decide whether a model can *drive a tool loop*, which is what
"build me a game" actually requires.

### The footprint

```
Q4_K_M weights            20.61 GB
262K context, q8 cache     2.50 GB   (20 KiB/token)
                          ---------
at 262K                   23.1 GB    <- right at the 24 GB wired ceiling
at 64K                    21.2 GB    <- comfortable
```

Tight but real. This is the largest model that runs on this machine without swapping.

### Why MoE and not a dense 27B

Measured elsewhere on Apple Silicon, Q4:

```
MoE 30-35B-A3B (3B active)   59-90 tok/s
dense 24B                    ~13 tok/s
dense 27-32B                 7-13 tok/s
```

**A dense 27B is unusable here** — 7-13 tok/s while heating the machine just as hard. Do not buy
"more parameters" in dense form on this hardware. The 3B-active MoE is both faster than the 9B dense
and far more capable, because decode is bandwidth-bound and it reads a third as many bytes per token.

### What it still cannot do

**It does not have frontier *knowledge*.** SWE-bench measures skill at modifying code that is in
front of it. It does not measure knowing every API, library or convention in the world. That lives
in parameter count and this is a 3B active model. Expect it to be strong at the code it can read
and thin on obscure facts — which is what RAG is for (`RAG.md`).

---

## The 10 GB ceiling — settled 2026-11-06

The question was: anything under ~10 GB that is agentic and good at long codebases?

**No. The 9B was the ceiling — and it was downloaded, then deleted 2026-11-06.**

That leaves Nanbeige4.2-3B (38.6 tok/s measured, 262K native, 176 KiB/token) as the only
GGUF runner on disk. Granite 6.3 GB is the research target and is not an inference runner.

Every candidate between 9B and 35B fails on a different axis:

| model | weights Q4 | SWE-bench V | dense? | tok/s here | fits 10 GB? |
|---|---:|---:|---|---:|---|
| **Qwen3.5-9B** | **5.3 G** | ~53 | no (hybrid) | **28 measured** | **yes — 9.3 G @ 262K q8, 7.3 G @ q4** |
| Qwen3.6-27B | 15.7 G | **77.2** | **dense** | ~7-13 | no |
| Qwen3.6-35B-A3B | 20.6 G | 73.4 | MoE | 59-90* | no |
| Nanbeige4.2-3B | ~2.6 G | 63.6 (unverified) | dense | ? | **no — 88 KiB/token** |
| Qwen3-Coder-30B-A3B | 19.4 G | 60.4 | MoE | fast | no, and 96 KiB/token |

\* other Apple Silicon, not this machine.

**Qwen3.6-27B is the best coder that exists at this size** (SWE 77.2, Terminal-Bench 59.3 — it beats
the 397B-A17B flagship). It is also **dense**, which on this hardware means 7-13 tok/s while heating
the machine just as hard. Wrong tool for a silent laptop, right tool for a server.

### Nanbeige4.2-3B — measured, and it is the fastest thing here

I called this one dead on the 88 KiB/token figure. **That was wrong**, and the error was using q8.

**CORRECTION, measured 2026-11-06: it is 176 KiB/token, not 88.**

```
print_info: n_layer      = 44
print_info: n_layer_all  = 44
llama_kv_cache: size = 1408.00 MiB (8192 cells, 44 layers, 4/1 seqs)
                       K (f16) 704 MiB + V (f16) 704 MiB
```

The Looped Transformer runs its 22-layer stack **twice**, so llama.cpp allocates **44 KV slots** and
pays **2x the attention compute** per token. Everything below uses 44 layers. This also explains the
speed: it is not a cheap 3B, it is doing ~8B-class work per token.

```
Q4_K_M weights              2.40 GB
262K cache, q4             11.00 GB
262K total                 13.4 GB   <- MORE than the 9B's 9.3 GB, not less
```

q4 KV cache changes the verdict completely. And `src/models/nanbeige.cpp` **exists in llama.cpp
v0.9.0**, so it loads on the installed binary with no port.

**Measured on this machine, 8K ctx, 300-token answers:**

| | peak | avg | decay |
|---|---:|---:|---:|
| **Nanbeige4.2-3B Q4** | **40.5 tok/s** | **38.6 tok/s** | −9.5% |
| Qwen3.5-9B Q4 | 31.6 tok/s | 28.8 tok/s | −11.7% |

**+34% faster, at a third of the bytes per token (2.4 GB vs 5.3 GB).** If the goal is 30+ tok/s,
this is the only model tested that clears it comfortably.

**The catch is prefill, not decode.** All layers are *full* attention and the loop doubles the pass
count, so attention cost is O(n²) over the whole context at 44 layers. Decode stays fast at any
length; **loading a large context into it is a compute job** and heats the machine far more than
generation does. This is why a "3B" model can spin the fans like an 8B one.

### Measured prefill, and why 1M is not available at any setting

Prefill throughput collapses with length, exactly as O(n²) predicts:

```
  6,529 tokens    23.7 s   275 tok/s
 19,448 tokens   167.8 s   116 tok/s
```

At **1,048,576 tokens** three independent things block it:

| blocker | number |
|---|---|
| **memory** — 1M cache at q4 = 44.0 GB, + 2.35 GB weights = **46.4 GB** | roughly 2x the wired ceiling, at the most aggressive cache quant that exists |
| **no extension** — native `max_position_embeddings` is 262,144, with no YaRN/rope-scaling claim | would need scaling it was never trained on |
| **prefill compute** — 396,316 TFLOP of attention vs 24,770 at 262K | 16x the work at a rate already degrading |

**1M is not a flag you set. It does not fit, it was not trained for it, and the prefill would run
for hours before producing a single token.** The honest ceiling on this machine is ~16-32K, where
prefill is already 24-170 seconds.

Even Qwen3.5-9B, which *does* document 1M via YaRN factor 4 and fits in 13.4 GB, needs 144,115 TFLOP
of prefill to get there. **Memory can be made to fit; the compute cannot.** No model on this laptop
is doing 1M-token context in any useful sense — 262K is already borderline.

**Its scores are lab-reported and unverified** — 63.6 SWE-bench Verified, 72.5 LCB from a 3B. That
would be extraordinary. Treat the number as a claim, not a fact, until it reproduces.

### 10-task objective benchmark — **9/10**

`bench_code10.py`, 2026-11-06, thinking off, temp 0.0, hidden assertions executed in a subprocess.
No human judgement, no partial credit.

```
  [ 1] merge_sorted O(n+m)    PASS   116 tok    3.1s
  [ 2] word_freq              PASS   463 tok   12.6s
  [ 3] is_balanced            PASS   109 tok    3.2s
  [ 4] chunk_ranges           PASS   101 tok    2.9s
  [ 5] parse_duration         FAIL   700 tok   21.0s   <- ValueError: Unknown unit:
  [ 6] rle roundtrip          PASS   565 tok   16.6s
  [ 7] spiral order           PASS   336 tok    9.8s
  [ 8] LRU cache              PASS   177 tok    5.4s
  [ 9] group_anagrams         PASS   186 tok    5.8s
  [10] flatten                PASS   115 tok    3.5s
```

The one failure is a **genuine logic bug, not truncation**. The code is complete and complete-looking;
it tokenises numbers and units into *separate* entries `(num, '')` and `('', unit)` and then never
pairs them back up, so the first number falls through to `raise ValueError(f"Unknown unit: ")` with
an empty unit. It dies on `"1h30m"`, the very first assertion. The source even contains the false
assumption in a comment: *"our tokenization ensures numbers and units alternate"* — which the
code does not actually do.

That is the characteristic 3B failure mode: **over-engineering a simple parse into two passes, then
losing track of the invariant between them.** It reached for a hand-rolled tokeniser where
`re.findall(r'(\d+)([hms])')` finishes the job in one line.

**And one of the nine passes is wrong too — my test missed it.** `word_freq` does
`Counter.most_common(k)` first and *then* sorts by `(-count, word)`. The sort only reorders the k
already chosen; it cannot change **which** k were selected. Verified against the stored code:

```
word_freq('c b a', 2) -> ['b','c']   expected ['a','b']   WRONG
word_freq('z y x', 2) -> ['y','z']   expected ['x','y']   WRONG
```

So the honest tally is **8 clean / 1 passing-but-wrong / 1 outright fail**, not 9 clean.

**The shape of what it can and cannot do:** it is reliable on the algorithmic set-pieces that are
everywhere in training — spiral traversal, LRU via `OrderedDict`, RLE, deep flatten, two-pointer
merge, bracket matching. Its correct answers there are textbook-clean. It breaks on **parsing**,
where it has to invent a small state machine instead of recalling a pattern. That is the line.

It still writes correct code — verified, not assumed. Asked for an O(n+m) merge with no `sort()`
plus pytest coverage:

```python
def merge_sorted(a, b):
    merged = []
    i = j = 0
    len_a, len_b = len(a), len(b)
    while i < len_a and j < len_b:
        if a[i] <= b[j]:
            merged.append(a[i]); i += 1
        else:
            merged.append(b[j]); j += 1
    merged.extend(a[i:])
    merged.extend(b[j:])
    return merged
```

Correct two-pointer merge, stable on duplicates, correct tail handling, and the tests it wrote cover
both the duplicate case and the empty-list case it was asked for. `finish_reason: length` at 450
tokens — it wanted to keep going, which is fine.

**Prefill is genuinely fast at short context:** 296 tok/s on an 82-token prompt. Much faster than the
9B. The O(n²) worry only bites at tens of thousands of tokens.

**It needs `enable_thinking: false`, same as Qwen.** Without the flag: 320 tokens, 1,322 chars of
reasoning, `content: ""`. With it: 0 reasoning chars, clean code. `/no_think` in the prompt is not
a reliable switch on either model.

### Download size vs heat — the distinction that was getting confused

```
20 GB sitting on disk, model not loaded    0 W     no heat at all
20 GB being downloaded                    ~22% CPU hashing, no GPU
downloading WHILE benchmarking            this is what made the fans spin
20 GB resident on a 32 GB Mac             swap pressure -> slowdown AND heat
```

The first two are free. The fourth is real and is why the 35B is genuinely a bad idea on this
machine. Not the file size — the **resident** size, against 24 GB wired.

**The 9B at 262K is 9.3 GB resident with a q8 cache** (4.0 G cache + 5.3 G weights) — 39% of the
24 GB wired ceiling. At a q4 cache it is 7.3 GB. That is the quiet config, and it is already on disk.

---

## What this costs

Qwen3.5 has **100+ derivatives and at least 5 separate abliterations**. It is the opposite of
"untouched". The novelty argument that made SALA attractive is gone.

That trade is the right way round now:

```
the runner needs to WORK        -> Qwen3.5-9B
the experiment needs to be NEW  -> granite-4.1-8b-base, still
```

Two lanes, as already established. Do not try to make one model do both jobs.

---

## Depth-Hybrid-Attention — can it be dropped into the Nanbeige we have? **No.**

`Nanbeige/Depth-Hybrid-Attention` — created 2026-10-29, pushed 2026-11-03, 25 stars, **no license**.
Six Python files. It is a *paper repo*: config classes, an HF model definition, and benchmark tables.

```
code/configuration_nanbeige.py     looped decoder + GQA/SWA + KDA + MLA depth attention
code/configuration_mla_depth.py    Megatron <-> HF config validation and export
code/modeling_nanbeige.py          HF model
code/modeling_mla.py               MLA attention
code/modeling_latent_depth.py      the depth bank / depth router
code/latent_flex_attention.py      flex-attention masks
```

No training script. No checkpoints. No converter.

### Four independent reasons it cannot go into `models/nanbeige`

| # | blocker | evidence |
|---|---|---|
| 1 | **Nothing to put in.** No weights exist. | HF model search for `Depth-Hybrid-Attention`, `Depth Attention`, `DDA nanbeige` → zero results. |
| 2 | **The router is learned.** | `configuration_nanbeige.py:138` `linear_depth_logit_mul_learnable`, plus `depth_attention_stride`, `depth_attention_recent_window`, `window_depth_attention_isolation`. Those tensors are absent from our checkpoint. |
| 3 | **Our copy is a Q4_K_M GGUF.** | A GGUF is a fixed inference graph. llama.cpp has no `depth_attention` op to switch on, and architecture cannot be added after conversion. |
| 4 | **It cannot run here anyway.** | needs `transformers==5.15.0` + `fla-core==0.5.2` (Triton/CUDA). We have 5.22.1, `torch.cuda.is_available() == False`, no `fla`, no `triton`. Same wall that killed SALA. |

### The benchmarks are pretraining ablations, not retrofit results

```
MMLU      46.19 -> 45.51     flat / slightly down
CMMLU     41.77 -> 42.58     +0.8
GSM8K     37.60 -> 44.43     +6.8   <- the result
MMLU-Pro  18.18 -> 18.91     +0.7
```

MMLU-Pro at **18-20%** is consistent with a sub-1B model trained briefly from scratch. Read this as
"two small models trained from token zero, one with depth mixing, compare loss curves." It is evidence
the design is good **at pretraining time**. It is not evidence depth mixing can be added to a finished
model — the router has to be co-trained with the weights whose values it mixes.

### The part that is actually useful

`configuration_nanbeige.py:114` puts `num_loops` **in the same class** as the depth flags. The looped
transformer and the depth bank are one family. Our Nanbeige4.2-3B is **44 layers = 22 x 2 loops**
(measured from `n_layer`, not from the card); DHA is the next step of that exact design — instead of
repeating the stack, mix **across** depth.

So the verdict is the same as for Nanbeige4.3-3B: **wait for a checkpoint, do not graft.** And because
the repo carries **no license**, read it — do not vendor it into this project.

---

## The cache budget — why quant exists, and the 24 KiB/token club (2026-11-06)

Weights and KV cache draw from **one memory pool**. Every GB spent on weights is a GB unavailable for
context. That is the whole reason quantization exists here — not "make it dumber", but "buy context
with precision".

### Nanbeige4.2-3B inside a 12 GB budget (176 KiB/token at bf16)

| weights | size | cache left | ctx @ bf16 cache | ctx @ q8 cache |
|---|---:|---:|---:|---:|
| bf16 | 7.9 G | 4.1 G | 24K | 49K |
| Q8 | 4.0 G | 8.0 G | 48K | 95K |
| Q4_K_M | 2.4 G | 9.6 G | 57K | 114K |

**The 9/10 code benchmark was run on the Q4_K_M file.** 4.85 bits per weight, 57K of context
available, and it still wrote 9 correct programs. Quantization cost nothing measurable on that task —
which is the point: it degrades precise recall and arithmetic long before structure and syntax.

### Granite4.1-3B inside the same budget (80 KiB/token at bf16)

| weights | size | cache left | ctx @ bf16 cache | ctx @ q8 cache |
|---|---:|---:|---:|---:|
| bf16 | 6.3 G | 5.7 G | 74K | **131K (full window)** |
| Q4 | 1.9 G | 10.1 G | **131K (full window)** | **131K** |

Granite reaches its whole advertised window on 12 GB. Nanbeige cannot reach half of its own without
a q4 cache.

### Why nothing small hits 262K in 4-6 GB

262K in 6 GiB means **24 KiB/token**. Measured from live configs 2026-11-06:

| model | params | bf16 weights | KV/token | 262K costs |
|---|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next** | 180 B | 335 G | **24.0 KiB** | **6.0 G** |
| Ling-3.0-tiny | 7.89 B | 14.7 G | 192 KiB* | 48 G |
| Ling-3.0-flash | 127.5 B | 237 G | 672 KiB* | 168 G |
| Kimi-Linear-48B-A3B | 49.1 B | 91.5 G | 243 KiB* | 61 G |
| granite-4.1-3b | 3.40 B | 6.3 G | 80 KiB | 20 G |
| Nanbeige4.2-3B | 4.2 B | 7.9 G | 176 KiB | 44 G |

\* those two configs did not declare `layer_types`, so the parser assumed all-full attention. Treat
them as **upper bounds** — Kimi Linear and Ling are hybrid by name and their true cache is smaller.
Both are disqualified on parameter count anyway.

**The only model that hits 24 KiB/token is 180 B.** That is not luck. Linear attention replaces the KV
cache with a fixed-size recurrent state — the cost moves into parameters instead of memory. Cheap
cache is *bought with weight count*; you cannot have both.

Qwen3.8-Flash-Next's shape is the blueprint: 48 layers, only **12** of them full attention,
`kv_heads=2`, `head_dim=256` → 24 KiB/token. Granite is the best local approximation at 80 KiB.

**Verdict: there is no unquantized 262K-in-4-6GB model for this machine. Granite at bf16 + a q8
cache delivers the full 128K in 11.3 GiB, and that is as close as anything here gets.**

---

## Training base for Nanbeige — what is already built (2026-11-06)

Decision: **Nanbeige4.2-3B becomes the training target**; granite-4.1-3b stays the neuron-research model.

### The blocker that was not obvious

`models/nanbeige/` holds **only** `Nanbeige4.2-3B-Q4_K_M.gguf` and no `.safetensors`. A GGUF is an
inference-only container — **nothing currently on disk can be trained.** Training needs the 8.34 GB
bf16 checkpoint, and it needs `trust_remote_code`, because `transformers` 5.22.1 ships **no**
`nanbeige` architecture (`transformers.models` has zero matches; the checkpoint declares its own
`auto_map`).

Good news that kills the SALA comparison: `modeling_nanbeige.py` is **pure PyTorch**. It imports only
torch + transformers; `flash_attn` is behind `if is_flash_attn_2_available()`, and the one CUDA-only
branch is guarded by `query_states.device.type == "cuda"`. No Triton, no `fla`. So unlike SALA it
will load and train on MPS.

### The repo that saves the most hours

`johnhalloran/Nanbeige4.2-3B-mps-fix` (Apache-2.0, base revision `5d54321e`). Its `MPS_FIX_NOTES.md`
documents **five transformers-compatibility bugs in the stock checkpoint**:

> "crashes hard on Apple Silicon (MPS) after a few generation steps, and — even once that crash is
> patched around — produces structurally incoherent output (looping `</think>` tokens,
> character-level word salad) regardless of device, dtype, or sampling settings."

Its config differs from stock in exactly the ways that matter: `rope_parameters` instead of
`rope_scaling: null` (stock raises `KeyError: 'type'` at construction), and
`transformers_version: 5.8.1`.

**34 downloads. Unofficial and unendorsed.** Treat it as a hypothesis until a smoke test reproduces
clean output — but the notes are specific, cite the upstream revision, and the fixes are checkable.

### Architecture reality check — matters for the buzzword plan

From the same notes, corroborated against Nanbeige's own Go/MLX serving path:

| advertised feature | enabled in this checkpoint? |
|---|---|
| Looped Transformer (22 physical × `num_loops=2` = 44 effective) | **yes** |
| LoopSplit (`enable_double_loop_split`) | **no** |
| mHC / hyper-connections (`enable_mhc`) | **no** |
| depth attention (`enable_depth_attention`) | **no** |
| n-gram embeddings (`emb_neighbor_num`) | **no** |

Of four advertised ideas, **this checkpoint exercises one.** The other three exist as code in
`modeling_nanbeige.py` with no trained weights behind them — the DHA verdict restated: you cannot
switch on a mechanism the weights were never trained with. Nanbeige's own card says those three land
in **Nanbeige4.5** (training underway, later 2026).

### Available starting points

| repo | size | what it saves |
|---|---:|---|
| `johnhalloran/Nanbeige4.2-3B-mps-fix` | 8.34 G bf16 | 5 compatibility bugs + checkpoint-level fixes |
| `Nanbeige/Nanbeige4.2-3B` | 8.34 G bf16 | canonical, Apache-2.0, **already instruct/agentic** (built on `-Base`) |
| `Nanbeige/Nanbeige4.2-3B-Base` | ~8.3 G | only if you want to run your own SFT |
| `WaveCut/Nanbeige4.2-3B-heretic` | 7.80 G bf16 | abliteration done — ARA-LoRA merged `safe_merge=True`, only `attn.o_proj` + `mlp.down_proj` changed, validated tensor-by-tensor |
| `Nanbeige/Nanbeige4.2-3B-DSpark` | 1.60 G | **0.85B official speculative draft** → decode speed, no quality change |
| `Nanbeige/Nanbeige4.3-33B` | **61.7 G bf16** | the 4.3 teacher — **not runnable here** |

### Geometry, confirmed from the modelling code

`head_dim = 128` and it is **non-square**: 3072/48 = 64 ≠ 128. 48 query heads, 8 KV heads,
44 effective layers. That reproduces the measured 176 KiB/token exactly:
`2 × 44 × 8 × 128 × 2 = 180,224 B`.

Canonical card facts: **3B non-embedding** parameters (4.17B total — embeddings carry the rest),
262,144 context, `enable_thinking` **on by default**, thinking-mode sampling temp 0.6 / 131,072 cap.

### The teacher: Nanbeige4.3-33B is heavily quantized and is a *plain Llama*

**28 GGUF repos, 18 MLX repos.** `bartowski/Nanbeige_Nanbeige4.3-33B-GGUF` has 127,867 downloads —
more than the base repo itself. So no, nothing needs making.

```
33.12 B params | 52 layers | 56 q heads | 8 KV heads | head_dim 128
max_position_embeddings 524,288      model_type: llama      num_loops: None
KV = 208 KiB/token -> 32K=6.5 GiB   64K=13.0 GiB   262K=52.0 GiB
```

Two facts worth keeping: it is **not looped** (`model_type: llama`, vanilla dense), so it loads in any
engine on any device with no `trust_remote_code` — only the *student* is exotic. And its 512K window
costs 208 KiB/token, 2.6x granite's.

| quant | size | + 32K cache | fits 24 GB? |
|---|---:|---:|---|
| `i1-IQ1_S` | 7.13 G | 13.6 G | yes, but 1-bit |
| `i1-IQ2_XXS` | 8.69 G | 15.2 G | yes |
| `bartowski IQ2_XXS` | 9.38 G | 15.9 G | yes |
| **`i1-IQ3_XXS`** | **12.21 G** | **18.7 G** | **yes — the quality floor worth using** |
| `Q3_K_M` | 14.83 G | 21.3 G | tight |
| `Q4_K_M` | 19.36 G | 25.9 G | **no** |

**Do not distil from IQ2.** Distillation copies the teacher's damage into the student permanently;
there is no un-distilling a 2-bit teacher. If it is worth doing, IQ3 is the floor and you accept fewer
examples.

**Time, which is the actual constraint.** 5,000 examples x 600 tokens = 3 M tokens:

| where | speed | wall clock |
|---|---:|---:|
| this Mac, IQ3_XXS (12.2 G, ~400 GB/s) | ~16 tok/s | **~52 h at full fans** |
| Kaggle P100, IQ2_XXS + 16K cache (12.6 G of 16) | ~39 tok/s | **~21 GPU-hours** |

The Mac number violates the heat rule on its own. Kaggle can hold this model — 12.6 GiB fits a 16 GB
P100 — which makes the cloud lane useful for the first time for something other than the grid sweep.

**Corroboration:** `WaveCut`'s heretic recipe and the `mps-fix` notes both agree the shipped 4.2-3B
weights are clean and the bugs were in the Python port. The bf16 download verified at 8,339,632,672 B,
223 tensors, 4.17 B params, 22 physical blocks.
