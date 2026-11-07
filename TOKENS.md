# Tokens: what they cost, and where to stop paying

Written 2026-11-06. Lane 2/3. Companion to `CONTEXT_1M.md` — that one is about *getting*
a big window, this one is about **spending fewer tokens inside it**.

---

## 1. Measured on this repo (not estimated)

`corpus.txt` + the 14 Python files in this directory, Granite tokenizer (100,352 vocab):

| corpus | chars | words | tokens | tok/word | **chars/token** |
|---|---|---|---|---|---|
| `corpus.txt` (prose) | 728,846 | 127,378 | 171,543 | 1.35 | **4.25** |
| 14 × `.py` (code) | 50,558 | 6,001 | 15,291 | 2.55 | **3.31** |

**Code costs 28% more tokens per byte than prose.** Same characters, 28% less text.
Identifiers, brackets, indentation and snake_case all fragment into separate tokens.

What one million tokens therefore buys:

```
prose   1M x 4.25 chars = 4.25 MB   (~700K words, ~7,000 pages)
Python  1M x 3.31 chars = 3.31 MB   (~94,000 lines at 35 chars/line)
```

**Use 94K lines as the planning number for a codebase**, not "a million words of code".
A 1M window holds one large repo. It does not hold three.

## 2. Where the tokens actually go in an agent loop

From vLLM's production traces of agentic workloads (610 traces):

```
turn 30 context length          ~80,000 tokens
longest contexts              >180,000 tokens
new tokens added per turn        a few hundred to a few thousand
average input:output ratio       131 : 1
```

**131 input tokens for every output token.** The model is almost never generating — it is
re-reading. And 99%+ of what it re-reads it has already read before.

This is the single most important fact about token cost: **the bill is re-reading, not
writing.** Every optimization below is a way to stop re-reading.

## 3. The levers, ranked by measured effect

### L1 — Prefix caching. Free, no quality change, biggest lever.

llama.cpp: `cache_prompt: true` on every request, `-np N` to keep N prefixes hot.
The KV state of an unchanged prefix is reused instead of recomputed; only tokens after the
first difference cost anything.

| measurement | result |
|---|---|
| vLLM + Mooncake, agent traces (cache hit 1.7% → 92.2%) | **3.8× throughput, 46× lower TTFT** |
| llama.cpp, Qwen3-0.6B, M1 Max, 4-turn chat | 1.23-1.31×, 110 ms/turn |
| cached input vs uncached, commercial API | **90% cheaper** |

The gap between the two llama.cpp numbers and the vLLM number is *hit rate*, not technique.
A 1.3× gain means the cache is missing.

**The rule that decides whether you get 1.3× or 46×: stable content first, volatile last.**
The hash matches from the start, so one changed byte near the top invalidates everything
after it.

```
WRONG                                  RIGHT
  timestamp / "current file list"        system prompt (unchanging)
  system prompt                          tools + schemas (unchanging)
  tools                                  examples (unchanging)
  conversation                           conversation history
```

Putting a timestamp, a directory listing, or a git status at the top of the prompt is the
most common way people accidentally pay full price on every single turn.

### L2 — Quantized KV cache. Halves memory, negligible quality cost.

`--cache-type-k q4_0 --cache-type-v q4_0`. In the Nemotron benchmark this is exactly what
turned 96K into 768K of context on a 24 GB card. Quality cost is small; the memory
saving is 4×.

### L3 — Query-aware trimming of tool output. 94-99% of the tokens, 100% of the accuracy.

This is the one almost nobody does and it is nearly free. `tooltrim` benchmark, exact
tokenizer counts, one planted fact per output:

| tool output | before | after | saved | needle kept |
|---|---|---|---|---|
| web page (HTML) | 2,816 | 13 | **99.5%** | yes |
| REST response (JSON) | 15,119 | 325 | **97.9%** | yes |

Versus blind truncation, which is what most agents do:

| budget | truncate-head | truncate-tail | query-aware |
|---|---|---|---|
| 128 | 1.8% retained | 1.8% | **100%** |
| 256 | 3.6% | 3.6% | **100%** |
| 800 | 12.5% | 14.3% | **100%** |

Blind truncation drops the needed fact **98% of the time**. Trimming against the actual
question keeps it and still removes 97-99% of the bytes. If your agent dumps a whole JSON
response or a whole HTML page into context, this is the highest ratio of gain to effort in
this document.

### L4 — Speculative decoding. 2-3× decode, but **measure it, do not assume it**.

| method | acceptance length | reported speedup |
|---|---|---|
| independent draft model | ~3.6 tok | 2-3× |
| Medusa-2 | 3.0-3.5 | 2.3-3.6× |
| EAGLE-3 | 5-7.5 | up to 6.5× (paper) |
| MTP (built into model) | — | up to 3× |

**The disciplined measurement contradicts the paper numbers.** SpecDecode-Bench (vLLM, H100,
five methods, four models): EAGLE reaches 1.96× at batch 1 on Llama-70B, but falls to
**1.21× at batch 128**. And a controlled GPT-2 test got **1.10×** at 62.5% acceptance under
greedy decoding, but **0.85× — slower than baseline —** under sampling.

Why slower: every rejected draft token is wasted compute, and sampling draws a random number
per candidate. Speculation is a bet that pays only when acceptance is high and the draft is
cheap. **At batch 1, which is your case, it is worth trying. Just measure tokens/sec, not
acceptance rate** — the paper that got 6.5× and the test that got 0.85× used the same method.

### L5 — Tokenizer compression (zip2zip). +54% on code, inference-time, no retraining.

LZW-style online compression of the input stream into "hypertokens", applied at inference.
Reported gains are **largest exactly where you need them**:

| tokenizer | code | math | chat |
|---|---|---|---|
| Llama-3 | 4.1 B/tok | 2.7 | 5.1 |
| + zip2zip | **6.3 (+54%)** | 4.x | — |

This is a research-grade tool, not a llama.cpp flag. Worth knowing it exists; not worth
blocking on. It also confirms the diagnosis in §1: **code is the worst-tokenized domain and
the most compressible one.**

## 4. What not to do — measured, and both are negative

### KV cache eviction (SnapKV, H2O, StreamingLLM, TOVA, K-Norm)

SnapKV looks great on retrieval benchmarks — **1.46% average drop at 10% KV retention**.
But ACL 2026 ("The Pitfalls of KV Cache Compression") tested five eviction methods on
Llama-3.1-8B and Qwen2.5-14B under multi-instruction prompting:

- **Certain instructions degrade much faster than others — to the point of being completely
  ignored by the model.**
- Eviction causes **system-prompt leakage**.
- The damage depends on the eviction policy, the instruction order, and the model.

So: evicting tokens passes the benchmark you measured and silently breaks the instructions
you did not. This is the same failure as context compression in NanoHarness (**−4.86**) —
removing tokens removes the thing that made the agent right.

### Summarizing the conversation to fit

NanoHarness, ProgramBench, 10 models: **context compression −4.86 points.** Delegating to a
subagent so the overflow never happens is fine. Compressing to survive overflow is not.

### The distinction that matters

```
delegation     move the work to a fresh window     GOOD   (+5.91)
trimming       drop bytes that aren't relevant     GOOD   (100% retained)
eviction       drop tokens the model already read  BAD    (instructions break)
summarizing    rewrite the history smaller         BAD    (−4.86)
```

**Never remove something the model has already attended to. Decide what enters context
instead.** That is the whole principle.

## 5. The config this repo should actually run

```bash
# build first - nothing here works today, llama.cpp/mlx/ollama are all absent
llama-server -m <model>.gguf \
  -ngl 99 \
  -c 32768                    # start small; raise only after measuring
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -np 2                       # two hot prefixes: system prompt + tools
  --cache-reuse 256           # tolerate small prefix drift instead of full miss
```

Request side: `"cache_prompt": true`, and **put every unchanging byte at the front**.

Then measure, in this order:

1. `tokens/sec` prefill with cache cold vs warm on the same 8K prompt → this is L1's real gain
2. `tokens/sec` decode with cache q8_0 vs f16 → confirm quality holds on a real task
3. tool-output bytes before/after query-aware trimming → this is L3, expect 90%+
4. only then try a draft model, and only if 1-3 are done

**Do not implement L4 before L1.** Speculation buys 2×; a broken prefix cache costs 46×.
The order matters more than the technique.

## 6. The summary

```
Tokens are not uniform.   code costs 28% more than prose (measured here)
The bill is re-reading.   131 input tokens per output token in agent loops
The fix is caching.       free, lossless, up to 46x TTFT, needs stable-prefix ordering
Then trim inputs.         97-99% of tool output is droppable, needle survives
Never evict.              instructions break silently and system prompts leak
```
