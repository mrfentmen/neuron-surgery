# NCM — Next-Concept Modeling. What it is, and whether it kills anything

Reviewed 2026-11-06. Paper `arXiv:2610.14162` (9 Oct 2026), Hazy Research (Stanford) + Together AI.
Code `github.com/HazyResearch/ncm`, **MIT**, checkpoints and tokenised datasets on Hugging Face.
Preprint — **not peer reviewed**.

The headline circulating is "NCM, the new LLM killer." It is not that. Here is what it is.

---

## In plain English

**A normal model lays bricks.** One word at a time, left to right. A thousand-word answer is a
thousand sequential steps, and once a brick is laid it cannot be moved.

**NCM draws the sketch first, then fills it in.** Two stages:

1. **A compressor** (multiscale residual VQ-VAE) learns to turn any piece of text into a stack
   of codes: a handful at the top that capture the gist, more and more at each level below that
   capture the detail. Trained once, like a tokenizer.
2. **A generator** (an ordinary transformer) writes **one whole level per forward pass** — every
   code in that level predicted in parallel.

So an 11-level model needs **11 passes**, not 1,000. And because every level sees the entire
finished sketch, **it can revise what an earlier level decided**, which is impossible for a
brick-layer.

The idea is lifted from image generation: VAR, NeurIPS 2024 best paper — low-res sketch, then
upscale, then upscale.

---

## The numbers they report

All at **~100M parameters**, on **one H100**, and — this is the line that matters —
**every speed comparison is against diffusion language models, never against a tuned
autoregressive model.**

| task | result | vs |
|---|---|---|
| LM1B (unconditional) | **21× faster**, better generative perplexity | strongest DLM |
| OpenWebText (unconditional) | **10× faster**, competitive quality | strongest DLM |
| XSum (summarisation) | **31× faster** at matched ROUGE-L | strongest DLM |
| training cost | **4–5× fewer tokens** | strong DLMs |
| Sudoku, 30-clue hard | **86.1%** | 81.3% FMLM+ · **1%** same-size AR |
| Sudoku, easy | **99.5%** | 14.6% same-size AR |
| German→English | BLEU 77.8 by level 11 (level 1 is gibberish) | — |

The Sudoku row is the genuinely interesting one. **86.1% against 1% for an autoregressive model
of the same size** is not a speed result, it is a *shape* result: coarse-to-fine lets every level
see the whole board and fix earlier mistakes, which is exactly what a global constraint needs.

---

## The five caveats that end the "killer" framing

1. **The baseline class is wrong.** Diffusion LMs are not the thing anyone runs. There is no
   comparison to llama.cpp, vLLM, or any tuned AR stack. "31× faster than diffusion" says
   nothing about "faster than ChatGPT."
2. **100M parameters.** Scaling to 725M is described as preliminary. Frontier models are
   three to four orders of magnitude larger, and the coarse-to-fine advantage at that scale
   is unmeasured.
3. **One checkpoint per task.** XSum, WMT14, Sudoku, LM1B, TinyStories, OpenWebText — six
   separate models. There is no general instruction-following NCM, no tool calling, no chat.
4. **Fixed depth.** All levels always run. A two-word answer pays the full 11 passes.
5. **It loses to AR without tricks.** German→English *without* extra machinery still trails an
   autoregressive model of the same size.

---

## Two independent reproductions, both unflattering

**DGX Spark, batched serving** (MindStudio, 29 Oct 2026):

```
single request, translation      ~4x faster than matched AR   <- the headline holds
single request, long summary     ~2x faster                   <- holds, weaker
batch of 64 concurrent requests  standard AR wins             <- the headline dies
```

Serving is batched. A speedup that only exists at batch 1 does not reduce anyone's bill.

**MLX, TinyStories, trained from scratch** (`ncm-mlx`, 2 Nov 2026) — the reproduction that
should be read before anyone runs this on a Mac:

> "They reconstruct held-out stories almost perfectly, but the text they write on their own is
> a lot worse than text from a plain autoregressive baseline built from the same transformer
> blocks."

```
NCM, strong corruption    teacher-forced 2.249   generation 47.9
NCM, reduced corruption   teacher-forced 1.665   generation 54.9
NCM, no corruption        teacher-forced 1.390   generation 97.8   <- best loss, worst text
```

**The training change that improved the loss made the output worse**, reproducibly, across two
generator seeds. That is textbook exposure bias: a model that only ever saw perfect earlier
levels during training never learned what to do with its own mistakes.

---

## Verdict for this repo

**Track 1 (neuron surgery): no overlap at all.** NCM has no MLP to hook in the sense this
project uses. Nothing here touches the map.

**Track 2 (the 1M / agentic lab): nothing usable today.** It is not a chat model, not
instruction-tuned, not tool-calling, not one model for many tasks, and 100M parameters against
a 9B Qwen3.5 is not a comparison worth making.

**But two ideas are worth stealing, and neither requires the code:**

1. **No growing cache per token.** Because generation is a fixed number of passes over a fixed
   number of codes, the "notes per word" problem in `CONTEXT_1M.md` largely stops applying to
   *generation*. That is a real answer to a real cost — it just is not an answer for *reading*,
   which is what 1M context is actually for.
2. **Revise instead of commit.** The Sudoku result is evidence that an architecture allowed to
   revisit earlier decisions beats one that cannot, by 85 points at matched size. That is the
   same claim this repo already makes from the other direction: behaviour is set by *where the
   model can look and what it can change*, not by how many neurons it has.

**Do not install it.** Watch it. If a 1B+ general, instruction-tuned NCM appears with tool
calling and a batched-serving story, that is a different conversation — and it will be
announced loudly enough that missing it is not a risk.

---

## Sources

- Paper: `arxiv.org/abs/2610.14162`
- Authors' blog: `hazyresearch.stanford.edu/blog/2026-10-12-ncm`
- Code: `github.com/HazyResearch/ncm` (MIT, 5 commits at review)
- Independent batching test: MindStudio, 29 Oct 2026
- Independent MLX reproduction: `dev.to` `ncm-mlx`, 2 Nov 2026
