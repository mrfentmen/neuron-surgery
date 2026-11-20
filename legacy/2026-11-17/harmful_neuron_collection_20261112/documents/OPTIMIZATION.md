# What actually makes a small model good

Written 2026-11-06. Every effect size below is someone's measurement, with the paper named.
Nothing here is an estimate from this repo. Where a number is a range or a single benchmark,
it says so.

---

## 0. The premise to kill first

> "fine tuning does nothing if its low context"

**Wrong, and it costs time.** Fine-tuning changes weights. Context is a runtime budget.
A model SFT'd on arithmetic does arithmetic with a 300-token prompt. What context limits is
**task composition** — holding a repo slice + tool results + turn history at once — not whether
learning transfers.

The sharpest counter-evidence is a direct comparison on the same task
(`arXiv 2604.11465`, Qwen3-8B on AppWorld, one 24 GB GPU):

```
FP16,  12K context  ->  better task completion
AWQ 4-bit, 32K context ->  worse
```

**Less context at full precision beat nearly 3x the context at 4-bit.** Precision, not window
size, was the binding constraint. Every "we need 1M" decision should be checked against this.

---

## 1. The levers, ranked by measured effect

| # | lever | measured effect | source |
|---|---|---|---|
| 1 | **Harness / scaffold** | **+19.4 pts** range across harnesses on SWE-bench Verified — against a **+22.8 pt** range across models. The scaffold is worth about as much as the model. | arXiv 2609.32459 |
| 2 | **Test-time compute** (sample + verify) | Llama-3.2-**1B** on GSM8K: `pass@1 0.35 -> self-consistency 0.46 -> best-of-N 0.71`. **2x from sampling alone.** | arXiv 2606.05932 |
| 3 | **Data curation** | 817 rows: AIME24 `6.5% -> 57.1%`, MATH500 `59.2% -> 94.8%`. 400 rows already gets most of it; gains plateau past ~800. | LIMO, arXiv 2502.03387 |
| 4 | **RLVR with process rewards** | Qwen2.5-**0.5B**: process-only `63.73%` vs outcome-only `53.75%` on GSM8K. Qwen2.5-**3B**: `+8-10pp`, plus `+14pp` on correctly saying "unknown". | arXiv 2607.02869, ICLR 2026 Z3-GRPO |
| 5 | **Model size** | +22.8 pts, for ~10x the memory and lower tok/s | arXiv 2609.32459 |

**Read the ordering.** Four of the top five are free at inference time or cost a data-prep
afternoon. Only #5 costs hardware. This is why "get a bigger model" is the wrong fifth move —
it is roughly as valuable as fixing the harness, and it is the only one that cannot be undone.

---

## 2. The fact that matters most for a 3B

> **Weaker models gain MORE from scaffolding.** (`arXiv 2609.32459`, matched-model table)

```
SWE-agent  on Claude 3.7 Sonnet   +9.60
SWE-agent  on Claude 4 Sonnet     +4.07
OpenHands  on Claude 3.7 Sonnet  +13.60
OpenHands  on Claude 4 Sonnet     +5.47
```

The stronger the model, the smaller the scaffold's marginal gain. **A 3B is the model class
that benefits most from harness engineering**, which is the opposite of the intuition that
says "scaffolding is for when you already have a good model."

---

## 3. How a 3B actually fails — and it is not where you think

`arXiv 2605.12129` ran 2-3B models through three harness levels (raw prompt / wrapper tags /
4-stage `plan -> execute -> verify -> recover`).

```
LLaMA 3.2 3B, raw prompt:  format violations = 7
                           content correct (VTSR) = 0.952
                           task success  (TSR)    = 0.429
```

**The content was right 95% of the time and the task still failed.** The model produced the
right answer in the wrong shape and the pipeline could not read it. That is *scaffold
collapse*, and it is a formatting problem masquerading as an intelligence problem.

Consequence: **enforce structure structurally.** Do not ask a 3B for JSON and hope. Constrain
the decoder, or parse-and-retry, or split into stages where each stage's output is validated
before the next begins. Each pipeline stage contributed ~24.7% of total gain; verification
caught 62.5% of errors that reached it.

---

## 4. What not to do (measured to hurt)

| anti-pattern | evidence |
|---|---|
| **More context at lower precision** | FP16/12K beat 4-bit/32K (arXiv 2604.11465) |
| **Generic subagents on repo tasks** | "task-specific subagents provide the most stable improvements" — general ones do not (arXiv 2609.32459) |
| **Context compression on repo-generation** | "can hurt repository-generation performance" (same paper) |
| **Outcome-only rewards on a small model** | 10 pts below process rewards at 0.5B (arXiv 2607.02869) |
| **Assuming SFT creates ability** | see §5 |

---

## 5. Why SFT sometimes "does nothing" — the real mechanism

This is the LIMO hypothesis (`arXiv 2502.03387`), and it is the most useful single idea here:

> The threshold for eliciting complex reasoning is not set by task complexity. It is set by
> (1) how completely the base model already encodes the domain, and (2) whether the post-training
> examples work as **cognitive templates**.

**SFT selects and formats. It does not create.** If the capability is in the weights, a handful
of perfect demonstrations unlocks it. If it is not in the weights, ten thousand rows of SFT will
teach the model a *style* and leave the skill where it was — which is exactly the "I fine-tuned
it and nothing happened" experience.

**The test, before spending a single GPU-hour:** put a perfect worked example in the *context*
of the base model and ask the same task. If it can follow the exemplar, the capability is there
and SFT will lock it in. If it cannot, SFT will teach it to imitate the exemplar badly. That is
a 10-minute test that saves a week.

Corollary for abliteration ordering (see `PIPELINE.md`): if the base **refuses** the training
data, you must unblock it first — otherwise there is no capability to elicit.

---

## 6. The small-model trick that beats a bigger model outright

`ICLR 2026, T1: Tool-integrated Verification`:

> "with T1, a Llama-3.2 **1B** model under test-time scaling outperforms the significantly
> larger Llama-3.1 **8B** model" on MATH.

Mechanism: small models fail verification when it requires **memorization** (arithmetic,
fact-checking). So offload the memorization to a tool — a code interpreter — and keep the
model only for judgment. The model stops needing to know `17+26` and starts needing to know
*what to compute*.

That is the general form, and it is the same argument as the persona result in this repo:

```
the model supplies   -> judgment, register, selection
the tool supplies    -> exactness (python, pytest, grep, retrieval)
```

"Quietly correct" is a system property. It was never going to be a weights property.

---

## 7. For *this* 3B, in order

The only sequence where each step is cheap and each step is measured:

```
0.  fix the instrument        45 -> 200 max_new, grader bug    (PATTERNS 5 caveat)
1.  build the harness         plan -> execute -> verify -> recover
                              format-constrained output at every stage
2.  sample and verify         N=8, pick by verifier. 2x is on the table for free.
3.  measure the ceiling       exemplar-in-context test (section 5). Decides everything after.
4.  SFT                       400-817 curated rows, not 400k
5.  RLVR                      process rewards, on Kaggle, not the laptop
6.  (only then) bigger model
```

Steps 1-2 need no GPU, no data, and no download. They are the largest measured gains on the
list and they are sitting unused.

---

## 8. The honest ceiling

A 3B with a perfect harness is not DeepSeek. The gap that remains after steps 1-6 is
**knowledge** — facts, long-tail reasoning depth, compositional generalization — and that gap
is closed only by parameters or by retrieval.

What the harness *does* buy is the difference between "can't do the task" and "can do the
bounded version of the task reliably." For a laptop model, that is the whole game.

The closest thing in the wild to "small beats big" is a 3B-**active** MoE: Ornith-1.5-35B-A3B
scores 68.5 on Terminal-Bench 2.1 against 43.4 for dense Gemma-4-31B, activating a tenth of the
parameters. That is architecture, not post-training — and it is the one item on the electronics
list that cannot be retrofitted onto a dense checkpoint.
