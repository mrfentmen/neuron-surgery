# Dataset sheet — Granite 4.1 3B lab work

Compiled 2026-11-06 on the M1 Max. Every row and byte figure below was read from
the files on disk (parquet metadata / line counts), not copied from a model card.
Path is `~/Desktop/neuron-surgery/datasets/`.

**34 repos · 7.73 GB · ≈2,253,000 rows.** Free space 749 GB.

---

## 1. On disk now

| repo | rows | MB | columns | what it actually is |
|---|---:|---:|---|---|
| `AMAImedia/NOESIS-1M` | **1,000,000** | 1431 | `text, domain, src, tok_len` | NEW. Multilingual router across Opus 4.7 / DeepSeek V4 / Qwen3.6 / Gemini 3.1 / R1 / GPT-5.4. Short records — 1M rows in 1.4 GB. A **sampling pool**, not an epoch. |
| `Kassadin88/Claude-Distills` | **153,056** | 959 | `messages, source` | NEW. Only **2** data files: `claude_distill.jsonl` (911 MB) + `claude_merged.jsonl` (48 MB). The "six pre-sized Opus sub-sets" claim is **not** what the repo holds. |
| `qsardor/Claude-Sonnet-Opus` | 115,264 | 1827 | `messages, source` | 92.4% Sonnet 4.6 / 7.6% Opus 4.6–4.7, reasoning inside ` thinking`. Biggest usable single corpus here. |
| `Magpie-Reasoning-V2-250K` (DeepSeek-R1-Llama-70B) | 249,922 | 1616 | `instruction, response, conversations` | R1-style reasoning, 70B teacher. Volume, not frontier. |
| `opc-sft-stage2` | 436,347 | 537 | `instruction, output, tag` | Curated code SFT. Compressed well; cheap per row. |
| `Magpie-Reasoning-V1-150K` | 150,000 | 368 | `instruction, response` | Older R1 distill, same shape. |
| `cudabenchmarktest/Astra-COT-40K` | 40,924 | 175 | `messages` | **NOT GPT-6 Astra.** Opened it: nuclear-policy conversation. Beike ASTRA agentic project, unrelated. |
| `angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k` | 38,504 | 260 | `category, messages, model` | More than the name says (38.5k), Opus 4.6/4.7 traces. |
| `open-r1/Mixture`-style `Open-CoT-Reasoning-Mini` | 10,200 | 77 | `id, input, output` | Distilled CoT, sub-10B oriented. |
| `math-rlvr-15k` | 15,000 | 7.6 | `prompt, label, metadata` | RLVR prompts + verifier labels, **no traces**. Reward-side data. |
| `Jackrong/DeepSeek-V4-Distill-8000x` | 7,716 | 142 | `conversations, input, output, domain` | DeepSeek V4 distill, multi-turn. |
| `DiagSafe` | 6,978 | 23 | `id, rubric` | Safety rubric set. Rubric field, not instruction/response — needs a formatter. |
| `spitfire4794/Fable-5-traces` | 4,665 | 70 | `session, model, context` | Fable 5 agent traces. |
| `opus-4.7-reasoning-cot` | 4,810 | 11 | `messages` | Compact Opus 4.7 CoT. Good <30 min run. |
| `blythet/deepseek-v4-pro-math-cot-1k` | 2,355 | 39.6 | `id, source_line, url` | **No answers.** A manifest pointing at where data lives. Trains on nothing. |
| `tiny-think-dpo-math-n-stem` | 2,861 | 17 | `prompt, chosen, rejected` | **DPO triplets**, not SFT. Different loss. |
| `Supra-Reasoning-CoT-Dataset-v2` | 2,785 | 3.2 | `user, CoT, assistant` | Architectural Q&A CoT. |
| `deepseek-v4.1-flash-distill-zh-2k` | 2,000 | 22 | `instruction, thinking, response` | Chinese, has explicit `thinking`. |
| `cot-reasoning-2k` | 2,151 | 7.2 | `messages, quality_score` | 2,151 rows, all 5/5 algorithmic verification. |
| `Ling-Tiny-SFT` | 1,500 | 9.7 | `bucket, input, ground_truth` | Extreme formatting (JSON / raw math eval). |
| `Deepseek-v4-pro-max-distill-1500x` | 1,499 | 42 | `prompt, reasoning, response` | DS V4 Pro Max. |
| `s1K` | 1,000 | 6.9 | `solution, question, cot_type` | The published 1k set. |
| `s1K-1.1` | 1,000 | 22 | same | Harder slice. |
| `babysor/deepseek-v4-pro-max-distill-1k` (`ansulev/…-1k`) | 1,000 | 26 | `prompt, reasoning, response` | Same shape, parquet. |
| `gpt-5.6-luna-sft-900x` | 890 | 3.0 | `topic, model, messages` | GPT-5.6 Luna SFT. |
| `LIMO` | 817 | 16 | `question, solution, answer` | 817 rows — the published claim checks out. |
| `cybersecurity-reasoning-cot-v1` | 89 | 0.2 | `threat_logic, vulnerability_analysis, mitigation_reasoning` | 89 rows, analysis prose, no instruction field. |
| `MATH-COT` | 80 | 0.1 | `question, prompt, response` | 80 rows. A probe set, too small to train. |
| `Code-sonnet-5-gpt-5.5-kimi-k2.5` | 36 | 1.2 | `instruction, output, model` | 36 rows across three teachers. A sample, not a set. |
| `mondk/Deepseek-v4.1-CoT` | 23 | 0.2 | `instruction, output` | 23 rows. |
| `GPT-6Astra-scientific-research` | 20 | 2.7 | `messages, metadata` | The only genuine GPT-6 Astra corpus. 20 rows. |
| `GPT-6.1-Sol-Problem-solving-Reasoning-Distilled` | 1 obj | 0.8 | JSON blob | One JSON object, not tabular. Needs conversion. |
| `Distilled-Opus5-GPT5.6sol-Math-Reasoning` | 1 obj | 7.2 | JSON blob | Same problem. |
| `refusalbench` | 0 | 2.3 | — | Empty / gated. Nothing usable. |

---

## 2. Found, measured, deliberately not pulled

| size | repo | rows | one epoch* |
|---:|---|---:|---:|
| 104 GB | `RESMP-DEV/Fable-GPT-5.5-Distillation-Traces` | 1.9M train + 81k test | ~105 h |
| 209 GB | `Solstice-AI/Solace-1.0-Omni` (ModelScope) | — | not on this machine |
| 12.6 GB | `open-r1/OpenR1-Math-220k` | 220k | ~12 h |
| 11.6 GB | `Manusagents/GPT-5.5-…-and-more-Distillation` | 18.5M | 1000 h+ |
| 8.0 GB | `Jackrong/DeepSeek-V4-Pro-Distilled-200K` | 200k | ~11 h |
| 6.1 GB | `open-r1/Mixture-of-Thoughts` | 349k | ~19 h |
| 3.2 GB | `thetrillioniar/claude-sonnet-4.6…mythos-5-fable-5` | ~250k | ~14 h |
| 1.9 GB | `Avtrkrb/combined-reasoning-…` | 1M–10M | single `messages` schema, 41 sources |
| 1.4 GB | `AMAImedia/NOESIS-1M` | 1M | **PULLED** |
| 1.05 GB | `greghavens/gpt-5.6-sol-coding-and-debugging-traces` | mixed | — |
| 959 MB | `Kassadin88/Claude-Distills` | 153k | **PULLED** |
| 948 MB | `samcheng0/lumia-reasoning-sft-v1` | — | pull on request |
| 825 MB | `VINAY-UMRETHE/Sonnet-Opus-…-Gemini-3.1` | 70.2k | ~4 h |
| 735 MB | `Roman1111111/claude-sonnet-4.6-100000X-filtered` | 100k | ~5.5 h |
| 260 MB | `angrygiraffe/…-8.7k` | 38.5k | **PULLED** |
| 126 MB | `bespokelabs/Bespoke-Stratos-17k` | 17k | ~1 h |
| 51 GB | `tinycompany/Tiny-Short-R1-CoT-Distill-HF-ChatML` | — | "tiny" = context chunks, not download |

\* arithmetic at ~5 samples/s, 512 tokens, LoRA r=16. **Not measured** — the sweep owns the GPU.

**Gated (401):** `ykckevin/astra` needs a login.

---

## 3. The scaling wall

```
   8,706 rows  ->    29 min
 115,264 rows  ->     6.4 h
 153,056 rows  ->     8.5 h   <- Claude-Distills
 349,000 rows  ->    19 h
1,000,000 rows  ->    55 h    <- NOESIS, if you ran an epoch (don't)
1,900,000 rows  ->   105 h
```

Time scales linearly with rows. Both new pulls are **pools to sample from**, not
things to run an epoch on: draw 20k–50k NOESIS rows (≈1–3 h) and 10k–20k
Claude-Distills rows, not the whole file.

---

## 4. Heretic — checked, not a dataset

`p-e-w/heretic` (pip `heretic-llm`, **2.0.0**, released 2026-11-01; Codeberg mirror
exists). It is a **tool**, so there was nothing to pull into `datasets/`.

- Directional ablation (abliteration) + Optuna TPE study that **co-minimises refusals
  and KL divergence** from the original model. Default action is refusal suppression.
- Also ships configs for `noslop`, `nohumor`, `nosyco`, `emotions`.
- **2.x defaults to the ARA modifier, not abliteration** — you must set `modifiers`
  in the TOML to use the classic ablation path.
- Usage is `heretic <hf-model-id>`; output is a modified model on disk, plus a
  `reproduce.json` so the run can be replayed exactly.

It is the step the roadmap already calls for (ablate refusal, then steer the
register), and it does it automatically instead of by hand.

Two things before installing: it wants `torch>=2.6` and will drag in `optuna`,
`tyro` and friends. The project venv holds `torch 2.15.0` + `transformers 5.22.1`,
so install it into a **separate** venv (`.venv-heretic`) rather than risk the sweep's
pinned stack. And an Optuna study is 200 trials by default on a 3B model — that is a
multi-hour GPU job on this machine, so it is not something to start while the Kaggle
lane is the only thing keeping the map moving.

---

## 5. Machine state at time of writing

```
local runner     STOPPED (STOP file present; remove it to allow a restart)
                 rows preserved: 1,180 across 10 grid_L*_r.jsonl floors
kaggle lane      RUNNING  aeeeeeeee/granite-random-gap-l20-l32
training         none
disk             7.73 GB datasets · 749 GB free
```
