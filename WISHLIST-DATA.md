# Data wishlist — every link, measured, ordered small → large

Compiled 2026-11-07. **Nothing was downloaded.** Every size and row count below came
from a metadata call to the Hugging Face API / datasets-server, not from a dataset card
and not from pulling files. 25 dataset links + 2 GitHub repos + 2 collections + 1 paper
were resolved; all 25 dataset repos exist.

The rule for this file: **small first, one at a time, delete or keep on purpose.**
No parallel pulls. No "just grab the whole thing".

> **Scope.** This sheet is only about `datasets/` and what to pull into it.
> `runs/data/` belongs to the neuron-nudging track and is not SFT data — it is
> deliberately sampled equal-N per dataset for controlled comparison. Nothing in
> this file applies to it, and it should not be rebuilt from the wishlist.

---

## 0. The constraint that decides everything

```
want, if every link were pulled whole   ~ 3,800 GB
disk free right now                        745 GB
on disk already, datasets/                 7.3 GB
on disk already, whole project            18.0 GB
(so ~7.8 GB is data; 10 GB is weights + venvs)
```

**You cannot hold this list.** Not "shouldn't" — the whole wishlist is five times the
free space. So the giants are not "later", they are **sample-only or never**. That is
the plan below: tiers A–C get pulled, tier D gets read through a filter or skipped.

---

## 1. TIER A — small pulls. Do these first.

Total for the whole tier: **~0.4 GB.** These are an evening's work and they are where
the actual value is, because they are *traces* — finished reasoning and finished agent
sessions — not raw code dumps.

| # | dataset | rows | size | what it actually is | verdict |
|---:|---|---:|---:|---|---|
| A1 | `DeepNLP/Coding-Agent-Github-2025-Feb` | 750 | <1 MB | coding-agent traces from GitHub activity | **pull.** Tiny, on-target. Warm-up. |
| A2 | `Convence/Rust-Coder` | 12,000 | 1 MB | Rust SFT | 87 bytes/row — **verify 1 row before trusting.** Likely stubs. |
| A3 | `greghavens/glm-5.2-coding-and-debugging-traces` | 1,821 | 2 MB | GLM-5.2 coding + debugging traces | **pull.** Same author/shape as the gpt-5.6 set already proven useful. |
| A4 | `greghavens/kimi-k3-coding-and-debugging-traces` | 3,956 | 3 MB | Kimi K3 coding + debugging traces | **pull.** |
| A5 | `greghavens/gpt-5.6-sol-coding-and-debugging-traces` | ? | 1,003 MB | GPT-5.6 Sol coding + debugging traces | **pull, but it's tier B by size.** Listed in `DATASETS.md` §2 as seen-not-pulled. |
| A6 | `echel0nn1881/kimi-cyber-reasoning` | 997 | 10 MB | Kimi cyber reasoning | **pull.** Already referenced in your security cluster. |
| A7 | `AlicanKiraz0/Agentic-Chain-of-Thought-Coding-SFT-Dataset` | 429 | 14 MB | agentic CoT coding SFT | **pull.** Exact shape for stage 4 (tool use). |
| A8 | `Manusagents/Vibe-Coding-Claude-Fable-5` | 1,100,000 | 28 MB | "vibe coding", Claude Fable 5 | 25 bytes/row for 1.1M rows — **that is a stub or a pointer set.** Open 1 row first. Do not assume 1.1M traces. |
| A9 | `Vezora/Tested-22k-Python-Alpaca` | 22,608 | 47 MB | *tested* Python instructions | **pull.** "Tested" is the word that matters — seeds for rejection sampling, stage 5. |
| A10 | `flytech/python-codes-25k` | 49,626 | 49 MB | short Python snippets | **pull.** Bread-and-butter code SFT volume, cheap. |
| A11 | `Roman1111111/gemini-3-pro-10000x-hard-high-reasoning` | 10,031 | 85 MB | Gemini 3 Pro hard reasoning | **pull.** CoT, stage 3. |
| A12 | `armand0e/claude-fable-5-claude-code` | **63** | 85 MB | Claude Code agent sessions | **pull, and read one.** 1.35 MB per row = enormous real sessions. 63 rows is a reference set, not training volume. |

**What tier A buys:** roughly 200k rows of finished traces for 400 MB. That is already
more good data than the 817-example lesson in `OPTIMIZATION.md` says you need, in
shapes the pipeline can consume without conversion.

---

## 2. TIER B — medium. One at a time, after A is measured.

| # | dataset | rows | size | what it is | verdict |
|---:|---|---:|---:|---|---|
| B1 | `PentesterFlow/OffsecCyber` | ? | 117 MB | offensive-security corpus | **pull.** Core of the security cluster. |
| B2 | `smcleod/golang-coder` | 304,692 | 173 MB | Go code | 570 bytes/row = snippets. Fine for code SFT, useless for CoT. |
| B3 | `OpenMed/synthvision-validated-qwen-by-kimi` | 114,835 | 180 MB | medical / vision, Kimi-validated | **off-target for a coder.** Skip unless you want the medical branch. |
| B4 | `angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k` | 38,504 | 239 MB | Opus 4.6/4.7 reasoning | **already on disk** (248 MB). Do not re-pull. |
| B5 | `Nexlab/fable5-agentic-coding-sft` | 159,972 | 562 MB | agentic coding SFT | **pull.** Biggest genuinely-agentic set in the list at a sane size. |
| B6 | `greghavens/gpt-5.6-sol-coding-and-debugging-traces` | ? | 1,003 MB | GPT-5.6 Sol traces | **pull.** See A5. |

---

## 3. TIER C — big single pulls. These need a reason before they run.

| # | dataset | rows | size | warning |
|---:|---|---:|---:|---|
| C1 | `MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x` | 15,000 | **2.38 GB** | 159 KB/row. These are *long* reasoning traces. Good for CoT, but one epoch at your measured speed is hours, not minutes. Sample 3–5k, don't run the file. |
| C2 | `Voidreaper2026/coding-master-dataset` | ? | **3.57 GB** | `gated:auto` — needs accepting terms on the site first. |
| C3 | `PrimeIntellect/verifiable-coding-problems` | 144,169 | **10.07 GB** | **The most valuable single item on this list.** Verifiable = it ships the tests. That is stage 5 (rejection sampling) and stage 7 (GRPO/RLOO) in one file. Also the only thing here that can *score* itself. |

**C3 is the one to justify the disk.** 10 GB of problems with executable checks is worth
more than 300 GB of code without them, because it is the only data that can tell the
model "that answer was wrong" without a human or a bigger teacher.

---

## 4. TIER D — the giants. Sample-only. Never `--whole`.

| # | dataset | rows | size | why it is not a pull |
|---:|---|---:|---:|---|
| D1 | `bigcode/the-stack-v2-train-full-ids` | ? | **42.3 GB** | `gated:auto`. And it is **an index, not code** — the actual file contents live in a requester-pays S3 bucket, so the 42 GB is pointers. Verify this before building anything on it. |
| D2 | `Vyber07/cyber-security` | ? | **88.5 GB** | `gated:auto`. 88 GB of "cyber" will be mostly binaries, PCAPs and noise, not instruction data. |
| D3 | `bigcode/starcoderdata` | ? | **289.5 GB** | `gated:auto`. Raw code. You do not need it — `opc-sft-stage2` and `python-codes-25k` already give you code in *instruction* shape. |
| D4 | `codeparrot/github-code` | ? | **301.7 GB** | Same argument, worse: raw repos, no tests, no instructions. |
| D5 | `IFM/Code-Reasoning` | **451,546,891** | **3,052 GB** | 3 TB. 451 million rows. This is a warehouse, not a dataset. If you ever touch it, it is with a streaming filter that keeps <1% — and at 745 GB free you cannot even cache the parquet. |

**D5 arithmetic, because it ends the debate:** 3,052 GB wanted, 745 GB free. Even at 4:1
compression it does not fit. The only usable form is `--include` a narrow shard, or the
datasets-server row API for a targeted slice.

---

## 5. Links that are not datasets

| link | what it is | status |
|---|---|---|
| https://github.com/statice/awesome-synthetic-data | 273 stars, 9 KB, **a list of synthetic-data *tools***, last push **2024-01-11** | Real, but it is a tool directory and it is stale. Nothing to pull. Read it for ideas, not data. |
| https://github.com/pentesterflow/offensiveset | 107 stars, 4.4 MB, **MCP server that generates pentest commands**, pushed 2026-04-02 | A **generator**, not a dataset. Relevant if you want to *manufacture* security traces. |
| https://huggingface.co/collections/illidn18/offensive-security-datasets | collection, **18 items** | See §6, expanded below. |
| https://huggingface.co/collections/tuandunghcmut/llm-for-security-benchmarks-datasets | collection, **14 items** | Mostly **benchmarks**, not training data. See §6. |
| https://huggingface.co/papers/2502.11191 | *"Primus: A Pioneering Collection of Open-Source Datasets for Cybersecurity LLM Training"*, 2025-02-16 | The paper behind the Primus datasets. Read this before the security tier. |
| https://huggingface.co/datasets | the Hub root | Not a dataset. Index page. |

### The two collections, expanded

**`illidn18/offensive-security-datasets` (18):**
`WNT3D/Ultimate-Offensive-Red-Team` · `Panagiotis12/hackingai-training` ·
`tandevllc/offsec_redteam_codes` · `suryanshp1/kali-linux-pentesting-data` ·
`echel0nn1881/kimi-cyber-reasoning` · `oi-uae/cyber-security` ·
`Carlosaug47/GPT-5.5-Gemini-3.1-Pro-Grok-4-Claude-Fable-5-Mythos-5-Qwen-3.7-Max-and-more-Distillation-Dataset` ·
`Zia-Data-Labs/PenetrationTesting200M` · `Isamu136/penetration_testing_scraped_dataset` ·
`WhitzardAgent/CyberSecurity-1M` · `ethanolivertroy/nist-cybersecurity-training` ·
`Machivelli/Cybersecurity-Dataset-Fenrir-v2.1` ·
`Trendyol/All-CVE-Chat-MultiTurn-1999-2025-Dataset` · `Waiper/ExploitDB_DataSet` ·
`tandevllc/offsec_redteam_info` · `CJJones/Synthetic_PenTest_Reports` ·
`justinwangx/CTFtime` · `TrueNix/ctf-solver-dataset`

**`tuandunghcmut/llm-for-security-benchmarks-datasets` (14):**
`cybermetric_10000_v1` / `_2000_v1` / `_500_v1` / `_80_v1` · `cti_bench_rcm_2021` ·
`cti_bench_rcm` · `cti_bench_taa` · `cti_bench_vsp` · `cti_bench_ate` ·
`XuanwuAI/SecEval` · `cti_bench_mcq` · `2502.11191` ·
`trend-cybertron/Primus-Nemotron-CC` · `trendmicro-ailab/Primus-FineWeb`

> These 32 are **not in the ordered queue below**, because they have not been measured
> and the two collections mix training sets with eval sets. Promote them in a second
> pass once tier A–B are proven — the `cybermetric`/`cti_bench` ones are *benchmarks*,
> which is what you score yourself against, not what you train on.

---

## 6. Duplicates and already-held items in your list

| your link | status |
|---|---|
| `bigcode/the-stack-v2-train-full-ids` | **appears 3×** in your message. One row: D1. |
| `armand0e/claude-fable-5-claude-code` | **appears 2×**. One row: A12. |
| `greghavens/gpt-5.6-sol-coding-and-debugging-traces` | appears **once** in this list, but is already catalogued in `DATASETS.md` §2 (1.05 GB, seen not pulled). |
| `angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k` | **already on disk**, 248 MB, 38,504 rows. |
| `huggingface.co/datasets` | the Hub front page, not a dataset. |

**Already on disk — `datasets/`, 7.3 GB, 36 folders.** Do not re-pull any of these:
`Astra-COT-40K` · `Claude-Distills` · `claude-opus-4.6-4.7-reasoning-8.7k` ·
`Claude-Sonnet-Opus` · `Code-sonnet-5-gpt-5.5-kimi-k2.5` · `cot-reasoning-2k` ·
`cybersecurity-reasoning-cot-v1` · `DeepSeek-V4-Distill-8000x` ·
`deepseek-v4-pro-math-cot-1k` · `Deepseek-v4-pro-max-distill-1500x` ·
`deepseek-v4-pro-max-distill-1k` · `Deepseek-v4.1-CoT` ·
`deepseek-v4.1-flash-distill-zh-2k` · `DiagSafe` ·
`Distilled-Opus5-GPT5.6sol-Math-Reasoning` · `Fable-5-traces` · `gpt-5.6-luna-sft-900x` ·
`GPT-6.1-Sol-Problem-solving-Reasoning-Distilled` · `GPT-6Astra-scientific-research` ·
`LIMO` · `Ling-Tiny-SFT` · `Magpie-Reasoning-V1-150K` · `Magpie-Reasoning-V2-250K-…` ·
`MATH-COT` · `math-rlvr-15k` · `NOESIS-1M-…` · `opc-sft-stage2` ·
`Open-CoT-Reasoning-Mini` · `opus-4.7-reasoning-cot` · `refusalbench` · `s1K` ·
`s1K-1.1` · `Supra-Reasoning-CoT-Dataset-v2` · `SWE-bench-extra` · `SWE-rebench` ·
`tiny-think-dpo-math-n-stem`

---

## 7. Gotchas found while measuring

1. **`hf` and `git-lfs` are both missing on this machine.** Nothing can be pulled until
   one is installed. That has not been done — it is a decision, not an oversight.
2. **Four datasets are `gated:auto`** — `Voidreaper2026/coding-master-dataset`,
   `bigcode/the-stack-v2-train-full-ids`, `Vyber07/cyber-security`,
   `bigcode/starcoderdata`. A pull will 401 until the terms are accepted on the site.
3. **Three row/byte ratios say "stub, not dataset":** `Vibe-Coding` (25 B/row),
   `Rust-Coder` (87 B/row), `LIMO`-scale is fine but `Rust-Coder` at 12k rows / 1 MB
   will not teach Rust. Open one row before spending time converting.
4. **`IFM/Code-Reasoning` is 3 TB.** It is not a candidate. If it shows up in a plan,
   the plan is wrong.
5. **Row counts are `?` where the datasets-server has not indexed the repo** (the giant
   ones usually aren't). That is not a failure — it just means size came from file blobs.

---

## 8. The pull ledger — one at a time

Tick one row before starting the next. If a pull is still running, nothing else starts.

| | item | GB | notes |
|---|---|---|---|
| ⬜ | A1 `DeepNLP/Coding-Agent-Github-2025-Feb` | 0.00 | warm-up |
| ⬜ | A2 `Convence/Rust-Coder` | 0.00 | verify a row |
| ⬜ | A3 `greghavens/glm-5.2-…` | 0.00 | |
| ⬜ | A4 `greghavens/kimi-k3-…` | 0.00 | |
| ⬜ | A6 `echel0nn1881/kimi-cyber-reasoning` | 0.01 | |
| ⬜ | A7 `AlicanKiraz0/Agentic-CoT-…` | 0.01 | |
| ⬜ | A8 `Manusagents/Vibe-Coding-…` | 0.03 | verify a row first |
| ⬜ | A9 `Vezora/Tested-22k-Python-Alpaca` | 0.05 | seeds for stage 5 |
| ⬜ | A10 `flytech/python-codes-25k` | 0.05 | |
| ⬜ | A11 `Roman1111111/gemini-3-pro-…` | 0.08 | |
| ⬜ | A12 `armand0e/claude-fable-5-claude-code` | 0.08 | read one session |
| ⬜ | B1 `PentesterFlow/OffsecCyber` | 0.11 | security |
| ⬜ | B2 `smcleod/golang-coder` | 0.17 | |
| ⬜ | B5 `Nexlab/fable5-agentic-coding-sft` | 0.55 | agentic volume |
| ⬜ | A5/B6 `greghavens/gpt-5.6-sol-…` | 0.98 | |
| ⬜ | C1 `MoreThought/Fable-5.1-Max-Reasoning…` | 2.38 | sample, don't epoch |
| ⬜ | C2 `Voidreaper2026/coding-master-dataset` | 3.57 | gated |
| ⬜ | C3 `PrimeIntellect/verifiable-coding-problems` | 10.07 | **the valuable one** |
| ⛔ | D1 `bigcode/the-stack-v2-train-full-ids` | 42.3 | index only, requester-pays S3 |
| ⛔ | D2 `Vyber07/cyber-security` | 88.5 | gated, mostly noise |
| ⛔ | D3 `bigcode/starcoderdata` | 289.5 | gated, raw code |
| ⛔ | D4 `codeparrot/github-code` | 301.7 | raw code |
| ⛔ | D5 `IFM/Code-Reasoning` | 3,052 | exceeds free disk |

Tier A + B + C1–C3 = **~18.2 GB.** That is the whole realistic wishlist against 745 GB
free — it fits forty times over, which is the point: **the value is in the small sets.**

---

## 9. The commands, for when you actually pull

Not run. `#` = optional.

```bash
# one-time tooling (currently MISSING on this machine, nothing installed yet)
python3 -m pip install --user -U huggingface_hub
export PATH="$HOME/Library/Python/3.14/bin:$PATH"

# one dataset, one at a time
hf download Vezora/Tested-22k-Python-Alpaca \
  --repo-type dataset \
  --local-dir datasets/Tested-22k-Python-Alpaca \
  # --include "*.parquet"

# then always these two, before ticking the ledger
du -sh datasets/Tested-22k-Python-Alpaca
python3 - <<'PY'
import glob, json
f = sorted(glob.glob("datasets/Tested-22k-Python-Alpaca/*.jsonl"))[0]
row = json.loads(open(f).readline())
print(f)
print({k: (str(v)[:120]) for k, v in row.items()})
PY

# gated ones need the terms accepted on the website first
# giants: never omit --include
hf download IFM/Code-Reasoning --repo-type dataset --include "*0000*.parquet" --dry-run
```

The two commands after each pull are the point: **measure the folder, and look at one
real row.** Every "this dataset is a stub" trap in §7 is caught by that second command
for free.

---

## 10. Machine state when this sheet was written

```
sweep              KILLED  (layer_sweep.py at L26 terraced; STOP file present)
launchd jobs       none named neuron — nothing will respawn
free RAM           17.9 GB
disk               745 GB free of 926 GB
neuron-surgery/   18.0 GB total
  datasets/        7.3 GB, 36 folders   <- the datasets this sheet is about
  runs/data/       0.5 GB               <- NEURON-NUDGE CORPUS. NOT SFT DATA.
                                           Equal-N sampling (9,802 rows per set)
                                           is deliberate. Out of scope here.
                                           DO NOT TOUCH OR "FIX".
  models/          6.3 GB               <- weights, not data
  .venv + 2 more   3.3 GB               <- python envs
  .git             0.2 GB
hf / git-lfs       MISSING
downloaded for this sheet  0 bytes of data (metadata API calls only)
```

Rows already written by the sweep are untouched in `grid_L*.jsonl`. Restarting later
needs `rm STOP` first, or every runner exits immediately.
