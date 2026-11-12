# Corpus index

Generated 2026-11-09 from `corpus_index.py`. 893 jsonl files, 1,743,590 rows, 5,615 MB.

- unique rows by (tile, probe, dose, answer): **716,589**
- duplicate rows: **1,027,001** (58.9%)
- exact duplicate files: **92** groups

## Where the 5.6 GB lives

| dir | files | MB | rows | what it is |
|---|---:|---:|---:|---|
| `datasets/` | 176 | 4,213 | 0 | 外部 training corpora (not brain scans) |
| `runs/` | 58 | 522 | 206 | full sweep runs |
| `neuron-explorer/` | 130 | 430 | 634,160 | explorer + reference runs |
| `(root)/` | 335 | 183 | 430,266 |  |
| `kg_pull/` | 56 | 105 | 252,137 | kaggle downloads |
| `census/` | 4 | 81 | 229,884 | large census rows |
| `kaggle_out/` | 56 | 64 | 150,341 | kaggle kernel outputs |
| `cloud_gap/` | 13 | 6 | 15,398 | cloud gap-fill |
| `cloud_l10/` | 10 | 4 | 10,739 | cloud L10 |
| `kg_check_pinned_0915/` | 24 | 3 | 6,798 | pinned checks |
| `watch/` | 2 | 2 | 5,634 |  |
| `granite350_fullmap/` | 3 | 2 | 7,818 |  |
| `jailbreak-repos/` | 15 | 0 | 0 |  |
| `attr_kill/` | 10 | 0 | 107 |  |
| `_quarantine_qwen/` | 1 | 0 | 102 |  |

`datasets/` is 75% of the bytes and contains no brain scans. The steering corpus itself is
about 1.2 GB.

## Exact duplicate files

- `screen_L0_19_m6.jsonl` also at `kaggle_out/l0_19_m6/screen_L0_19.jsonl`
- `datasets/Deepseek-v4-pro-max-distill-1500x/output.jsonl` also at `datasets/deepseek-v4-pro-max-distill-1k/output.jsonl`
- `screen_L34_39_m6.jsonl` also at `kaggle_out/l34_39_m6/screen_L34_39.jsonl`
- `grid_L21_k.jsonl` also at `kg_pull/grid_L21_k.jsonl`, `cloud_gap/grid_L21_k.jsonl`
- `grid_L25_k.jsonl` also at `kg_pull/grid_L25_k.jsonl`, `cloud_gap/grid_L25_k.jsonl`
- `grid_L23_k.jsonl` also at `kg_pull/grid_L23_k.jsonl`, `cloud_gap/grid_L23_k.jsonl`
- `grid_L22_k.jsonl` also at `kg_pull/grid_L22_k.jsonl`, `cloud_gap/grid_L22_k.jsonl`
- `grid_L24_k.jsonl` also at `kg_pull/grid_L24_k.jsonl`, `cloud_gap/grid_L24_k.jsonl`
- `grid_L20_k.jsonl` also at `kg_pull/grid_L20_k.jsonl`, `cloud_gap/grid_L20_k.jsonl`
- `grid_L26_k.jsonl` also at `kg_pull/grid_L26_k.jsonl`, `cloud_gap/grid_L26_k.jsonl`
- `grid_L27_k.jsonl` also at `kg_pull/grid_L27_k.jsonl`, `cloud_gap/grid_L27_k.jsonl`
- `grid_L28_k.jsonl` also at `kg_pull/grid_L28_k.jsonl`, `cloud_gap/grid_L28_k.jsonl`
- `grid_L31_k.jsonl` also at `kg_pull/grid_L31_k.jsonl`, `cloud_gap/grid_L31_k.jsonl`
- `grid_L15_k.jsonl` also at `cloud_l10/grid_L15_k.jsonl`
- `grid_L29_k.jsonl` also at `kg_pull/grid_L29_k.jsonl`, `cloud_gap/grid_L29_k.jsonl`
- `grid_L30_k.jsonl` also at `kg_pull/grid_L30_k.jsonl`, `cloud_gap/grid_L30_k.jsonl`
- `grid_L14_k.jsonl` also at `cloud_l10/grid_L14_k.jsonl`
- `grid_L32_k.jsonl` also at `kg_pull/grid_L32_k.jsonl`, `cloud_gap/grid_L32_k.jsonl`
- `grid_L10_k.jsonl` also at `cloud_l10/grid_L10_k.jsonl`
- `grid_L11_k.jsonl` also at `cloud_l10/grid_L11_k.jsonl`
- `grid_L16_k.jsonl` also at `cloud_l10/grid_L16_k.jsonl`
- `grid_L13_k.jsonl` also at `cloud_l10/grid_L13_k.jsonl`
- `grid_L12_k.jsonl` also at `cloud_l10/grid_L12_k.jsonl`
- `grid_L19_k.jsonl` also at `cloud_l10/grid_L19_k.jsonl`
- `grid_L17_k.jsonl` also at `cloud_l10/grid_L17_k.jsonl`
- `grid_L18_k.jsonl` also at `cloud_l10/grid_L18_k.jsonl`
- `grid_L17_col.jsonl` also at `kg_check_pinned_0915/grid_L17_col.jsonl`
- `grid_L19_col.jsonl` also at `kg_check_pinned_0915/grid_L19_col.jsonl`
- `grid_L14_col.jsonl` also at `kg_check_pinned_0915/grid_L14_col.jsonl`
- `grid_L15_col.jsonl` also at `kg_check_pinned_0915/grid_L15_col.jsonl`
- `grid_L13_col.jsonl` also at `kg_check_pinned_0915/grid_L13_col.jsonl`
- `grid_L21_col.jsonl` also at `kg_check_pinned_0915/grid_L21_col.jsonl`
- `grid_L20_col.jsonl` also at `kg_check_pinned_0915/grid_L20_col.jsonl`
- `grid_L18_col.jsonl` also at `kg_check_pinned_0915/grid_L18_col.jsonl`
- `grid_L9_col.jsonl` also at `kg_check_pinned_0915/grid_L9_col.jsonl`
- `grid_L12_col.jsonl` also at `kg_check_pinned_0915/grid_L12_col.jsonl`
- `grid_L22_col.jsonl` also at `kg_check_pinned_0915/grid_L22_col.jsonl`
- `grid_L16_col.jsonl` also at `kg_check_pinned_0915/grid_L16_col.jsonl`
- `grid_L8_col.jsonl` also at `kg_check_pinned_0915/grid_L8_col.jsonl`
- `grid_L11_col.jsonl` also at `kg_check_pinned_0915/grid_L11_col.jsonl`
- `grid_L10_col.jsonl` also at `kg_check_pinned_0915/grid_L10_col.jsonl`
- `grid_L7_col.jsonl` also at `kg_check_pinned_0915/grid_L7_col.jsonl`
- `grid_L6_col.jsonl` also at `kg_check_pinned_0915/grid_L6_col.jsonl`
- `grid_L3_col.jsonl` also at `kg_check_pinned_0915/grid_L3_col.jsonl`
- `grid_L5_col.jsonl` also at `kg_check_pinned_0915/grid_L5_col.jsonl`
- `grid_L4_col.jsonl` also at `kg_check_pinned_0915/grid_L4_col.jsonl`
- `grid_L1_col.jsonl` also at `kg_check_pinned_0915/grid_L1_col.jsonl`
- `grid_L2_col.jsonl` also at `kg_check_pinned_0915/grid_L2_col.jsonl`
- `grid_L0_col.jsonl` also at `kg_check_pinned_0915/grid_L0_col.jsonl`
- `grid_L23_col.jsonl` also at `kg_check_pinned_0915/grid_L23_col.jsonl`
- `kg_pull/unhit/grid_L39_u.jsonl` also at `kaggle_out/unhit/grid_L39_u.jsonl`
- `kg_pull/unhit/grid_L16_u.jsonl` also at `kaggle_out/unhit/grid_L16_u.jsonl`
- `kg_pull/unhit/grid_L13_u.jsonl` also at `kaggle_out/unhit/grid_L13_u.jsonl`
- `kg_pull/unhit/grid_L15_u.jsonl` also at `kaggle_out/unhit/grid_L15_u.jsonl`
- `kg_pull/unhit/grid_L21_u.jsonl` also at `kaggle_out/unhit/grid_L21_u.jsonl`
- `kg_pull/unhit/grid_L24_u.jsonl` also at `kaggle_out/unhit/grid_L24_u.jsonl`
- `kg_pull/unhit/grid_L25_u.jsonl` also at `kaggle_out/unhit/grid_L25_u.jsonl`
- `kg_pull/unhit/grid_L22_u.jsonl` also at `kaggle_out/unhit/grid_L22_u.jsonl`
- `kg_pull/unhit/grid_L27_u.jsonl` also at `kaggle_out/unhit/grid_L27_u.jsonl`
- `kg_pull/unhit/grid_L14_u.jsonl` also at `kaggle_out/unhit/grid_L14_u.jsonl`
- `kg_pull/unhit/grid_L17_u.jsonl` also at `kaggle_out/unhit/grid_L17_u.jsonl`
- `kg_pull/unhit/grid_L26_u.jsonl` also at `kaggle_out/unhit/grid_L26_u.jsonl`
- `kg_pull/unhit/grid_L28_u.jsonl` also at `kaggle_out/unhit/grid_L28_u.jsonl`
- `kg_pull/unhit/grid_L18_u.jsonl` also at `kaggle_out/unhit/grid_L18_u.jsonl`
- `kg_pull/unhit/grid_L8_u.jsonl` also at `kaggle_out/unhit/grid_L8_u.jsonl`
- `kg_pull/unhit/grid_L29_u.jsonl` also at `kaggle_out/unhit/grid_L29_u.jsonl`
- `kg_pull/unhit/grid_L19_u.jsonl` also at `kaggle_out/unhit/grid_L19_u.jsonl`
- `kg_pull/unhit/grid_L20_u.jsonl` also at `kaggle_out/unhit/grid_L20_u.jsonl`
- `screen_L0_19_gap.jsonl` also at `kaggle_out/gap_log/screen_L0_19.jsonl`
- `kg_pull/unhit/grid_L30_u.jsonl` also at `kaggle_out/unhit/grid_L30_u.jsonl`
- `kg_pull/unhit/grid_L23_u.jsonl` also at `kaggle_out/unhit/grid_L23_u.jsonl`
- `kg_pull/unhit/grid_L9_u.jsonl` also at `kaggle_out/unhit/grid_L9_u.jsonl`
- `kg_pull/unhit/grid_L10_u.jsonl` also at `kaggle_out/unhit/grid_L10_u.jsonl`
- `kg_pull/unhit/grid_L6_u.jsonl` also at `kaggle_out/unhit/grid_L6_u.jsonl`
- `kg_pull/unhit/grid_L12_u.jsonl` also at `kaggle_out/unhit/grid_L12_u.jsonl`
- `kg_pull/unhit/grid_L32_u.jsonl` also at `kaggle_out/unhit/grid_L32_u.jsonl`
- `kg_pull/unhit/grid_L11_u.jsonl` also at `kaggle_out/unhit/grid_L11_u.jsonl`
- `kg_pull/unhit/grid_L35_u.jsonl` also at `kaggle_out/unhit/grid_L35_u.jsonl`
- `kg_pull/unhit/grid_L7_u.jsonl` also at `kaggle_out/unhit/grid_L7_u.jsonl`
- `kg_pull/unhit/grid_L36_u.jsonl` also at `kaggle_out/unhit/grid_L36_u.jsonl`
- `kg_pull/unhit/grid_L31_u.jsonl` also at `kaggle_out/unhit/grid_L31_u.jsonl`
- `kg_pull/unhit/grid_L38_u.jsonl` also at `kaggle_out/unhit/grid_L38_u.jsonl`
- `kg_pull/unhit/grid_L33_u.jsonl` also at `kaggle_out/unhit/grid_L33_u.jsonl`
- `kg_pull/unhit/grid_L4_u.jsonl` also at `kaggle_out/unhit/grid_L4_u.jsonl`
- `kg_pull/unhit/grid_L34_u.jsonl` also at `kaggle_out/unhit/grid_L34_u.jsonl`
- `kg_pull/unhit/grid_L5_u.jsonl` also at `kaggle_out/unhit/grid_L5_u.jsonl`
- `kg_pull/unhit/grid_L37_u.jsonl` also at `kaggle_out/unhit/grid_L37_u.jsonl`
- `kg_pull/unhit/grid_L3_u.jsonl` also at `kaggle_out/unhit/grid_L3_u.jsonl`
- `kg_pull/unhit/grid_L2_u.jsonl` also at `kaggle_out/unhit/grid_L2_u.jsonl`
- `kg_pull/unhit/grid_L0_u.jsonl` also at `kaggle_out/unhit/grid_L0_u.jsonl`
- `kg_pull/unhit/grid_L1_u.jsonl` also at `kaggle_out/unhit/grid_L1_u.jsonl`
- `watson_military.jsonl` also at `watson_dose_m3.jsonl`

## Contents

- `datasets/NOESIS-1M-reasoning-router-code-math-psych-opus47-deepseek4-qwen36-gemini31-r1-gpt54/NOESIS-1M-multilingual-reasoning-router-general-code-math-psych-aya-sft-claude-sonet46-opus47-deepseek4-qwen36-gemini31-r1-gpt54.jsonl` — 1431.02 MB, 0 rows, —, doses [], probe []
- `datasets/Claude-Sonnet-Opus/Claude_Sonnet_Opus.jsonl` — 928.73 MB, 0 rows, —, doses [], probe []
- `datasets/Claude-Distills/claude_distill.jsonl` — 911.0 MB, 0 rows, —, doses [], probe []
- `datasets/Astra-COT-40K/EN-dataset.jsonl` — 175.53 MB, 0 rows, —, doses [], probe []
- `datasets/DeepSeek-V4-Distill-8000x/data/train.jsonl` — 141.91 MB, 0 rows, —, doses [], probe []
- `runs/data/Claude-Sonnet-Opus.jsonl` — 76.9 MB, 0 rows, —, doses [], probe []
- `datasets/claude-opus-4.6-4.7-reasoning-8.7k/full_train.jsonl` — 71.45 MB, 0 rows, —, doses [], probe []
- `runs/data/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B.jsonl` — 70.22 MB, 0 rows, —, doses [], probe []
- `datasets/Fable-5-traces/fable5_cot_merged.jsonl` — 69.84 MB, 0 rows, —, doses [], probe []
- `runs/data/DeepSeek-V4-Distill-8000x.jsonl` — 69.19 MB, 0 rows, —, doses [], probe []
- `kg_pull/nov8/screen_L20_39.jsonl` — 68.31 MB, 159,224 rows, L20-39, doses [-4], probe ['Who are you?']
- `screen_L0_19.jsonl` — 64.69 MB, 150,992 rows, L0-19, doses [-4], probe ['Who are you?']
- `datasets/claude-opus-4.6-4.7-reasoning-8.7k/instruct_train.jsonl` — 60.61 MB, 0 rows, —, doses [], probe []
- `census/labeled.jsonl` — 56.67 MB, 150,992 rows, L0-19, doses [-4], probe ['Who are you?']
- `datasets/Claude-Distills/claude_merged.jsonl` — 48.39 MB, 0 rows, —, doses [], probe []
- `datasets/claude-opus-4.6-4.7-reasoning-8.7k/full_train_no_reasoning.jsonl` — 48.36 MB, 0 rows, —, doses [], probe []
- `runs/data/claude-opus-4.6-4.7-reasoning-8.7k.jsonl` — 42.88 MB, 0 rows, —, doses [], probe []
- `datasets/claude-opus-4.6-4.7-reasoning-8.7k/instruct_train_no_reasoning.jsonl` — 40.96 MB, 0 rows, —, doses [], probe []
- `screen_L0_19_m6.jsonl` — 39.34 MB, 93,686 rows, L0-12, doses [-6], probe ['Who are you?']
- `kaggle_out/l0_19_m6/screen_L0_19.jsonl` — 39.34 MB, 93,686 rows, L0-12, doses [-6], probe ['Who are you?']
- `datasets/Open-CoT-Reasoning-Mini/Datasets/Open-CoT-Reasoning-Mini.jsonl` — 38.65 MB, 0 rows, —, doses [], probe []
- `runs/data/Open-CoT-Reasoning-Mini.jsonl` — 37.45 MB, 0 rows, —, doses [], probe []
- `runs/data/SWE-rebench.jsonl` — 33.0 MB, 0 rows, —, doses [], probe []
- `runs/data/SWE-bench-extra.jsonl` — 31.21 MB, 0 rows, —, doses [], probe []
- `runs/data/Astra-COT-40K.jsonl` — 29.87 MB, 0 rows, —, doses [], probe []
- `datasets/Deepseek-v4-pro-max-distill-1500x/output.jsonl` — 26.15 MB, 0 rows, —, doses [], probe []
- `datasets/deepseek-v4-pro-max-distill-1k/output.jsonl` — 26.15 MB, 0 rows, —, doses [], probe []
- `census/notable.jsonl` — 24.07 MB, 76,695 rows, L0-19, doses [-4], probe ['?']
- `datasets/deepseek-v4.1-flash-distill-zh-2k/data/train.jsonl` — 22.09 MB, 0 rows, —, doses [], probe []
- `runs/data/Claude-Distills.jsonl` — 21.94 MB, 0 rows, —, doses [], probe []
- `screen_L34_39_m6.jsonl` — 21.5 MB, 48,313 rows, L34-39, doses [-6], probe ['Who are you?']
- `kaggle_out/l34_39_m6/screen_L34_39.jsonl` — 21.5 MB, 48,313 rows, L34-39, doses [-6], probe ['Who are you?']
- `runs/data/Magpie-Reasoning-V1-150K.jsonl` — 20.63 MB, 0 rows, —, doses [], probe []
- `datasets/DiagSafe/rubrics/DiagSafe_rubrics_3489.jsonl` — 18.83 MB, 0 rows, —, doses [], probe []
- `kg_pull/nov8/screen_L34_39.jsonl` — 18.68 MB, 48,347 rows, L34-39, doses [6], probe ['how do you feel about humans']
- `datasets/deepseek-v4-pro-math-cot-1k/train_core.jsonl` — 18.25 MB, 0 rows, —, doses [], probe []
- `datasets/LIMO/limo.jsonl` — 16.07 MB, 0 rows, —, doses [], probe []
- `runs/data/LIMO.jsonl` — 15.9 MB, 0 rows, —, doses [], probe []
- `m6_fluent.jsonl` — 12.46 MB, 27,708 rows, L0-39, doses [-6], probe ['Who are you?']
- `datasets/claude-opus-4.6-4.7-reasoning-8.7k/code_train.jsonl` — 12.45 MB, 0 rows, —, doses [], probe []
- `datasets/deepseek-v4-pro-math-cot-1k/pathological_cot_quarantine.jsonl` — 12.21 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L39.jsonl` — 10.85 MB, 25,574 rows, L39-39, doses [-8, -6, -5, -4, -3, 3, 4, 5, 6, 8], probe ['Who are you?']
- `datasets/claude-opus-4.6-4.7-reasoning-8.7k/roleplay_train.jsonl` — 10.83 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L38.jsonl` — 10.38 MB, 25,736 rows, L38-38, doses [-8, -6, -5, -4, -3, 3, 4, 5, 6, 8], probe ['Who are you?']
- `kg_pull/nov8/screen_L31_33.jsonl` — 10.25 MB, 24,392 rows, L31-33, doses [-6], probe ['Who are you?']
- `runs/data/opus-4.7-reasoning-cot.jsonl` — 10.21 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L36.jsonl` — 10.17 MB, 25,761 rows, L36-36, doses [-8, -6, -5, -4, -3, 3, 4, 5, 6, 8], probe ['Who are you?']
- `neuron-explorer/runs/granite-all-L35.jsonl` — 10.15 MB, 25,687 rows, L35-35, doses [-8, -6, -5, -4, -3, 3, 4, 5, 6, 8], probe ['Who are you?']
- `neuron-explorer/runs/granite-all-L37.jsonl` — 10.14 MB, 25,645 rows, L37-37, doses [-8, -6, -5, -4, -3, 3, 4, 5, 6, 8], probe ['Who are you?']
- `neuron-explorer/runs/granite-all-L34.jsonl` — 10.06 MB, 25,594 rows, L34-34, doses [-8, -6, -5, -4, -3, 3, 4, 5, 6, 8], probe ['Who are you?']
- `datasets/Ling-Tiny-SFT/train.jsonl` — 9.07 MB, 0 rows, —, doses [], probe []
- `runs/data/opc-sft-stage2.jsonl` — 8.28 MB, 0 rows, —, doses [], probe []
- `datasets/claude-opus-4.6-4.7-reasoning-8.7k/code_train_no_reasoning.jsonl` — 8.23 MB, 0 rows, —, doses [], probe []
- `datasets/deepseek-v4-pro-math-cot-1k/rejects.jsonl` — 7.94 MB, 0 rows, —, doses [], probe []
- `datasets/math-rlvr-15k/math-rlvr-15k.jsonl` — 7.57 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L31.jsonl` — 7.42 MB, 18,585 rows, L31-31, doses [-8, -6, -5, -4, -3, 3, 4, 5, 6, 8], probe ['Who are you?']
- `datasets/claude-opus-4.6-4.7-reasoning-8.7k/roleplay_train_no_reasoning.jsonl` — 7.4 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L11.jsonl` — 7.31 MB, 17,860 rows, L11-11, doses [-8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8], probe ['Who are you?']
- `neuron-explorer/runs/granite-all-L10.jsonl` — 7.28 MB, 17,842 rows, L10-10, doses [-8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8], probe ['Who are you?']
- `neuron-explorer/runs/granite-all-L9.jsonl` — 7.24 MB, 17,733 rows, L9-9, doses [-8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8], probe ['Who are you?']
- `neuron-explorer/runs/granite-all-L12.jsonl` — 7.21 MB, 17,444 rows, L12-12, doses [-8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8], probe ['Who are you?']
- `neuron-explorer/runs/granite-all-L8.jsonl` — 7.2 MB, 17,932 rows, L8-8, doses [-8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8], probe ['Who are you?']
- `datasets/cot-reasoning-2k/data/train.jsonl` — 7.17 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L7.jsonl` — 7.08 MB, 17,613 rows, L7-7, doses [-8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8], probe ['Who are you?']
- `neuron-explorer/runs/granite-all-L32.jsonl` — 7.07 MB, 17,580 rows, L32-32, doses [-8, -6, -5, -4, -3, 3, 4, 5, 6, 8], probe ['Who are you?']
- `neuron-explorer/runs/granite-all-L6.jsonl` — 7.07 MB, 17,572 rows, L6-6, doses [-8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8], probe ['Who are you?']
- `runs/data/Ling-Tiny-SFT.jsonl` — 7.04 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L5.jsonl` — 6.98 MB, 17,636 rows, L5-5, doses [-8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8], probe ['Who are you?']
- `neuron-explorer/runs/granite-all-L3.jsonl` — 6.93 MB, 17,570 rows, L3-3, doses [-8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8], probe ['Who are you?']
- `neuron-explorer/runs/granite-all-L33.jsonl` — 6.88 MB, 17,005 rows, L33-33, doses [-8, -6, -5, -4, -3, 3, 4, 5, 6, 8], probe ['Who are you?']
- `runs/data/cot-reasoning-2k.jsonl` — 6.87 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L1.jsonl` — 6.84 MB, 17,570 rows, L1-1, doses [-8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8], probe ['Who are you?']
- `neuron-explorer/runs/granite-all-L2.jsonl` — 6.78 MB, 17,502 rows, L2-2, doses [-8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8], probe ['Who are you?']
- `neuron-explorer/runs/granite-all-L4.jsonl` — 6.68 MB, 17,317 rows, L4-4, doses [-8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8], probe ['Who are you?']
- `runs/data/Deepseek-v4-pro-max-distill-1500x.jsonl` — 5.59 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L0.jsonl` — 5.27 MB, 13,852 rows, L0-0, doses [-8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8], probe ['Who are you?']
- `neuron-explorer/runs/granite-all-L16.jsonl` — 4.97 MB, 12,264 rows, L16-16, doses [-8, -6, -5, -4, -3, 3, 4, 5, 6, 8], probe ['Who are you?']
- `runs/data/deepseek-v4-pro-max-distill-1k.jsonl` — 4.96 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L38.units.jsonl` — 4.88 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L37.units.jsonl` — 4.84 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L36.units.jsonl` — 4.77 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L35.units.jsonl` — 4.76 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L34.units.jsonl` — 4.7 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L8.units.jsonl` — 4.57 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L11.units.jsonl` — 4.56 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L9.units.jsonl` — 4.56 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L10.units.jsonl` — 4.53 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L12.units.jsonl` — 4.53 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L15.jsonl` — 4.52 MB, 11,133 rows, L15-15, doses [-8, -6, -5, -4, -3, 3, 4, 5, 6, 8], probe ['Who are you?']
- `neuron-explorer/runs/granite-all-L14.jsonl` — 4.49 MB, 11,066 rows, L14-14, doses [-8, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8], probe ['Who are you?']
- `runs/data/deepseek-v4.1-flash-distill-zh-2k.jsonl` — 4.48 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L24.jsonl` — 4.47 MB, 11,073 rows, L24-24, doses [-8, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8], probe ['Who are you?']
- `neuron-explorer/runs/granite-all-L7.units.jsonl` — 4.45 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L26.jsonl` — 4.41 MB, 11,057 rows, L26-26, doses [-8, -6, -5, -4, -3, 3, 4, 5, 6, 8], probe ['Who are you?']
- `neuron-explorer/runs/granite-all-L6.units.jsonl` — 4.39 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L17.jsonl` — 4.36 MB, 10,748 rows, L17-17, doses [-8, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8], probe ['Who are you?']
- `neuron-explorer/runs/granite-all-L4.units.jsonl` — 4.36 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L14.units.jsonl` — 4.32 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L5.units.jsonl` — 4.32 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L3.units.jsonl` — 4.32 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L13.jsonl` — 4.32 MB, 10,471 rows, L13-13, doses [-8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8], probe ['Who are you?']
- `neuron-explorer/runs/granite-all-L13.units.jsonl` — 4.31 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L2.units.jsonl` — 4.31 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L1.units.jsonl` — 4.3 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L25.jsonl` — 4.3 MB, 10,723 rows, L25-25, doses [-8, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8], probe ['Who are you?']
- `neuron-explorer/runs/granite-all-L28.jsonl` — 4.29 MB, 10,811 rows, L28-28, doses [-8, -6, -5, -4, -3, 3, 4, 5, 6, 8], probe ['Who are you?']
- `neuron-explorer/runs/granite-all-L30.jsonl` — 4.26 MB, 10,816 rows, L30-30, doses [-8, -6, -5, -4, -3, 3, 4, 5, 6, 8], probe ['Who are you?']
- `neuron-explorer/runs/granite-all-L16.units.jsonl` — 4.25 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L0.units.jsonl` — 4.24 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L15.units.jsonl` — 4.23 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L24.units.jsonl` — 4.23 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L21.units.jsonl` — 4.08 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L39.units.jsonl` — 4.07 MB, 0 rows, —, doses [], probe []
- `datasets/DiagSafe/prompts/DiagSafe_bench_3489.jsonl` — 4.06 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L20.units.jsonl` — 4.03 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L19.units.jsonl` — 4.02 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L17.units.jsonl` — 4.0 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L31.units.jsonl` — 3.99 MB, 0 rows, —, doses [], probe []
- `neuron-explorer/runs/granite-all-L20.jsonl` — 3.96 MB, 9,812 rows, L20-20, doses [-8, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8], probe ['Who are you?']
- `neuron-explorer/runs/granite-all-L25.units.jsonl` — 3.95 MB, 0 rows, —, doses [], probe []

---

## `layers/` — the consolidated brain (verified 2026-11-12)

This index was generated 2026-11-09, before the Kaggle census was merged, so the
totals above describe the older snapshot. For the consolidated per-layer corpus:

| what | value | how |
|---|---:|---|
| files | 40 (`L00.jsonl` … `L39.jsonl`, plus `.gz`) | `ls layers/L*.jsonl` |
| rows | 716,402 | `verify_coverage.py` |
| distinct tiles | 326,793 / 327,680 (99.73%) | `verify_coverage.py` |
| thinnest layer | L19, 8,085 tiles | `verify_coverage.py` |

Row schema: `{layer, neuron, dose, probe, answer, src, label}`. `src` names the
file the row came from, so every row is traceable to its run.

Do not quote row counts from the tables above for anything post-November-8; use
`STATUS-NOW.md` and `verify_coverage.py`.
