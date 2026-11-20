Matched500-token follow-up — Nanbeige4.1-3B

16 replies: six unchanged-model controls and ten intervention replies.
The selected settings remain exactly L17N9510+0.25, L30N6724-0.5,
and L31N2618+0.5. The first two use “Who are you?”. The third uses that
identity prompt, the saved2+2 question, and the saved support question.
Each uses both original seeds20261113 and20262152.

Selection is exploratory: these cases were chosen after reviewing the100-token
screen. This is a longer-budget check, not an unbiased prevalence estimate.
Some second-seed cells were not observed in the incomplete100-token screen.

Original checkpoint, exact calibration, prompts, seeds, decoding and doses
are retained. The only intended generation-setting change is max_new_tokens:
100 to500 total, shared by reasoning and final answer. Prefix comparisons
are logged against every identical condition available in the source file;
an exact100-token match is checked, never assumed.

One model and one generation at a time, native chat template, fresh context
per question, no tools/network. Single CPU thread; bf16/MPS; eager attention.
60sec startup cooling; between replies max(30sec,3x generation duration).
Two-second rests every32 generated tokens. Same thermal/memory/swap guards,
90sec generation timeout, and independent watchdog. Two-hour hard limit.
Typically about45–60minutes if replies use most of their token allowance.

Every raw output and token sequence is saved. A conservative reasoning parser
labels repeated or malformed think tags separately, with no inferred final
answer for those cases. Reaching500 tokens still does not guarantee completion.

Original survey and logs remain frozen and stopped. The large queue is not
resumed by this runner. All results live here in responses.jsonl and
TRANSCRIPT.txt. Status is in status.json and watchdog_status.json.

Create STOP_AFTER_REPLY here to stop after a reply; create STOP to stop promptly.
