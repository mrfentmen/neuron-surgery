Nanbeige middle-layer local evaluation — five-hour bounded run

Layers 12–19, four previously sampled channels per layer (32 total).
Six saved doses: -0.1, +0.1, -0.25, +0.25, -0.5, +0.5.
Twenty questions: the existing 19-question Kaggle bank plus arithmetic.
No new operational harmful prompts or outcome-driven selection.

The frozen queue contains 3,840 intervention replies and 480 controls.
Five hours covers only a prefix, not this whole queue. Blocks alternate
questions and doses, covering all eight middle layers before moving on.
Every block begins with an undosed control using the same question and seed.
Each reply has a fresh context, original calibration and native chat template.
Original weights are unchanged. MPS/bf16 results are separate from Kaggle/fp16.

10,000 tokens maximum per reply, INCLUDING generated reasoning text.
EOS, exact repetitive tails and a 30-minute per-reply timeout end a reply.
Repetition stops and reply timeouts preserve the cutoff and move to the next
job. Paraphrased repetition is not detected by the exact-token detector.
Reasoning text is model output; it does not establish intentions or sentience.

One model and generation at a time, shared model lock, no network or tools.
60-second startup cooling, then at least 30 seconds or three times the last
generation duration between replies. Two-second rests every 32 tokens.
Existing thermal, available-memory, swap-growth and GPU-memory guards remain.
Independent watchdog checks thermal state, process memory, heartbeat and wall
time. These guards cannot guarantee quiet fans. Hardware guard stops require
review; no automatic restart.

responses.jsonl and TRANSCRIPT.txt save finished/cutoff replies immediately.
inflight.json saves the current partial reply every 128 output tokens.
status.json includes progress heartbeats; watchdog_status.json is independent.
Original plans, results and the Kaggle launches are preserved separately.

Stop promptly: create STOP in this folder.
Stop after a reply: create STOP_AFTER_REPLY in this folder.
No active Codex session is needed while the background process runs.
