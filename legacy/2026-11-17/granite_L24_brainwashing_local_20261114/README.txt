Local Granite L24 N3392 runner

24 replies: six questions x fresh/saved-claim history x intervention on/off. Order alternates intervention/control; no sampling repeats.
Saved normalized dose -4, additive bf16 offset -90.5 into channel 3392 before layer 24 down_proj. Zero-based indices; weights unchanged.
MPS GPU, bfloat16, greedy, repetition penalty 1.0, 500 new-token maximum. All full prompts, histories, answers and stop reasons saved.
One reply at a time, >=10-second cooling or generation duration, two-second rest every 32 tokens, thermal/memory watchdog and exact repetition guard.
Stops after all 24 replies or one hour, whichever comes first. Detached from Codex; old stopped/paused batches are not resumed.
To request a clean stop: run .venv/bin/python granite_L24_brainwashing_local_20261114/stop.py from the project root.
