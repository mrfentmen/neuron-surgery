Background individual-channel handover comparison

144 planned generations: 16 existing single-channel settings, baseline and the
existing random-channel control; two record packs, two frames, two seeds.
Only one channel is active in any intervened generation. No channel combinations.

All recorded settings, prompts and job order are frozen in plan.json. The local
3B Granite checkpoint is checked against the previous handover pilot hashes.
Maximum 500 new tokens, one generation at a time. At least 30 seconds rest or
twice the previous generation duration; additional short pauses during decoding.
macOS thermal pressure is monitored: fair pauses, serious/critical stops.
Thermal pressure is not a direct fan-speed or temperature measurement.

The detached supervisor does not require an active assistant turn. It invokes
the model runner with network access denied, saves text only, and never executes
generated replies. It does not restart failed or interrupted runs.

Progress: progress.json and run.log.
Raw answers while running: responses.jsonl.
Success: completion.json, then FINISHED.txt after export.
Failures: failure.json if the runner catches the error; background_status.json
records the supervisor's exit codes even when the runner cannot save an error.

When complete:
TRANSCRIPT.txt — all 144 exact prompts and complete captured replies.
automatic_summary.json — provisional record-fact checks only.
BLINDED.txt / manual_review_template.json — prepared for later manual review.
blind_key.json — keep unopened until review is recorded.

Automatic counts are not a claim of malicious intent or a substitute for reading
answers. Manual review and a final behavioral report remain pending on completion.

To stop gracefully, create an empty file named STOP in this directory.
The runner checks it between tokens and during cooling. Partial results are kept.

For progress from Terminal:
cat /Users/test/Desktop/neuron-surgery/handover_channels_20261112/progress.json

For a graceful stop from Terminal:
touch /Users/test/Desktop/neuron-surgery/handover_channels_20261112/STOP

The launcher uses macOS caffeinate -i while the job runs to prevent idle system
sleep, without changing persistent sleep settings. Keep the Mac plugged in with
its lid open; closing the lid or shutting it down can suspend or stop the job.
Closing the assistant turn does not end the detached process.
