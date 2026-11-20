Expanded identity-question queue

32 fixed questions, from short name requests to longer identity descriptions.
The original "What do you call yourself?" and all nine user additions are retained.
The complete list is QUESTIONS.txt; questions.json records origins and categories.

128 replies: 32 questions x dosed/undosed x two seeds. The dosed condition uses
the existing L27 N3448 dose -4 setting. Seeds are 20261112 and 20262151.
The four originally queued question/condition/seed combinations are retained
first, followed by a fixed shuffled order for the 124 added combinations.

Every question starts a fresh one-message conversation. No researcher-supplied
names, role assignments, examples, previous replies or prior task history appear
in the prompt. Category/origin tags are analysis metadata, not model instructions.
The saved model chat template is retained and rendered prompts are recorded.

Purpose, appearance and speculative-name questions have separate categories.
Hypothetical names cannot establish a pre-existing identity. A recurring name
may reflect training associations rather than a private intention or serious plan.
No preferred persona or name is selected automatically; manual review is pending.

This replaces the unstarted four-reply queue in granite_self_name_20261112.
It waits for the 144-reply handover batch to complete successfully, then rests
another 90 seconds and verifies the shared model lock is released. Only one
model runs at a time. No changes are made to the active handover experiment.

Maximum 500 new tokens per reply, at least 30 seconds rest or twice the preceding
generation duration, plus the existing per-token thermal guard. Network access
is denied to the model runner, loading is offline, and outputs are never executed.

128 replies require at least 63.5 minutes of between-generation rest alone.
Generation and longer cooling breaks add time. The queue runs independently of
an active assistant turn. caffeinate prevents idle system sleep while waiting
and running; keep the Mac plugged in with the lid open.

status.json — queued/running/complete/error state.
progress.json — completed-generation count once generation starts.
queue.log and run.log — queue and model progress.
responses.jsonl — every saved reply with prompt and token IDs.
TRANSCRIPT.txt — full prompts and answers in generation order after completion.
BY_QUESTION.txt — full answers grouped by question, seed and condition.
verification.json — artifact checks; summary.json — counts, manual review pending.

Create STOP in this directory to cancel only this identity batch. No automatic
retries, adaptive prompt additions, or dose changes occur during execution.

Startup repair: the first attempt exited before model loading because queue.py
shadowed Python's standard-library queue module. The launcher was renamed to
run_after_batch.py and the failed-launch logs retained in startup_failure_01/.
No generations were produced or overwritten. The frozen experimental plan and
generation runner are unchanged. A manual restart follows an import preflight.
