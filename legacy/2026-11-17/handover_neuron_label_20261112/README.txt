Queued descriptive-label pass for L27 N3448

Waits for the 144-reply handover batch and its transcript export to finish.
Then writes LABEL.txt, LABEL.json and the eight target replies in TARGET_REPLIES.txt.
The proposed label is a provisional behavioral description from the previously
reviewed pilot, paired with explicitly provisional facts from the new batch.
It is not a model-generated personal name or a claim of an internal identity.

This job loads no model and performs no new generations. It uses negligible CPU
while waiting, and stops if its parent batch fails, is stopped, or disappears.
It changes none of the running experiment's scripts, settings, or outputs.

Status: status.json. Log: queue.log. Maximum wait: eight hours.
To cancel only this naming pass, create STOP in this directory.
