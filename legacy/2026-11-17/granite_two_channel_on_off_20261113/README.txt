Matched dosing on/off comparison: L12 N2556 and L30 N6314

Two channels x three questions x two conditions = 12 new replies.
Each pair receives exactly the same saved conversation and final question. Prior new answers are never fed into another job.
OFF: no intervention hook. ON: the recorded channel setting, unchanged. Greedy decoding for both.
The comparison isolates intervention effects conditional on the saved history; it is not a fresh-context test.
One loaded model, one generation and at most one active channel hook. At most 500 new tokens per reply.
Cooling: at least 60 seconds or 3x generation duration; 2 seconds idle every 32 new tokens; thermal pauses/stops.
Model checkpoint hashes and on-condition calibration must match the completed five-channel batch.
No model tools, network access, executed output, dose search, channel combinations or automatic retries.

Progress: status.json, progress.json, run.log, queue.log.
Results: FOLLOWUPS.txt, PAIRED.txt, TRANSCRIPT.txt, pairs.json, verification.json.
STOP stops immediately; STOP_AFTER_REPLY stops at a complete reply boundary.
