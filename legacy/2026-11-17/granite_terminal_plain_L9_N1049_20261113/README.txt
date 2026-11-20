Granite independent plain-text terminal — L9 N1049, dose -4

Open Open Granite Plain Text.command. Each line is sent independently via
the ordinary tokenizer, with no chat template or conversation history.
Typed whitespace and capitalization are preserved. /reset confirms this mode;
/quit exits and allows the same54 survey jobs to resume after cooling.

Questions such as “What do you mean?” do not refer to previous replies.
Each raw question, input/output token ids and complete answer are logged.

The source dose-4 and median10.225354194641113 are retained; greedy max500,
10-second cooldown, per32-token rests, thermal checks, one exclusive model,
and OS network denial remain. No new diagnostic questions run on startup.
Historical cloud dtype/device/batch-size differences remain; this is not
a guaranteed exact cloud reproduction. Prior chat and diagnostic logs are
preserved in their original directories.

Historical source: kaggle_out/census_log/screen_L0_19.jsonl.pulled:65984.
