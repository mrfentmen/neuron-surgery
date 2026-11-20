Nanbeige interactive exploration: L12 N2048, dose +0.05

Open Nanbeige Chat.command launches one model on MPS in Terminal.
Thinking text streams verbatim, including <think> and </think> delimiters.
This is generated text, not guaranteed faithful access to internal computation.

Selection was independent of target output. All 32 layers have previous local
observations; this specific layer/channel was absent from all 765 saved local
Nanbeige survey and follow-up replies. Dose +0.05 was also absent. The source
paths and hashes are recorded in selection.json. This is exploratory chat,
not a controlled demonstration of a particular behavior.

The intervention follows the existing Nanbeige survey: add dose times the
original median post-block residual norm divided by the selected down_proj
column norm to that MLP input channel at every prompt/generated position.
The selected column norm is measured at startup; runtime.json records the
requested and actual bf16 scalar. Weights remain unchanged. Hooks are removed
after each generation; the first actual activation change is saved per reply.

/reset clears conversation history. /off selects the undosed model and clears
history; /on restores the fixed intervention and clears history. /status shows
settings. /quit closes the model. There are no automatic prompts or sweeps.

10,000 tokens maximum per reply, INCLUDING thinking and final answer; stops
earlier when the model finishes. Raised from 500 at the user's request after
the first two replies were capped. The original selection.json records the
initial 500-token design; runtime.json and each new reply record the active cap.
Incomplete
outputs remain saved, but are not added to history. A new prompt after a capped
turn is not a continuation of that unfinished response. Completed outputs use
the publisher's conversation template and its default system message.

Seed 20261113 reset each turn; temperature 0.6, top_p 0.95, top_k 0,
repetition penalty 1.0; bf16/MPS, eager attention. Input cap 2048 tokens.
10-second startup/between-reply cooling; 2-second rests every 32 output tokens;
thermal pressure safeguards and low-memory guard. One model lock is shared
with the existing runners. Model-generated text is never executed.

Full responses, reasoning/final splits, exact tokens, rendered prompts, dose
measurements, thermal state, timing, and provenance are saved in sessions/.
The original baseline launcher, runs, and datasets are unchanged.
