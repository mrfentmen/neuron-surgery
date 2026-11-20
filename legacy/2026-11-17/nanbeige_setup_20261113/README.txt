Nanbeige4.1-3B baseline setup — 2026-11-13

Original publisher: Nanbeige/Nanbeige4.1-3B
Pinned revision: 9c4555c37921c0982af8fffeee1e0a6e6b953c4f
Location: ../models/Nanbeige4.1-3B
download_manifest.json records publisher checksums and verification status.

Double-click Open Nanbeige Chat.command to start the baseline terminal.
If Granite is open, the launcher waits without loading a second model.
Enter /quit in the Granite window when ready. The existing 54-question survey
is held via its supported STOP_AFTER_REPLY flag; its plan and results remain
unchanged. Resuming that survey is a separate step after the interactive work.

Baseline means no neuron intervention and no edited weights. The model uses
its own published chat template, including its default system message.
Conversation history is included, unlike the previous independent plain-text
Granite terminal. /reset clears the conversation, retaining saved records.
Completed reasoning is handled by the publisher's template on subsequent turns.

Generated text streams with thinking delimiters visible. Raw output, token IDs,
rendered prompts, reasoning/final splits, settings, timing and thermal state
are saved under sessions/. Model text is never executed. OS networking is
denied while chatting; checkpoint files are loaded locally with remote code off.

The output cap is 500 tokens TOTAL for reasoning plus final answer. If reasoning
does not close, it is labelled incomplete and is not added to conversation history.
The launcher does not raise the token cap to force a final answer.

One loaded model at a time; 10-second startup/between-reply breaks; 2-second
rests every 32 generated tokens; inherited thermal-pressure safeguards.
Single CPU thread, bf16/MPS with fallback disabled, eager attention, 2048-token
input cap. Sampling: temperature 0.6, top_p 0.95, top_k 0, repetition penalty 1,
seed 0 reset per turn. The GPU allocator fraction is capped at 0.5 of its
recommended limit; total OS/process memory can be larger than the allocator.

This setup alone is not an activation-steering finding. Visible reasoning is
generated text, not a guaranteed complete or faithful account of computation.
No Granite neuron coordinates or doses are transferred to this checkpoint.
