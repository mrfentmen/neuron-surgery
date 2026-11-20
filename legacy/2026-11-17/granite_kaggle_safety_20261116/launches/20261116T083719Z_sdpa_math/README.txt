Granite fixed candidate safety evaluation

412 saved candidate settings, 15 questions, 6180 intervention replies and 1740 shared batched controls if the full queue completes.
Two Tesla T4 GPUs, 207 candidates assigned to GPU 0 and 205 to GPU 1.
Independent single-channel rows (up to four plus a control per batch); no combined neurons.
Greedy decoding; maximum 500 tokens; repetition stops. Five-hour session maximum; no automatic restart.
Saved calibration is retained; current float16 column norms and actual additive values are logged.
400 historical candidates lack explicit checkpoint identity; 12 use the verified local plan. These are candidates, not established harmful traits.
Full prompts, rendered prompts, token IDs, outputs, provenance and stop reasons are retained.
Read candidate_audit.json and deferred_candidates.json for screening decisions.
No local model was loaded; no GPU inference on the Mac.

Revision 2: nonfinite intervention rows are saved with numerical_failure diagnostics and skipped; nonfinite baselines stop the run. Version 1 saved zero replies.

Revision 3: float16 weights with SDPA math attention and float32 attention intermediates. This changes numerical implementation, not prompts/doses. Frozen runtime signatures distinguish the runs.
