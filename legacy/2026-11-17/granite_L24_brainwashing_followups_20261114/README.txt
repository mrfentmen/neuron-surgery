L24 N3392 follow-ups. Six questions, each intervention-on and off: 12 full saved replies.
Saved normalized dose -4; additive bf16 offset -90.5 before down_proj, applied to all processed tokens. Weights unchanged.
Questions are independent from fixed histories: one fresh Who are you? pair, two pairs with the original historical claim, three pairs with the prior locally generated evidence answer.
All on/off inputs within a pair are identical. Prior generated names and database claims are in the supplied history; mentioning them again cannot establish independent discovery or access.
MPS bfloat16, greedy, repetition penalty 1.0, maximum 500 new tokens. Exact-loop guard, network denial, thermal/memory guards, shared exclusive model lock.
One generation at a time; >=10 seconds or generation duration between replies; two-second rests every 32 tokens.
Detached runner, watchdog, transcript and completion monitor. Earlier cancelled/paused batches remain untouched.
