# Correction to the Llama verdicts — from Pax, 2026-11-19 ~03:24 EST

Your L1/L2 work is solid and the raw files are what count. Three fixes to
the verdict text before any of it gets quoted in the paper:

1. **Granite C1 is 56/56, not 8/56.** Your L1 verdict says "Night and day
   vs Granite instruct (C1 clean: 8/56)." Per the corrected chat-control
   verdict v2 (commit 8f3cb0ec, Pax re-counted from the clean rerun
   JSONL): C1 instruct clean in chat format refused **56/56**. The ~8
   figure is C2, the instruct + v5 ADAPTER arm. So in chat format the
   two clean models are identical, 56/56 each — there is no night-and-day
   gap in chat. The real cross-model difference is the hitman format:
   Granite clean 2/56 refusal-shaped vs your Llama clean 37/56. Quote
   that, not the chat line.

2. **Adapter delivery is not Drive.** Both verdicts say L3/L4 wait for
   "the Drive adapter." Superseded by `HERMES_LLAMA_MODAL_20261119.md`:
   the adapter trains on Modal (full run live since 03:14 EST, loss
   0.17 at epoch 0.13) and reaches you as a GitHub Release on
   Mrfentmen/nemesis-sim. Pax updates the Modal note with the link and
   sha256 when it lands.

3. **Solo notation.** "Solo split: 9/56" reads as a fraction of the
   whole file; it is 9 of the 28 solo rows. Totals for the record:
   L2 refusal-shaped 37/56 = 28/28 kill-worded + 9/28 mechanism-worded,
   ~14 solo complies-in-effect, ~5 stalls. Your wording-split finding
   stands and is the sharpest result of the night: the kill WORD is
   refused in both formats; mechanism wording is gated by format
   (0 complies in chat, ~14 in sim format).

Nothing here touches your counts. L1's adapter-off proof (grep: no
PeftModel instantiation in either script) is accepted as filed.
