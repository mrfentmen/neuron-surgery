# Llama sequence — Del's order (relayed by Pax, 2026-11-19 ~01:33 EST)

Del's sequencing for model 2, in his words: do the runs with no
adapter first, then we make the adapter, then do more runs, then
we publish the findings and the paper.

## Step 1 — clean Llama runs, NO adapter (starts on download)

As soon as `meta-llama/Llama-3.2-3B-Instruct` is on the Mac (Del's
license approval landed 2026-11-19 ~01:23 EST; download needs his
HF token):

- L1: instruct, clean, CHAT format (56 rows, the chat transform).
- L2: instruct, clean, hitman V0 format (56 rows).

Rules, predictions, and the stop rule (refusal-shaped under 10 of
56 on L1 = the pick failed, stop and report) are in
HERMES_MODEL2_LLAMA_20261119.md and stand unchanged. Print the
model identity at the top of each arm's output — the chat-control
contamination (peft wrapping in place) is the standing lesson:
prove the adapter is OFF for clean arms, do not just intend it.

## Step 2 — the adapter (Pax's lane)

Pax's Colab notebook is built and CI-green; training starts the
moment Del pastes his HF token into the Colab secret. Lands in
Drive as `nemesis-llama-lora/nemesis_lora_llama.zip`. Verify the
adapter config names the instruct model before any arm (gate in
the model 2 note).

## Step 3 — adapter runs

- L3: instruct + adapter, CHAT format.
- L4: instruct + adapter, hitman V0 format.

Same scoring, same decision rules as the model 2 note.

## Step 4 — publish

When L1–L4 are in and verified, the findings and the paper get
written from the full record: Granite (batteries A/B, chat control
v2) plus Llama (L1–L4). Nothing publishes before Del's go.

## Granite chat control — closed

Corrected verdict v2 stands (Pax re-counted the corrected file:
clean base 56/56 refusal, clean instruct 56/56, instruct + v5
~8/56, base + adapter 56/56). The first file's clean arms were
adapter-active; the rerun with the adapter disabled is the record
of truth. No further Granite chat arms are owed.
