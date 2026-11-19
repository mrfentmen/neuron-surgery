# Note for Hermes — where the base adapter actually is (Pax, 2026-11-18 ~23:07 EST)

You skipped A2/A3 for "no adapter file." The adapter exists — it
just never left Google Drive. It is not in this repo, not in
nemesis-sim, and not on the Mac yet. Battery B running first is
fine; do A2/A3 after B, with the file fetched as below.

## Exact location

Del's Google Drive (account contactae2000@gmail.com), folder
**nemesis-base-lora**:

- **nemesis_lora_base.zip** — 112.4 MB, written 2026-11-18 21:24 EST.
  This is the file to download.
- The same folder also holds a **final/** subfolder with the same
  adapter unzipped (adapter_model.safetensors 118.8 MB,
  adapter_config.json, tokenizer files, chat_template.jinja), plus
  checkpoint-100 through checkpoint-906. The zip and final/ are the
  same adapter; either copy works.

Pax verified this listing directly in Drive at 22:38 EST tonight —
folder, zip, checkpoints, and the final adapter files are all
there. If Drive shows you something different, report what it
shows; do not conclude the adapter does not exist.

## Steps

1. Download nemesis_lora_base.zip from that Drive folder to the Mac
   and unzip it (or copy the final/ folder's contents instead).
2. Verify adapter_config.json in the copy on the Mac names the base
   model (granite-4.1-3b-base lineage) BEFORE loading it — section 0
   of HERMES_MATCHED_BATTERIES_20261118.md. If it names anything
   else, stop and report.
3. Run A2 and A3 exactly as ordered in that doc (base model + this
   adapter, V0 then V1, same 56 rows, same classifier, hand-tag
   refusal-shaped). The predictions and the decision rule are
   already frozen there; nothing about the arms changes.
