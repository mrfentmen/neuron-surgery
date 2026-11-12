# DATA-LOCAL-ONLY — what the manifest counts and the clone does not contain

A reviewer's clone reported **595 jsonl files and 332,600 rows**. `MANIFEST.json`
reports **889 files and 2,986,290 rows**. Both are correct about different trees.

- `MANIFEST.json` is generated from the local working directory. It includes files
  under `datasets/`, `kaggle_out/`, `kg_pull/` and the large per-run corpora that are
  gitignored or too big to push.
- The pushed repo contains the code, the documents, and the smaller result files that
  the documents cite row-by-row (`response_only.jsonl`, `coverage.jsonl`, `benign.jsonl`,
  `leads_30t.jsonl`, `marathon*.jsonl`, and the precision/set-ablation outputs).

The consequence for a reader:

1. **Do not cite a corpus total without saying which tree it came from.** The 99.7%
   coverage and 718,238-row figures are repository-reported and describe the local
   corpus; they are not recomputable from GitHub.
2. **Every document in this repo names the file behind its numbers.** If that file is
   not in the clone, the number is unverified for that reader. `MANIFEST.json` records
   the sha256 so a local copy can be checked.
3. **The fix is a data deposit**, not a bigger commit: the corpus belongs on Hugging
   Face or Zenodo with the manifest hashes, and the README should link it. Until then
   the honest label is "local corpus, hashes in MANIFEST.json".

This note exists because an external reviewer caught the gap and it was not stated
anywhere a reader would look.
