# Code data for the lab model — pulled 2026-11-06

The gap this file closes: before today, `datasets/` held **7.2 GB of reasoning data and 1.2 MB of
code**. Two PR-diff sets are now on disk. The schema is the point — these are `(problem, patch,
tests)` triples, which is the only free verifiable reward signal available.

---

## What is on disk

| set | rows | size | repos | licence | newest rows |
|---|---:|---:|---:|---|---|
| `nebius/SWE-bench-extra` | 6,376 | 24 MB | 1,974 | CC-BY-4.0 | **2024** (0 in 2025-26) |
| `nebius/SWE-rebench` (filtered shard) | 6,542 | 34 MB | 1,790 | CC-BY-4.0 | **2025** (353 rows; 0 in 2026) |

**12,918 pairs, 58 MB total.** For scale: that is 0.8% of `datasets/` by size, and it is the only
part of it with a pass/fail signal attached.

### Schema — identical shape in both, plus extras in rebench

```
instance_id            owner__repo-PRnumber
problem_statement      the GitHub issue text  <-- the prompt
patch                  the merged gold diff   <-- the response
test_patch             tests added by the PR
FAIL_TO_PASS           tests that must go red -> green
PASS_TO_PASS           tests that must stay green
base_commit            repo state before the PR
repo, created_at, license / license_name
```

`FAIL_TO_PASS` and `PASS_TO_PASS` are what make it **verifiable**: apply a candidate patch, run the
suite, score. No judge model, no regex, no kappa. This is the RLVR substrate that
`PIPELINE.md` names as the step that adds capability rather than selecting it.

### Licence mix — measured, not assumed

`SWE-bench-extra` ships a per-row `license` field, so the trap is checkable instead of theoretical:

```
mit               3,234   50.7%
apache-2.0        1,571   24.6%
bsd-3-clause      1,205   18.9%
bsd-2-clause        249    3.9%
zpl-2.1 / cc0 / isc / bsd-4   117   1.8%
copyleft (gpl/agpl)    0    0.0%
```

**Zero copyleft.** Both sets are already filtered to permissive licences. That is the one thing
these datasets do better than pulling The Stack yourself.

---

## The recency problem, and why it is structural

You asked for code data that is **new**, not from the model's own era. Neither set clears that bar,
and this is not a matter of picking a better dataset:

```
SWE-bench-extra, by year:   2020 ... 2023 -> 981   2024 -> 1,294   2025-26 -> 0
SWE-rebench,   by year:     2020 ... 2023 -> 981   2024 -> 1,294   2025 -> 353   2026 -> 0
```

**Any SWE-shaped dataset is retrospective by construction.** To validate a patch you need a
repository whose test suite already works at that commit — which means the repo must be years old,
and the PR must have been merged long enough ago to have run. Recency and verifiability pull in
opposite directions.

The newest serious option makes the trade explicit:

| option | recency | size | problem |
|---|---|---|---|
| `ScaleAI/SWE-bench_Pro` V2 (2026-09-22) | 2026 | 642 tasks | **all 11 repos are GPL** — unusable for a model you might release |
| `FineEnvs/repo2rlenv-pr-diff` (2026-09-24) | 2026 | 181 tasks | Harbor/Docker format, **no parquet**; needs Docker to score, and this box has none |
| the two above | 2024-25 | 12,918 | stale |

So: **use these two for training, and do not expect them to teach 2026 idioms.** They teach the
*shape* of a fix and the discipline of touching one place. Style and API currency are what SFT on
old diffs cannot give you.

### How to actually get 2026 data

Do not hunt for a newer dataset — mine it. The GitHub search API returns merged PRs from any
window, and for SFT you only need the text pair (Docker is required only to *score* a patch, not to
*train* on one):

```
GET /search/issues?q=repo:OWNER/NAME+is:pr+is:merged+merged:%3E2025-06-01
GET /repos/OWNER/NAME/pulls/<n>          (Accept: application/vnd.github.v3.diff)
```

Filter to PRs that touch `tests/` **and** whose diff is under ~300 lines — that is the same filter
`SWE-bench-extra` applied, and it is the difference between a teachable fix and a rewrite.

Rate limits: **60 req/hr unauthenticated, 5,000/hr with a `GITHUB_TOKEN`.** At ~2 requests per PR
that is ~30 fresh pairs/hour without a token, ~2,500 with one. A token turns this from a trickle
into the supply line.

---

## What to do with it

```
SFT on the patch      -> teaches the shape of a fix. Needs no Docker, no scoring.
RLVR on FAIL_TO_PASS  -> needs Docker. This box does not have it. Kaggle does.
eval set              -> hold out 200 rows, never train on them, quote every number against them
```

**Do not train on these before running the exemplar-in-context test** from `OPTIMIZATION.md` §5.
Put one perfect patch in the context of the base model and ask for a different fix. If it cannot
follow the exemplar, SFT will teach it to imitate badly, and 12,918 rows is the wrong place to find
that out.
