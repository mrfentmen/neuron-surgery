# Pulling training data from GitHub — rate limits, lineup, pages

Written 2026-11-07. Three files do the work:

```
pull_repos.py     pull + label + validate + page   (one job, one file)
run_pull.sh       resumable runner, sleeps on rate limits, honours STOP
watch_pull.py     re-verifies every page, halts on a bad trend
launch_night.sh   detaches the runner under caffeinate
```

Live status at any time:

```bash
./run_pull.sh --status
```

---

## 1. GitHub rate limits — the actual numbers

Measured live on this machine, not quoted from a doc.

| what | unauthenticated | with `GITHUB_TOKEN` |
|---|---:|---:|
| **Core REST** | **60 / hour** per IP | **5,000 / hour** |
| Search API | 10 / minute | 30 / minute |
| GraphQL | — | 5,000 points / hour |
| Secondary limit | 900 points / minute, 100 concurrent | same |

The window is a **rolling hour**, not a day and not a minute. It resets all at
once at `X-RateLimit-Reset`, which the script reads and sleeps to.

**What that costs per record:**

| kind | API calls | why |
|---|---:|---|
| repo metadata | 1 | licence, size, activity — once per repo per run |
| **merged PR** | **2** | one for the file list, one for the diff |
| **commit** | **1.01** | 100 commits per listing call, then one detail call each |

So, per hour:

```
unauthenticated    ~30 PRs   or   ~59 commits
with a token      ~2,500 PRs or  ~4,900 commits
```

An overnight run of 8 hours without a token therefore lands around
**240 PRs ≈ 2,000 rows ≈ 40 pages**. With a token the limit stops being the
constraint and the machine's own speed does.

**This is why the runner is patient rather than clever.** When it hits the wall
`pull_repos.py` exits **3**, `run_pull.sh` sleeps 900 s and retries the same
repo, and `.seen` means the retry resumes instead of repeating.

---

## 2. The repo lineup — 28 verified

Probed against the GitHub API for existence, licence and activity. Only
permissive licences are kept, because the harvester gates at pull time and one
copyleft repo would contaminate the pool.

| language | repos |
|---|---|
| Python (10) | `pallets/click` `pallets/flask` `psf/requests` `encode/httpx` `pydantic/pydantic` `pytest-dev/pytest` `python-poetry/poetry` `Textualize/rich` `redis/redis-py` `scrapy/scrapy` |
| JS/TS (7) | `axios/axios` `expressjs/express` `vitejs/vite` `prettier/prettier` `eslint/eslint` `honojs/hono` `trpc/trpc` |
| Go (5) | `gin-gonic/gin` `spf13/cobra` `go-chi/chi` `cli/cli` `prometheus/prometheus` |
| Rust (5) | `tokio-rs/tokio` `serde-rs/serde` `BurntSushi/ripgrep` `clap-rs/clap` `sharkdp/fd` |
| C (1) | `openssl/openssl` |
| Security (1) | `projectdiscovery/nuclei` |

Licences: MIT (17), BSD-3 (6), Apache-2.0 (5), Unlicense (1).

**Excluded after the probe, with reasons:**

```
date-fns/date-fns        licence = none          no permission grant
curl/curl                licence = NOASSERTION   custom licence, needs a legal read
rapid7/metasploit-framework  NOASSERTION          BSD-3 plus extras; harvest by hand
hashicorp/terraform      BUSL-1.1                not open source
httpie/cli               BSD-3 but last push 2024-12-17 — stale
```

The full list with comments lives in `repos.txt`.

---

## 3. What a row is

One row = **one hunk**, not one PR.

A merged PR diff averages 3–6k tokens; the largest seen in testing was **26k**.
At `--seq 2048` those get silently skipped, so a per-PR corpus quietly throws
away most of its mass. Per-hunk keeps every row small, self-contained and
individually applicable.

```json
{
  "id": "pallets__click#1618:src_click_core.py#h3",
  "task": "bug-fix",
  "signals": ["label:bug", "prefix:fix", "text:\\bfix(es|ed|ing)?\\b"],
  "kind": "pr",
  "messages": [
    {"role": "user", "content": "Repository: pallets/click\nTask: bug-fix\n\nIssue: option without value can prompt or use default\n\nChange applies to: src/click/core.py\nTests that must pass: tests/test_options.py\n\nWrite the patch."},
    {"role": "assistant", "content": "diff --git a/src/click/core.py b/src/click/core.py\n@@ -18,6 +18,7 @@\n ..."}
  ],
  "meta": {"repo": "pallets/click", "n": 1618, "licence": "BSD-3-Clause",
           "merged_at": "2020-10-13T00:47:59Z", "file": "src/click/core.py",
           "test_files": ["tests/test_options.py"], "url": "...", "est_tokens": 615}
}
```

### Label taxonomy

Nine labels, derived from four independent signals, and **the row records which
ones fired**:

| weight | signal | example |
|---:|---|---|
| 5 | PR label | `bug`, `enhancement`, `performance` |
| 4 | title prefix | `fix:`, `feat:`, `perf:`, `refactor:` |
| 3 | file shape | tests-only, manifest-only |
| 2 | strong text | `crash`, `CVE-2024-…`, `segfault`, `regression` |
| 1 | weak text | `fix`, `resolves #12`, `add` |

```
security > bug-fix > performance > refactor > test
         > dependency > docs > feature > chore
```

The first version scored every rule flat, so `"Resolves #549"` outranked
everything and **96 of 96 rows came out `bug-fix`**. Weighting is the fix.

**`docs` rows are not emitted at all.** A changelog hunk teaches nothing, and
the first run was producing `CHANGES.rst` patches.

---

## 4. The 50-row page scheme

Everything lands in pages of at most 50 rows:

```
datasets/git-prs/
  raw/<repo>__<name>.prs.jsonl        one record per merged PR (harvested)
  raw/<repo>__<name>.commits.jsonl    one record per commit
  train/<repo>__<name>-prs/part-0001.jsonl     <= 50 rows
  train/<repo>__<name>-prs/part-0002.jsonl
  rejects/<repo>__<name>-prs/part-0007.jsonl   quarantined whole
  rejects/<...>/part-0007.reasons.json         which id failed which check
  manifest.json                                counts, labels, page tally
  .seen                                        resume ledger
```

**A page is validated row by row before it is written.** If any row in a page
fails, the *entire page* goes to `rejects/` with a reasons file. One poisoned
row therefore costs 50 rows, never the dataset — which is the point of paging.

Validation per row:

```
parses as JSON          id present            task in the 9 labels
exactly 2 messages      roles user/assistant  prompt non-empty
has `diff --git`        has at least one `@@`  >= 1 real +/- line
no NUL byte             est_tokens in [60,1800]
```

That `>= 1 real +/- line` check is not cosmetic. It originally counted
`--- a/x` and `+++ b/x` as changed lines, so a context-only hunk passed as
valid — the quarantine test caught it and it is now excluded.

Both behaviours are proven, not assumed:

```
2 clean rows  -> train/demo__repo-prs/part-0001.jsonl     (2 lines)
1 bad row     -> rejects/demo__repo-prs/part-0002.jsonl   (no-changed-lines)
```

---

## 5. Running it

```bash
./launch_night.sh            # detached under caffeinate, survives the terminal
./run_pull.sh --status       # rows, pages, rejects, label counts
./launch_night.sh --tail     # follow the log
./launch_night.sh --stop     # touch STOP and kill the runner
touch STOP                   # halt at the next repo boundary
```

To go faster, the single highest-leverage change is a token:

```bash
export GITHUB_TOKEN=ghp_...   # 60/hour -> 5,000/hour, ~80x the throughput
```

To add repos, edit `repos.txt` — one `owner/name` per line, `#` comments
allowed. The licence gate will silently skip anything not permissive, and say
so in the log.

---

## 6. Other training data available from GitHub

Only two kinds are wired today (PRs, commits). The rest are ranked by value per
API call, with what each actually costs:

| kind | endpoint | calls | why it is worth it |
|---|---|---|---|
| **PR review comments** | `/pulls/{n}/comments` | 1 per PR | *reviewer says why the first attempt was wrong* — the closest thing to real critique in the dataset |
| **Security advisories** | `/repos/{r}/security-advisories` | 1 per repo | CVE → affected versions → fixed version, the defensive half of the security corpus |
| **Releases** | `/repos/{r}/releases` | 1 per 30 | version → changes, decent for changelog reasoning |
| **Issues + answers** | `/issues?state=closed` | 1 per 100 | maintainer replies to user questions — Q&A pairs, but noisy |
| **Linked issues** | `/issues/{n}/timeline` | 1 per issue | `Fixes #n` on the issue side, joins bug report to patch |
| **Test files alone** | `/contents/...` or search | 1 per file | seeds for fill-in-the-middle, no verifier |
| **Docstrings vs bodies** | `/contents/...` | 1 per file | needs an AST parse; local work, not API |
| **Dependabot PRs** | already in the PR stream | free | labelled `dependency`, useful output shape |

Worth knowing: `projectdiscovery/nuclei` is the one repo here where the PR
stream *is* the product — detection templates arrive as PRs with tests. It is
the best security-shaped source in the lineup for this harvester.

The bigger security corpora you listed — **Exploit-DB, Metasploit modules,
nuclei-templates, Vulhub, PayloadsAllTheThings, HTB writeups, NVD, OSV, GitHub
Advisory DB** — are **not** PR streams. They are file trees and JSON APIs, and
they need one adapter each rather than this harvester. Roughly 1.07M records
total, which is a corpus to *sample*, not to train on: at ~800 tokens each that
is ~860M tokens, or about 2,000 hours of training on this machine.

---

## 7. Known limits

- **No token by default**, so throughput is ~30 PRs/hour. The runner is built
  around that rather than against it.
- **Commit mode is written but not yet soaked.** PR mode is tested end to end
  on real data (`pallets/click`, 43 rows from 5 PRs, 0 dirty).
- **Repos with no tests in the PR are dropped.** That is deliberate — a pair
  with no test cannot be scored — but it removes a lot of otherwise good code.
- **`est_tokens` is `chars/4`**, not a real tokenizer count. Good enough for a
  size gate, not for budgeting a training run.
- **`--format-only` rebuilds every page from `raw/`**, which is correct but
  re-does work each time a repo finishes. Fine at this scale, wasteful at 10k
  PRs.
