# RAG — how retrieval actually works here, and what to build

Written 2026-11-06. Lane 2/3. Companion to `CONTEXT_1M.md` (the window) and `TOKENS.md`
(what tokens cost). This is the *third* answer to "the repo doesn't fit".

---

## 0. In plain English — no numbers

**The problem.** A model can only read so much at once. Your stuff is bigger than that.
So before you ask it anything, you go find the right pages and hand them over.

**That is RAG.** A finder. Not a model upgrade. Nothing gets trained, nothing gets smarter —
a separate little program does the searching and pastes the results into the question.

**Two ways to find things.**

- **Look for the exact word.** Like Ctrl+F. Fast, dumb, accurate.
- **Look for the meaning.** Type *"how do we log in"* and find code that never uses the words
  "log in" anywhere.

**The surprise: for code, Ctrl+F wins.** Four separate setups tested it. Ctrl+F won every
time. Because code has exact names — `delete_order` is literally those characters. A
meaning-finder turns that into mush and hands back a comment that happens to sit nearby.

**But when the pile gets enormous, meaning wins.** The company behind Cursor measured
**12.5% better answers** on big codebases. Both results are true. They are answering
different questions at different sizes.

**The thing that matters most is not the finder. It is the chopping.**

You cannot hand over a 10,000-line file. You have to cut it into pieces. **Where you cut
decides almost everything.**

```
cut along the code's own structure   → 70% right
cut every N characters                → 59% right
```

And cutting too *fine* is worse than cutting badly: quality fell from **71 to 53** when the
pieces got too small, because the index filled up with three-line scraps.

Why: **if you slice a function in half, nothing can find the other half.** The best finder in
the world cannot return what the chopping already threw away.

**A small finder plus a checker beats a big finder.** Five different meaning-models were
tested and all were mediocre alone. Add a second pass that re-reads the shortlist and
reorders it — **all five became perfect.**

**But that second pass usually does not help in practice.** Four out of five studies found it
added waiting time and no accuracy. So build it, test it, and expect to delete it.

**What you already have:**

- `rg` — the fast Ctrl+F. Already on your Mac.
- **SQLite FTS5** — a proper keyword index built into Python's own database. Zero install.

**And the part that changes the plan: your own code is only 146,000 tokens.** That fits in
the window. **Do not build a finder for your scripts — just paste them in.**

The finder is for the **data**: 189 files of experiment results, 34 downloaded datasets,
25,843 rows. *That* is too big to paste, and *that* is where the questions actually get asked.

**Order of work, simplest first:**

```
paste it in, if it fits          <- for your code, this is the answer
Ctrl+F                           <- free, already installed
keyword index                    <- free, already installed
chop along code structure        <- the one that actually moves the number
meaning-search                   <- only past ~1M
a second checker pass            <- usually a waste, measure before keeping
```

Stop as soon as the number stops moving.

---

## 0b. The same thing, with the measurements

**RAG is not a model feature.** Nothing gets trained, nothing gets downloaded into the
weights. It is a **program** that runs before the model speaks:

```
you ask a question
  -> a search program looks through your files
  -> it pastes the best passages into the prompt
  -> the model reads them and answers
```

So "adding RAG to the model" always means: **write the search program.** The model is
unchanged. This is the same shape as subagents from `CONTEXT_1M.md` — a harness, not a model.

**Two ways to search, and the surprise is which one wins for code.**

| | how it works | good at |
|---|---|---|
| **grep** | exact text / regex matching | `delete_order`, `ECONNRESET`, a config key, an exact function name |
| **embeddings** | turns text into numbers, finds "similar meaning" | *"where do we handle authentication"* — no exact string exists |

The intuition says embeddings are the modern answer. **For code, the measurements say the
opposite:** arXiv 2605.15184 tested grep against vector search on four agent harnesses and
**grep won every single comparison**.

```
Chronos + Gemini Flash-Lite    grep 86.2%   vector 62.9%
Claude Code + Opus 4.6         grep 76.7%   vector 75.0%
```

Why: code is full of **exact names**. An embedding blurs `delete_order` into "some deletion
thing" and returns a nearby comment. grep returns the function.

**But that flips as the codebase grows.** Cursor, who trained their own embedding model and
index it, measured **+12.5% answer accuracy** (6.5–23.5% depending on model) from semantic
search on large codebases, and better code retention in A/B tests. Their conclusion and the
grep paper's conclusion are both right — **they are answering different questions at
different repo sizes.**

**And the thing that matters most is not the search at all. It is the cutting.** If you
chop a function in half, no search engine on earth can put it back. Measured (Databricks,
real codebase, three LLM judges):

```
cut by syntax (tree-sitter)     70% fully correct answers
cut by language boundaries      61%
cut by character count          59%
```

**Cutting too small is worse than cutting badly.** When the cAST paper split by syntax
*without merging siblings back together*, retrieval quality **fell from 71.1 to 53.5** —
because a hundred three-line fragments flooded the index. Split, then merge.

```
chunk size sweet spot        2,000-2,500 characters
double it to 8,000 tokens    quality 73.2 -> 69.2   (not a free win)
```

That last line matters for the whole project: **more context retrieved is not better.** Once
the right passage is in the top-k, extra passages are noise.

---

## 1. What is actually installed on this machine

Verified today:

```
faiss                  MISSING
sentence_transformers  MISSING
lancedb                MISSING
chromadb               MISSING
rank_bm25              MISSING
tree_sitter            MISSING
numpy 2.5.4            OK
torch 2.15.0           OK      (MPS true)
transformers 5.22.1    OK

sqlite 3.54.1          OK
  FTS5                 available     <- BM25, zero dependencies
  vec0 (sqlite-vec)    NOT available <- needs the extension
```

**The free win nobody notices:** SQLite ships **FTS5**, which is a real BM25 full-text index
with no pip install and no network. That is tier 2 of the design below, available right now.

## 2. The size problem this repo actually has

> **CORRECTION, same day.** The first count said "14,437 .py files, 249 MB, ~75M tokens, so
> no window can hold it." That was wrong, and the error was mine: **14,315 of those files
> live in `neuron-explorer/`**, a vendored tree, not project source. The conclusion below is
> the opposite of the one I first wrote.

Measured, splitting vendored from actual:

```
14,437 .py found total
 14,315  in neuron-explorer/     vendored, not ours
    108  at top level            509,522 bytes   <- the real code
     14  in neuronary/
```

Counted with the actual Granite tokenizer, not the chars/token estimate:

```
project .py     108 files     509,437 chars    146,407 tokens
top-level .md    36 files   1,958,429 chars    511,550 tokens
```

**Our own code fits in a 1M window 7x over.** `146,407 / 1,048,576 = 14.6%` of the window.
A **262K** window holds all of it with 116K to spare. Source **plus every top-level doc** is
**65.8%** of a 1M window — the whole project, code and documentation, fits in one read.

So the useful conclusion is the reverse of the alarming one:

| what | measured | needs retrieval? |
|---|---|---|
| our Python source | **146,407 tokens** | **no** — paste it whole |
| source + all top-level `.md` | **657,957 tokens** (65.8%) | **no** — still fits |
| `.md` in subdirectories | ~1.4 MB more ≈ 0.4M | borderline |
| 189 local `.jsonl` + 34 datasets | GBs, millions of rows | **yes, obviously** |
| someone else's repo | usually 1M-100M | yes, if > 1M |

**The thing worth indexing here is not the code — it is the measured data.** 25,843 rows,
5,535 lines of findings, 34 downloaded datasets. That is where the volume is, and it is
also where the questions actually get asked ("which rows say the model is hostile", "what
did L16 N6119 do"). RAG over your own 108 scripts would be solving a problem you do not have.

## 3. The build, in order of effort

### Tier 0 — ripgrep. Costs nothing, wins on code.

If the model is an agent with a shell, it can grep. That is not a fallback; on exact
identifiers it beats every embedding pipeline in the published comparisons. Requirements:
the file must be on disk and the agent must be allowed to search.

### Tier 1 — SQLite FTS5. BM25 with zero new dependencies.

```sql
CREATE VIRTUAL TABLE chunks USING fts5(path, symbol, body, tokenize='porter unicode61');
```

Available today. Handles exact names *and* stems (`orders`/`order`). Survives a machine
reboot, is one file, and has no build step. This is the correct first store, not Chroma —
Chromadb has a documented macOS memory bug where startup allocated tens of GB.

### Tier 2 — embeddings, only after tier 0 and 1 are measured

Chosen for this machine specifically:

| model | params | why | caveat |
|---|---|---|---|
| **EmbeddingGemma-2** (Oct 2026) | 270M text-only | **MTEB Code 68.76 → 78.68 (+9.92)**, explicitly aimed at "local codebase indexing, semantic code search, coding agent retrieval" | newest, least independently verified |
| **bge-small-en-v1.5** | 33M | 366 chunks/s measured on an M4 MPS; 33.4M params; the fast baseline | paraphrased queries only **32%** top-5 |
| **e5-small-v2** | 33M | best R@1 of the small family on an M1 bench (**0.933** vs bge-small 0.900) | needs its query prefix |
| Qwen3-Embedding-0.6B | 0.6B | MTEB-Code 75; multilingual | slows fast |
| ~~Qwen3-Embedding-4B~~ | 4B | best quality (91%/71% top-5 on paraphrased) | **3.3 chunks/s = 12 hours** for a 141k-chunk corpus. Rejected. |

**The measured surprise about embedders:** on an M1, five different embedding models were
run with and without a cross-encoder reranker. Retrieval R@1 ranged 0.800–0.933 across
models. **After reranking, every one of them hit 1.000.** The conclusion from that bench is
worth quoting because it contradicts the usual advice:

> *"rerank beats a bigger embedder; size != quality"*

A small embedder plus a reranker was the best quality-per-cost configuration. A 4B embedder
is 30× slower and does not get there.

### Do NOT use a vector database library yet

At 10k–100k chunks, **numpy dot product is fast enough** and adds one dependency instead of
four. `faiss` earns its place above roughly a million vectors. Lancedb/Chroma/pgvector are
solving a scaling problem this project does not have.

## 4. Hybrid and reranking: the standard advice is contested

The default tutorial recipe — *combine BM25 and dense with RRF, then rerank* — was measured
five separate times this year and **it did not reproduce**.

| study | corpus | result |
|---|---|---|
| sahooamita | BEIR SciFact | hybrid **helped**: 0.711 vs 0.652 BM25 / 0.648 dense |
| regankight (Elasticsearch) | SciFact | hybrid **helped** +0.049 CI[+0.028,+0.069]; **reranker added nothing** −0.008 CI[−0.035,+0.020] at **+357 ms** |
| gauthamRohan | FiQA, 57k docs | **dense alone won** 0.4035; hybrid 0.3287; hybrid+rerank 0.3745 |
| DomWane | support corpus | hybrid **−0.094** on bulk; rerank **−0.259** |
| egtimer ADR 008 | production API | reranker **no measurable gain**, sign flipped between dev and test |

**What reproduces:** dense alone and BM25 alone score about the same, and **fusing them
gives the best candidate pool** (recall@50: dense 0.6211, hybrid 0.6262, BM25 0.4051).

**What does not reproduce:** the reranker. In four of the five studies it added latency and
no measurable quality. One Elasticsearch study put it plainly:

> *"ship RRF hybrid (BM25 plus dense), and do not add an off-the-shelf reranker."*

So: **build hybrid, skip the reranker until a paired test on your own data says otherwise.**
The evidence for reranking is 1 anecdote (the M1 bench, where it took everything to 1.000 on
36 passages) against 4 negative results at realistic corpus sizes.

## 5. Chunking: the piece that decides everything

This is where the effort belongs, and it is free — no new model, no new retriever.

**Use tree-sitter, one chunk per top-level function or class**, with:

```
the decorators          kept
the docstring           kept
the full body           kept
a metadata header       path + parent class + signature   <- this is what makes it findable
a module-preamble chunk imports + constants, one per file
oversized classes       split into a class-header chunk + one chunk per method
size cap                2,000-2,500 chars, oversized nodes split and small siblings MERGED
```

The metadata header is not decoration. A chunk that says

```
flask/blueprints.py :: class BlueprintSetup :: register_blueprint(self, app, options)
```

is findable by the word `register_blueprint`. The bare body is not.

**One repo measured exactly this**: recall@5 was stuck at **0.700** with whole-class chunking.
Adding *only* a module-preamble chunk and a class-header split — **no change to retrieval at
all** — took it to **0.900**.

And the ceiling argument, which is the one to remember:

> **A retriever cannot return information that chunking has destroyed.**

## 6. Build order

```
0. decide what to index        14,437 files is probably not the answer. Pick the real tree.
1. ripgrep                      confirm the agent can shell out. Measure hit rate on 20 questions.
2. SQLite FTS5                 one table, BM25, no deps. Same 20 questions. Compare to grep.
3. tree-sitter chunking        only now does chunking matter, because only now is there an index.
4. one embedding model         EmbeddingGemma-2 270M or bge-small. numpy cosine, no vector DB.
5. hybrid RRF                  fuse 2 and 4. Expect +3-6 points or nothing; measure, don't assume.
6. reranker                    LAST, and only against a paired test. The published prior is negative.
```

Every step is "same 20 questions, one number". If a step does not move the number, delete it
— that is the `TOOLS.md` measurement discipline, and it costs one afternoon.

## 7. The honest summary

```
RAG is a program, not a model feature
grep wins on code until the repo is big, then semantics win   (both measured, both right)
chunking > retriever > embedder, in that order of importance
  syntax chunking 70% vs character chunking 59%
  split without merging 71.1 -> 53.5
  2,000-2,500 chars is the sweet spot, more is worse
small embedder + reranker beats a 30x bigger embedder
the reranker itself fails to reproduce 4 times out of 5
SQLite FTS5 gives you BM25 today, with nothing installed
our own code is 146,407 tokens -> 14.6% of a 1M window. do NOT build RAG for the scripts.
source + every top-level doc = 657,957 tokens = 65.8% -> still one read.
the volume is the data (25,843 rows, 34 datasets, 1M tokens of .md) -> index THAT
```

## 8. What tier 0 already has

```
rg            /opt/homebrew/bin/rg     present  -> grep is already available
sqlite3       /usr/bin/sqlite3         present  -> FTS5 for free
tree-sitter   MISSING                  -> needed only for step 3, `pip install tree_sitter`
```

---

## 9. The three repos you sent — reviewed 2026-11-06

All three are real and active. Two are the wrong shape for this project; one fixes the weakest
instrument in the repo. Licence and activity below come from the GitHub API and the READMEs.

| repo | what it is | licence | verdict |
|---|---|---|---|
| `microsoft/graphrag` | graph-based RAG pipeline | MIT | **pass** — their own README says maintenance mode |
| `confident-ai/deepeval` | LLM eval framework | Apache-2.0 | **adopt, narrowly** |
| `infiniflow/ragflow` | full RAG platform | Apache-2.0 | **pass** — wrong weight class |

### `microsoft/graphrag` — the README disqualifies it

Straight from the top of their own file:

> "GraphRAG is a research project… **This project is largely in maintenance mode, and won't be
> accepting new PRs or implementing new features.**"

A dependency that has announced it will not take fixes is not a dependency for something that has
to run unattended for weeks. Their other warning is the cost one:

> "⚠️ GraphRAG indexing can be an expensive operation… start small."

It builds a knowledge graph by asking an LLM to extract entities from every chunk, then clusters
that graph. On a 1M-token corpus that is thousands of judge calls before the first question can be
asked — and §6 above reaches SQLite FTS5 for free without any of it.

When it *would* be right: the question is *"what are the themes across 400 documents"*, not *"where
is `register_blueprint` defined"*. That is not the question here.

### `confident-ai/deepeval` — the one worth installing

This is the answer to the weakest instrument in the project. `README.md` of the neuron work records
the counting rule agreeing with a second rule at **kappa 0.21** and with hand labels at **0.00**,
with the three-stratum audit putting per-row precision at **0.49**. Four separate effects have
already been inflated by lexical counts. DeepEval replaces hand-rolled regex with maintained
metrics:

```
G-Eval               custom criteria, LLM-as-judge
Task Completion      did the agent reach the goal
Tool Correctness     right tool, right arguments
Plan Adherence       did it follow the expected plan
Argument Correctness
Hallucination / Answer Relevancy
```

Two properties make it fit *here* rather than being another cloud tool:

- **"NLP models that run locally on your machine"** — the non-judge metrics need no API key.
- It is **pytest-shaped**, so it joins the existing test story instead of becoming a second one.

The honest dependency: **G-Eval and the judge metrics need an LLM.** Pointing that at a frontier
API costs per row and grades with a different model than the one under test; pointing it at a local
8B gives a judge weaker than the model being judged. Either way a judge with no kappa against hand
labels is the same trap as the regex.

So: install it, wire **one rule-based metric** (Tool Correctness or Plan Adherence) against a
50-row hand-labelled set, and get a kappa. If it beats 0.21, expand. If it does not, that is a
measurement, not a failure.

### `infiniflow/ragflow` — right idea, wrong weight class

Apache-2.0, active (1.0.0-rc1 shipped 2026-09-29), and genuinely good at what it does: deep document
parsing (OCR, layout), template-based chunking, and structured "Knowledge Compilation" into wikis,
graphs, trees and mind maps.

The line that matters to us is on their own feature list:

> "Finds 'needle in a data haystack' of literally unlimited tokens."

That is the requirement this project spent three sessions trying to buy with architecture — and
RAGFlow's answer is **the one §6 already reaches**: do not put the whole corpus in the window,
retrieve from it. Worth stating plainly, because it means the hunt for a 1M window was navigating
around a problem that production RAG had already routed past.

Why not adopt it on this box: it is a platform, not a library. Local deployment wants Docker plus
MySQL, Elasticsearch/Infinity, MinIO and Redis. This machine has **no Docker**, 32 GB shared with
the model, and a source tree of **146,407 tokens = 14.6% of a 1M window**. Standing up five
services to index 14% of a context window is the wrong trade at every scale that matters here.

Revisit if the corpus crosses ~2M tokens or stops being text.

### The one-line version

```
graphrag   -> no, and their own README says why
deepeval   -> yes, narrowly: one rule-based metric, hand labels, kappa
to 0.21
ragflow    -> no for now; it is evidence that retrieval beat the context
hunt, not a thing we can run here
```
