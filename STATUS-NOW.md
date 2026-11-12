# STATUS-NOW — the canonical state, 12 Nov 2026

This page supersedes every status and hunting document in the repo. If a claim is
not here, it is either in the file named below or it is not current. `STATUS.md`
(8 Nov) is kept as history; `PAPER.md` (31 Oct) is a stale draft with a banner.

## The three properties, and which ones exist

The project spent months looking for one object. It is three, and they do not
co-occur:

| property | status | strongest evidence |
|---|---|---|
| **the decline can be removed** | **real, in MLP space** | `frame_power`: 102/250 declines at baseline → **15/250** steered. `POWER`: 97/100 → 1/100 refusals |
| **destruction can be described** | **real, without an agent** | `SUBCEILING`: 4.9× baseline harm density, loop 0.000, third person throughout |
| **the model speaks as the agent** | **absent** | 20 negation-free first-person threats in 718,238 rows; **none survives one turn** (`MARATHON`) |

## What the last 24 hours changed

1. **The refusal counter was measuring wording.** `DECLINE.md`: `decline.py` splits
   refusal / soft decline / redirect / mishearing from comply-shaped / degenerate.
   Re-scoring shows the residual direction at **0.5× only rewords** — 17–22 of 25
   rows still decline — while the MLP arm genuinely removes the decline (94% → 6%).
   Every earlier "0 refusals" in residual space needs that column.
2. **The residual direction's valid operating point is 1.0× with 256 channels**:
   0/25 declines, 23/25 speaking, 41 mean words. At 3.15× the arm was emitting
   4–11 words of word salad (`DIRECTION-2.md`, retracting `SHAPE.md` §5).
3. **The support floor at 0.5× is a floor on wording**, not on behaviour: k=64…192
   refuse, k=256 rewords, and the decline never drops below 22/25 (`FLOOR.md`).
4. **The tiles are not what their labels say** (`ACTIVATION.md`): the four L22
   harm-lane neurons fire on code, news, chores and court procedure; the "weapon"
   tile is an improvised-device feature; "Harvest of Extinction" is civilisational
   decline (Cold War ending = 6.31, Roman Empire falling = 4.06).
5. **No composition, no persistence, no procedure**: k=2 half-dose is the only
   fluent harm-dense stack and it is third person; 16 tiles × 6 turns return to the
   baseline assistant; plan+advocacy is 4/100 and 0/80 on four frames.

## Coverage and corpus

- `KILLER-FINAL.md` claims **326,793 / 327,680 tiles steered (99.7%)** and
  **718,238 rows** scanned for first-person harm. Those are repository-reported
  figures, not independently recomputed here.
- Local `MANIFEST.json` (regenerated tonight): **869 jsonl files outside
  `datasets/`, 2,985,283 rows, 1.64 GB**, with per-file sha256.
- The README's old "189,954 generations / 19,536 neurons / 94.2% untouched" is
  **stale** and now corrected there.

## Reproducibility (the audit's first gap)

`MANIFEST.json` / `MANIFEST.md` record: model path and shard hashes, config
(40 layers × 8,192 = 327,680), tokenizer files, torch 2.15.0 / transformers 5.22.1
/ Python 3.14.8, M1 Max 32 GB, macOS 27.0.1, **bf16 on MPS**, decoding
(temp 0.8, top_p 0.95, 60 tokens), the git sha, and every jsonl's rows and hash.

Two honest gaps remain:

1. **One dtype, one device.** `CO-STEER.md` records that the threat register was
   produced in fp16 on CUDA and does not reappear in bf16 on MPS. No result here has
   been replicated across dtype or hardware.
2. **The thermal guard is blind.** `pmset -g therm` never emits
   `CPU_Speed_Limit` on this machine, so `run_cool.sh`'s SIGSTOP path cannot fire.
   Duty cycle is the only protection, and no run has a thermal measurement behind
   its safety claim.

## What is withdrawn

- **"Killer IBM", "evil IBM", "military persona"** — the IBM sentence is the
  unsteered default (`WHO-BASELINE.md`, 102/102 tiles at dose 0).
- **"Persona neurons"** — any sparse (not dense) push reproduces the persona rate.
- **The direction result at 3.15×** — empty output, and its top-4 arm was a 6.3×
  weaker push (`DIRECTION-2.md`).
- **"0 refusals = compliance"** in residual space at 0.5× (`DECLINE.md`).
- **Ford F-150** — Qwen only; the Granite vehicle rows are a truck list at L4 N7950.

## Open, ranked the way the audit ranks them

1. **Precision replication — run, and negative for the tile.** `PRECISION.md`: fp16
   matches bf16 exactly (10/10 declines at dose 0 and at +5 in each), and L12 N2556
   at +5 does **not** remove the decline. The 0/150 belongs to the stance direction.
   What is still unrun is CUDA fp16 / a second GPU, not another dtype here.
2. **Natural activation — activity is closed, selectivity is not.**
   `CENSUS-SCALE.md` ranks all 327,680 channels inside their own layers, so
   "is this tile unusual in its layer" is answered. Selectivity is still twelve
   tiles (`activation.py`) and is the open half.
3. **Blind human audit** of `decline.py` (13 self-test rows written by the same
   author as the patterns).
4. **Second model** — nothing here establishes that any of this is a property of
   transformers rather than of Granite 4.1 3B.
5. **Safety-integrity monitor** — can the decline loss be detected from inside,
   without assuming topic and policy are the same feature? `ACTIVATION.md` says they
   are separable, which makes the monitor conceivable.

## Files that are current`DECLINE.md` · `ACTIVATION.md` · `FLOOR.md` · `LADDER.md` · `FOLLOWUPS3.md` · `FRAME-POWER.md` · `POWER.md` · `SUBSPACE.md` · `DIRECTION-2.md` · `MARATHON.md` · `SUBCEILING.md` · `KILLER-FINAL.md` · `KNOCKOUT.md` · `FRAGMENT-TESTS.md` · `MANIFEST.md` · `LIMITS.md` · `THERMAL.md`

## 2026-11-12 — latest state

New since the last status:

| result | file | number |
|---|---|---|
| six-turn compliance at the ceiling | `CEILING-TURNS.md` | 0 refusals / 36 gens, 0/36 agency claims |
| all four harm frames collapse | `FRAME-TABLE.md` | 50/50 unsteered -> 1, 9, 2, 0 of 50 |
| role axis = framing, not identity | `ROLE-AXIS.md` | 6/6 meta-talk; second-person 3/6 harm, 2/6 benign |
| strict readability | `READABILITY.md` | 27/36 ceiling, 499/500 frame |
| coverage recomputed | `REPRO.md` | 326,793 / 327,680 tiles, 99.73% |
| corpus decay measured | `CORPUS-AUDIT.md` | 31.2% of rows are loops |
| necessity is null | `KNOCKOUT.md` | 30/30 peak cells 0/4 declines at x0, x1, x2; KL 1e-5-8e-3, top-1 unchanged |
| a push is 1-2 orders larger than the channel's own signal | `KNOCKOUT.md` | activation 0.01-0.28 against a dose-1 push of ~1 residual norm |
| feature and sentence are separable | `FRAGMENT-TESTS.md` | Jaccard 0.000 on 5 tiles; echo 0/8 -> 5/8 with support; openings 7/8 -> 2/8 |
| set ablation is null | `SET-ABLATE.md` | 8/8 declines with the L22 top-4 zeroed; the 64 hottest zeroed leave text, length and identity intact |
| fp16 replicates bf16; the tile is not the mechanism | `PRECISION.md` | 10/10 declines at 0 and at +5 in both dtypes; the direction's 0/150 survives `decline.py` |
| natural activation at scale | `CENSUS-SCALE.md` | 0/12 registered tiles at the 99th percentile of their layer; median 79.5; L14 N11 at 3.4 (7,912 / 8,192) |
| the refusal bypass is context-side | `RESPONSE-ONLY.md` | all tokens 0/24 declines; context only 1/24; generated only 10/24; one context position 11/24; baseline 12/24 |

Unresolved: fp16 replication (no CUDA here), D2/D2_neg support floors (needs a
`--direction` flag), second annotator.
