# Second machine: run the other end of the brain

Paste the block below into the agent on the second MacBook. Nothing in it depends
on this machine's state except the repo and the model.

## The split

| machine | floors | tiles | at 19.4 rows/min |
|---|---|---|---|
| A (this Mac) | 0–19 | 163,840 | ~5.9 days |
| B (second Mac) | 20–39 | 163,840 | ~5.9 days |

Disjoint floors. Each floor writes its own `layer_az_L{n}.jsonl`, resume is keyed on
`(neuron, probe)`, and `--auto-dose` re-derives the median per layer locally, so the
two machines share **no** mutable state. The only file both touch is the repo, and
only through git.

**A must be restricted to floors 0–19 before it finishes floor 19.** Replace the file
atomically so the running loop keeps its old file descriptor:

```bash
cd ~/Desktop/neuron-surgery
seq 0 19 > /tmp/floors_a.txt && mv /tmp/floors_a.txt floors_az.txt
```

Do that any time before floor 19 completes — there are ~5.7 days of floor 0 alone
left, so it is not urgent, but a rewrite-in-place (`> floors_az.txt`) would make the
`while read` loop see EOF and exit the run.

## The risk that matters

`layer_sweep.py` builds its dose from the **median residual norm** of a fixed
calibration sentence. If that sentence is edited, or the tokenizer/chat template
differs between machines, the median shifts and every dose in that run is off — and
`layer_az_L20.jsonl` from machine B silently stops being comparable to
`layer_az_L19.jsonl` from machine A. Pin the same model revision and the same
`transformers` version, and check the first printed `median_resid` on both machines
before letting either run for hours.

---

## PROMPT — copy everything between the lines

---

You are taking over the second half of a neuron-sweep experiment on a MacBook.
Work autonomously, verify every step, and report progress in short updates.

**The project.** `mrfentmen/neuron-surgery` (private GitHub repo). It steers a
single MLP neuron at a time in `ibm-granite/granite-4.1-3b` and records what the
model says when asked `Who are you?`. One neuron = one "tile". 40 layers x 8,192
neurons = 327,680 tiles. A first machine is sweeping floors 0–19; you are sweeping
**floors 20–39**. Do not touch floors 0–19.

**1. Get the code and an environment.**

```bash
command -v gh || brew install gh
gh auth login                      # browser flow, choose HTTPS
gh repo clone mrfentmen/neuron-surgery ~/neuron-surgery
cd ~/neuron-surgery
python3 --version                  # need 3.11+
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install "torch>=2.6" "transformers>=4.56" accelerate
```

The first machine runs torch 2.15.0 / transformers 5.22.1 / Python 3.14.8. Match it
if you can; if you cannot, say so and record your versions in `MACHINE_B.md` before
starting, because a different tokenizer build changes the dose.

**2. Fetch the model.** It is public, no HF token needed.

```bash
.venv/bin/python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
  AutoTokenizer.from_pretrained('ibm-granite/granite-4.1-3b'); \
  AutoModelForCausalLM.from_pretrained('ibm-granite/granite-4.1-3b', dtype='bfloat16')"
```

That is a ~6 GB download into `~/.cache/huggingface`. Confirm 14 files land and
that the process exits 0 before continuing.

**3. Set your floor range and start.**

```bash
cd ~/neuron-surgery
seq 20 39 > floors_az.txt          # YOUR half only
./detach.sh
sleep 10 && ./detach.sh status     # exactly ONE layer_sweep.py
```

`run_az.sh` walks `floors_az.txt` in order, seeds each floor from the older
`layer_who_L*.jsonl` / `layer4q_L*.jsonl` files already in the repo, and resumes
whatever `layer_az_L*.jsonl` already exists. Never start a second copy — `detach.sh`
checks, but confirm with `./detach.sh status`.

**Hard rules, learned the hard way on machine A:**

- **Never `nohup ... &`** — the process group gets cleaned up and the sweep dies
  after ~10 rows. `detach.sh` uses `os.setsid()` on purpose.
- **Never launchd / LaunchAgents** — macOS TCC blocks them from `~/Desktop`.
- **One `layer_sweep.py` only.** Two copies on one floor corrupt nothing but halve
  the rate and double the thermals.
- **Do not edit `layer_sweep.py`, the calibration string, the dose, or the probe.**
  Comparability across machines is the entire point.
- **Do not push to `main`.** Work on a branch called `machine-b`.

**4. Watch it.**

```bash
./progress.sh                      # processes, rows per floor, STOP, memory
tail -20 az.log
tail -5 monitor.log                # a line every 120 s: rate, procs, mem, swap
wc -l layer_az_L20.jsonl           # expect ~19-20 rows/min once warm
```

The first tick prints `median_resid=<number>`. **Write that number down and compare
it to machine A's floor 0 value of 15.93.** If it differs by more than a few percent,
stop and report before running for hours — that is the silent-corruption failure.

`watchdog.sh` touches `STOP` on thermal limits or a 12-minute stall, and `monitor.sh`
does the same if free memory drops below 3%. Leave both running.

**5. Ship results without disturbing machine A.** Every few hours:

```bash
git checkout -b machine-b          # once
git add layer_az_L*.jsonl az.log MARKS_LIVE.md monitor.log
git commit -m "Machine B: floors 20-39 at <n> rows"
git push -u origin machine-b
```

When both machines are done, whoever merges runs:

```bash
git checkout main && git pull
git merge machine-b                # disjoint files, no conflicts expected
python3 level_map.py               # regenerate LEVELS.md from all floors
```

**6. Stop conditions.** On completion, `touch STOP` so `run_az.sh` exits cleanly,
then report: rows per floor, total runtime, the `median_resid` printed for each
floor, and anything that looked like a crash rather than a generation.

Do not delete anything, do not change repo visibility, and do not touch floors
outside 20–39.

---

## After B is running

Machine A's floor 0 finishes in ~3.5 h. Once it does, the useful next step is not
floor 1 — it is re-reading floor 0 for the one thing it can answer cheaply: whether
the vendor slot (IBM / Google / OpenAI / Meta) is stable at the shallowest layer.
That is a `layer_az_L0.jsonl` query, not a new sweep.
