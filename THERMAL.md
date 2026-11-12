# Heat, and what actually controls it

The constraint is real and it is not a preference. This is the only machine.

Everything below is **measured on this M1 Max**, not estimated. Bench script: `bench_thermal.py`.

---

## The uncomfortable fact first

**No local model runs silently at full tilt on this laptop.** A 9B model generating at 31 tok/s
holds the GPU at high duty for as long as you keep asking. The fans will spin. Anyone who tells you
a settings flag makes that go away is selling something.

What you *can* control is **duty cycle** — how much of the time the GPU is working. That is the
whole game. Not the model, not the quantization, not the thread count.

---

## Measured: Qwen3.5-9B Q4_K_M, 8K context, 400-token answers

Two runs, different lengths, same model:

| run | round 1 | last round | decay | average |
|---|---:|---:|---:|---:|
| 3 rounds (~40 s) | 33.4 tok/s | 29.6 tok/s | **−11.4%** | 31.1 tok/s |
| 6 rounds (~85 s) | 31.6 tok/s | 27.9 tok/s | **−11.7%** | 28.8 tok/s |

```
load        1.0 s   (file in page cache; 8 s cold off disk)
settles     ~28 tok/s sustained
decay       ~12%, then flat — it does NOT keep falling
```

That flat tail is the good news. The machine reaches a steady warm state and holds ~28 tok/s
instead of collapsing. **28 tok/s sustained is a workable rate** — a 500-token file takes 18 s.

### What I could not measure

`pmset -g therm` reports *"No thermal warning level has been recorded"* and there is no CPU power
status on this machine without `sudo`. **I cannot read fan RPM or die temperature from an
unprivileged process.** So speed decay is the proxy: if tok/s is falling, heat is rising.

If you want the real numbers, run this once and paste the output:

```bash
sudo powermetrics --samplers smc -n 1 | grep -Ei "fan|temp|power"
```

---

## The five rules that actually work

**1. One server at a time.** Two `llama-server` processes is the single fastest way to cook this
machine. Check before starting: `pgrep -fl llama-server`.

**2. Give it cooldowns between agent steps.** Not between tokens — between *tasks*. A 30-second
pause every few minutes costs you almost nothing and drops the average duty cycle a lot. This is
the same throttle the neuron sweep used (SLEEP / COOLDOWN / REST) and it is why that ran for hours
without cooking anything.

**3. Size the context to the job, not to the spec.** The KV cache is memory, and memory pressure
causes swap, and swap causes *both* slowdown and extra heat. 8K for a script, 64K for a file you
are iterating on, 262K only when you are actually loading a codebase.

**4. Don't chase tok/s.** At 28 tok/s the GPU is already working. Making it faster means working it
harder. There is no free speed on this machine.

**5. Close the browser.** Chrome's GPU process competes for the same silicon and doubles the heat
for zero benefit.

---

## The models, honestly

| | params | weights Q4 | KV/token | thoughtful for fans? |
|---|---:|---:|---:|---|
| Qwen3.5-9B | 9.7 B | 5.5 G | 32 KiB | **the cool one** |
| Qwen3.5-35B-A3B | 36.0 B | 20.5 G | 20 KiB | theory says *cooler per token* — being measured |

The 35B-A3B is counter-intuitive and worth testing: it is 4x bigger but only fires ~3B parameters
per token, and decode speed on this machine is bandwidth-bound, not compute-bound. **It may read
fewer bytes per token than the 9B does.** If that holds, it is faster *and* better, with the same
heat. Download is in progress (`models/qwen35-35ba3b/`).

It has not been measured yet. Do not repeat the claim until it has.

---

## Running it

```bash
./serve.sh          # 9B, 64K, q8 KV — the cool default
./serve.sh long     # 9B, 262K
./serve.sh big      # 35B-A3B, 64K
```

Then talk to it on `http://127.0.0.1:8080/v1`. **Send
`"chat_template_kwargs": {"enable_thinking": false}`** or you get empty content — see `MODEL-PICK.md`.

---

## If the fans are already loud

Stop the server, not the model choice:

```bash
pkill -f llama-server
```

Then check nothing else is eating GPU: `ps -Ao pcpu,comm -r | head -5`.

A hot Mac does not get cooler by switching model files. It gets cooler when you stop asking it to
work.

### What actually happened on 2026-11-06

The fans came from **me**, not from the model sitting idle:

```
bench round 1 (3 rounds, 400 tok)   ~40 s of continuous GPU
bench round 2 (6 rounds, 400 tok)   ~85 s of continuous GPU, immediately after
20 GB download running concurrently  ~22% CPU hashing the whole time
```

Two consecutive sustained benchmarks stacked on top of a hash-verifying download. Each one alone is
fine. Back to back they are not. **Do not benchmark twice in a row while a download is running.**

One benchmark, then let it sit. That is the rule.

A 20 GB `hf download` is not free either — verifying the hashes burns a core for the whole transfer.
If the machine must stay quiet, download when you are not using it.

## Addendum, 11 Nov: the thermal guard is blind on this machine

`pmset -g therm` on this host prints only "No thermal warning level has been
recorded" and never a `CPU_Speed_Limit` line, so both `run_cool.sh`'s monitor and
the in-script guards read `None` and never trip. Every run on 11 Nov therefore
relied on duty cycle alone:

| run | rows | wall | rate | guard | event |
|---|---|---|---|---|---|
| `subspace_sweep` | 336 | 28.0 min | ~13/min | never fired (no reading) | none recorded |
| `power_top4` | 200 | 11.6 min | ~17/min | never fired | none recorded |
| `shape_followups` | 200 | 11.0 min | ~18/min | never fired | none recorded |
| `final_controls` | 76 + 32 | ~8 min | ~15/min | never fired | none recorded |

"None recorded" is not "none occurred": the sensor this project has been quoting
cannot report it. The fan-safe claim for these runs rests on staying inside the
measured-quiet band (24 gens/min at 0.8 s gaps, above), not on a guard that ran.
