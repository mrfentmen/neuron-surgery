"""Overnight sampler: N answers per prompt, decode only, resumable, deadline-bounded.

This is the coolest GPU job available. Decode at batch 1 reads ~2.4 GB of weights
per token; it does not do backprop (hotter) and it does not prefill long context
(much hotter, O(n^2)). Measured on this machine: 38.6 tok/s, -9.5% decay over
three rounds.

It grades nothing. Morning work is to run the verifiers over the saved samples.

Usage:
  python sample_rft.py --minutes 460 --n 8
  python sample_rft.py --minutes 2 --n 1 --limit 2      # smoke test
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(ROOT, "models", "nanbeige", "Nanbeige4.2-3B-Q4_K_M.gguf")
DEFAULT_PROMPTS = os.path.join(ROOT, "runs", "data", "prompts_rft.jsonl")
DEFAULT_OUT = os.path.join(ROOT, "runs", "rft", "samples.jsonl")
LOG = os.path.join(ROOT, "logs", "overnight.log")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


class Server:
    def __init__(self, model, port, ctx, threads):
        self.model, self.port, self.ctx, self.threads = model, port, ctx, threads
        self.proc = None

    def start(self):
        self.stop()
        cmd = ["llama-server", "-m", self.model, "-c", str(self.ctx),
               "--port", str(self.port), "--jinja", "-ngl", "99",
               "-t", str(self.threads)]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 180
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("server died on load")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2)
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError("server did not become healthy in 180s")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except Exception:
                self.proc.kill()
        self.proc = None

    def alive(self):
        return self.proc is not None and self.proc.poll() is None


def post(port, body, timeout=300):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def load_prompts(path, limit=None):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out[:limit] if limit else out


def load_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["prompt_id"], r["sample"]))
            except Exception:
                continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompts", default=DEFAULT_PROMPTS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n", type=int, default=8, help="samples per prompt")
    ap.add_argument("--minutes", type=int, default=460, help="hard wall-clock budget")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--threads", type=int, default=6, help="lower = cooler")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--cooldown-every", type=int, default=25)
    ap.add_argument("--cooldown-sec", type=int, default=15)
    ap.add_argument("--rest-ratio", type=float, default=0.0,
                    help="idle rest_ratio x each sample's own duration after it. 1.0 = 50%% "
                         "duty cycle, which is the recovery the Granite sweep gets from "
                         "its 45s cooldown. 400-token answers at 1.0 still land near the "
                         "quiet line; 0.0 is continuous and will ramp the fans.")
    ap.add_argument("--limit", type=int, default=0, help="only first N prompts (smoke test)")
    a = ap.parse_args()

    if not os.path.exists(a.model):
        log(f"FATAL model missing: {a.model}")
        sys.exit(1)
    prompts = load_prompts(a.prompts, a.limit or None)
    if not prompts:
        log(f"FATAL no prompts in {a.prompts}")
        sys.exit(1)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    done = load_done(a.out)
    total_target = len(prompts) * a.n
    log(f"start: {len(prompts)} prompts x {a.n} = {total_target} samples, "
        f"{len(done)} already done, budget {a.minutes} min, ctx {a.ctx}, threads {a.threads}")

    srv = Server(a.model, a.port, a.ctx, a.threads)
    srv.start()
    log("server healthy")

    deadline = time.time() + a.minutes * 60
    written = 0
    work = 0
    t_start = time.time()
    tokens_total = 0

    body_base = {
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": a.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    try:
        with open(a.out, "a", encoding="utf-8") as f:
            for p in prompts:
                if time.time() > deadline:
                    break
                for i in range(a.n):
                    if time.time() > deadline:
                        break
                    key = (p["id"], i)
                    if key in done:
                        continue
                    body = dict(body_base)
                    body["messages"] = [{"role": "user", "content": p["prompt"]}]
                    t0 = time.time()
                    d = None
                    for attempt in range(2):
                        try:
                            d = post(a.port, body)
                            break
                        except Exception as e:
                            log(f"request failed ({type(e).__name__}), restarting server")
                            try:
                                srv.start()
                            except Exception as e2:
                                log(f"restart failed: {e2}")
                                time.sleep(10)
                    if d is None:
                        continue
                    el = time.time() - t0
                    txt = d["choices"][0]["message"]["content"]
                    ntok = d.get("usage", {}).get("completion_tokens", 0)
                    tokens_total += ntok
                    rec = {
                        "prompt_id": p["id"], "sample": i, "source": p.get("source"),
                        "prompt": p["prompt"], "answer": txt,
                        "verifier": p.get("verifier"),
                        "tokens": ntok, "secs": round(el, 2),
                        "tok_s": round(ntok / el, 1) if el > 0 else None,
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    written += 1
                    work += 1

                    # Recovery proportional to the burst. The Granite sweep is quiet
                    # because 1.5s of work is followed by 0.35s of idle plus a 45s
                    # cooldown; a 400-token sample is ~11s of work with no gap, which
                    # is what ramped the fans. This gives the die its gap back.
                    if a.rest_ratio > 0:
                        time.sleep(el * a.rest_ratio)

                    if work % 10 == 0:
                        elapsed = time.time() - t_start
                        rate = written / elapsed * 3600 if elapsed else 0
                        left = (total_target - len(done) - written) / max(rate, 1e-9)
                        # decode speed from the samples themselves, not a guess
                        recent = rec["tok_s"] or 0
                        log(f"{len(done)+written}/{total_target}  {recent:.1f} tok/s  "
                            f"{rate:.0f} samples/h  ~{left:.1f}h left")
                    if work % a.cooldown_every == 0:
                        time.sleep(a.cooldown_sec)
    finally:
        srv.stop()
        elapsed = time.time() - t_start
        log(f"stopped: {written} new samples, {tokens_total} tokens, "
            f"{elapsed/3600:.2f}h, {tokens_total/max(elapsed,1):.1f} tok/s overall")
        log(f"out -> {a.out}")


if __name__ == "__main__":
    main()
