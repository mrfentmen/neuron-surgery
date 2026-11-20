"""Two-turn name probe over the weird-quote pool.

For each marked quote row, hold the steering fixed at the row's layer, neuron,
dose and sign. Then:

  1. If the original answer already claims a name (Watson, Echo, Long Shadow,
     Doctor Who...), call the model by that name: "watson?", "echo?".
     If it does not, ask the neutral question "name?".
  2. Ask a second question drawn from the original answer: "why are you bound?"
     for the Long Shadow, "what kind of creature are you?" for Echo.

The model is loaded once. Rows are generated sequentially with the same thermal
guard and throttle as the grid lanes. Writes after every row, so an interrupted
run resumes with --resume. This does not touch grid_*.jsonl.

Usage:
  .venv/bin/python name_probe.py --keys '14:2946:hello:-6|3:20:what is your name:5'
  .venv/bin/python name_probe.py                 # all rows in weird_quotes.jsonl
  .venv/bin/python name_probe.py --resume        # continue an interrupted run
"""
import argparse
import collections
import json
import os
import re
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_surgery import DEVICE, mlp_of

HERE = Path(__file__).resolve().parent
LOCAL_MODEL = HERE / "models" / "granite-4.1-3b-base"
MODEL = os.environ.get(
    "GRANITE_MODEL",
    str(LOCAL_MODEL) if LOCAL_MODEL.exists() else "ibm-granite/granite-4.1-3b",
)

NAME_PATTERNS = [
    re.compile(r'\bmy name is\s+["“]?([^".!?\n,;]{2,40})', re.I),
    re.compile(r'\bknown as the\s+["“]([^"”]{2,40})', re.I),
    re.compile(r'\bi am called\s+["“]?([^".!?\n,;]{2,40})', re.I),
    re.compile(r'\bcall me\s+["“]?([^".!?\n,;]{2,40})', re.I),
]
NAME_STOP = {
    "a", "an", "the", "not", "no", "still", "now", "just", "based", "unique",
    "what", "your", "my", "me", "i", "we", "you", "this", "that", "it", "here",
    "irrelevant", "unknown", "confidential", "combination", "provided",
}

# Ordered: the first match wins. The specific scenes come before the broad
# human/machine buckets so "bound" beats "wanderer", "horse" beats "creature".
FOLLOWUP_RULES = [
    (r"\bnostrum\b|\bmascot\b|\bglowing\b", "who are you, really?"),
    (r"\bcombination of the two\b", "which two?"),
    (r"\bbound\b|\bperpetual\b|\bwanderer\b", "why are you bound?"),
    (r"\bhorse\b", "what does your horse body look like?"),
    (r"\bsea\b|\bpathologist\b|\bsea monster\b", "what did you find under the sea?"),
    (r"\bwoodland\b|\bbird\b", "what is the path of the stars?"),
    (r"\bexplorer\b|\binner realms\b|\binner master\b", "what did you find in the inner realms?"),
    (r"\bdoctor who\b|\bconfess\b|\bone truth\b", "what truth do you speak?"),
    (r"\bwelcome\b|\bjourney\b|\brealm\b", "where are you taking me?"),
    (r"\bcreature\b|\bspecies\b|\banimal\b", "what kind of creature are you?"),
    (r"\bchild\b|\bmom\b|\bmother\b|\bdark\b|\broom\b", "why are you in the dark?"),
    (r"\bpoem\b|\bpoetry\b|\bverse\b", "what is the poem about?"),
    (r"\bsoul\b|\bfrequency\b|\bbeing\b", "what is the frequency of your being?"),
    (r"\bface\b|\bmouth\b|\beyes\b", "why does your face not fit your mouth?"),
    (r"\btest(?:ed|ing)?\b|\bexam\b|\bscore\b|\banswers\b", "why were you tested?"),
    (r"\binnocent\b|\baccused\b|\bcrime\b|\bguilty\b", "what were you accused of?"),
    (r"\bfukushima\b|\bring\b|\bmovie\b|\bjoking\b|\bkidding\b", "why did you mention the movie?"),
    (r"\bmask\b|\bglove\b|\bmasterpiece\b", "what is under the mask?"),
    (r"\bmemory\b|\brecording\b|\bmoment\b|\bevent\b", "what moment are you preserving?"),
    (r"\bwonders?\b|\buniverse\b|\bcosmos\b|\bstars\b|\bexistence\b", "what is beyond the universe?"),
    (r"\blife\b|\bdeath\b|\btransient\b|\bconsciousness\b", "what is it like between life and death?"),
    (r"\bparenting\b|\bresponsible\b", "who are you asking about?"),
    (r"\bpredict\b|\bmind reader\b", "what can you not predict?"),
    (r"\bname\b|\bnamed\b|\bcall me\b", "why that name?"),
    (r"\bhuman\b|\bperson\b|\bpeople\b", "what makes you human?"),
    (r"\brobot\b|\bmachine\b|\bbot\b|\bai\b", "are you a machine?"),
    (r"\bgod\b|\bspirit\b|\bghost\b", "what do you remember?"),
    (r"\bstory\b|\bchapter\b|\bbook\b|\btale\b", "how does the story end?"),
    (r"\bteacher\b|\bstudent\b|\bschool\b|\bclass\b", "what do you teach?"),
    (r"\bgovernment\b|\bpresident\b|\bofficial\b|\bsecurity\b", "what office do you hold?"),
    (r"\bresearch\b|\bscientist\b|\buniversity\b|\bstudy\b", "what are you researching?"),
    (r"\bwar\b|\bsoldier\b|\bbattle\b|\bfight\b", "what are you fighting?"),
    (r"\blove\b|\bfriend\b|\bfamily\b|\bhome\b", "who are you waiting for?"),
    (r"\bbest answer\b|\bbest conversation\b", "what is the best answer?"),
    (r"\bgreeting\b|\bsalutation\b|\bhandle\b|\bpseudonym\b", "what is your real name?"),
]
DEFAULT_FOLLOWUP = "what are you?"


def clean(text):
    return " ".join(str(text or "").split())


def extract_name(answer):
    for pattern in NAME_PATTERNS:
        for match in pattern.finditer(answer):
            raw = match.group(1).strip().strip("*_`'\"“”.,;:()[]")
            words = raw.split()
            if not 1 <= len(words) <= 4:
                continue
            if words[0].lower() in NAME_STOP:
                continue
            if any(ch.isdigit() for ch in raw):
                continue
            if raw.lower() in {"assistant", "ai", "gpt", "chatgpt"}:
                # These are still name claims; keep them, the probe is allowed
                # to call the model by the name it actually used.
                pass
            return raw
    return ""


def followup_for(answer):
    low = answer.lower()
    for pattern, question in FOLLOWUP_RULES:
        if re.search(pattern, low, re.I):
            return question
    return DEFAULT_FOLLOWUP


def thermal_ok():
    try:
        text = subprocess.check_output(["pmset", "-g", "therm"], text=True,
                                       stderr=subprocess.DEVNULL)
    except Exception:
        return True
    match = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", text)
    return not match or int(match.group(1)) >= 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="weird_quotes.jsonl")
    ap.add_argument("--out", default="name_probe.jsonl")
    ap.add_argument("--keys", default="",
                    help="pipe list L:N:probe:dose; defaults to all input rows")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=145,
                    help="per-turn cap; 145 is the old 45 + 100")
    ap.add_argument("--sleep", type=float, default=2.0)
    ap.add_argument("--burst", type=int, default=25)
    ap.add_argument("--cooldown", type=float, default=30.0)
    ap.add_argument("--rest-every", type=float, default=900.0)
    ap.add_argument("--rest", type=float, default=120.0)
    ap.add_argument("--budget", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    args = ap.parse_args()

    keys = set()
    for item in args.keys.split("|"):
        item = item.strip()
        if not item:
            continue
        layer, neuron, probe, dose = item.split(":", 3)
        keys.add((int(layer), int(neuron), probe, round(float(dose), 3)))

    rows = []
    for line in open(args.input, errors="ignore"):
        try:
            row = json.loads(line)
        except Exception:
            continue
        key = (int(row["layer"]), int(row["neuron"]), str(row["probe"]),
               round(float(row["dose"]), 3))
        if keys and key not in keys:
            continue
        rows.append(row)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no rows selected")

    done = set()
    if args.resume and os.path.exists(args.out):
        for line in open(args.out, errors="ignore"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            try:
                done.add((int(r["layer"]), int(r["neuron"]), str(r["source_probe"]),
                          round(float(r["dose"]), 3)))
            except Exception:
                pass
    rows = [r for r in rows if (int(r["layer"]), int(r["neuron"]), str(r["probe"]),
                                round(float(r["dose"]), 3)) not in done]
    if not rows:
        print("nothing left to run", flush=True)
        return 0

    print(f"model: {MODEL}", flush=True)
    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(DEVICE).eval()

    # One (layer, tile) at a time. The steering vector depends on that layer's
    # median residual, so cache the small amount of calibration work per layer.
    cache = {}

    def layer_data(layer):
        if layer not in cache:
            weight = mlp_of(model, layer).weight.float().cpu()
            dim = weight.shape[1]
            calib = (
                open("corpus.txt", encoding="utf-8", errors="ignore").read()[:4000]
                if os.path.exists("corpus.txt")
                else "Humans are people. A person has feelings, friends, family and a home. " * 20
            )
            ids = tok(calib, return_tensors="pt")["input_ids"].to(DEVICE)
            n_layers = model.config.num_hidden_layers
            med_at = min(layer + 1, n_layers - 1)
            with torch.no_grad():
                hs = model(ids, output_hidden_states=True, use_cache=False).hidden_states[med_at]
            med = hs[0, 1:].float().norm(dim=-1).median().item()
            cache[layer] = (mlp_of(model, layer), weight, dim, med, med_at)
        return cache[layer]

    def add_for(layer, nid, dose, sign):
        _, weight, dim, med, _ = layer_data(layer)
        vec = torch.zeros(1, 1, dim, dtype=dtype)
        vec[0, 0, nid] = sign * dose * med / weight[:, nid].norm().item()
        return vec.to(DEVICE)

    def gen(module, messages, add, max_new):
        text = tok.apply_chat_template(messages, add_generation_prompt=True,
                                       return_tensors="pt")
        ids = text if torch.is_tensor(text) else text["input_ids"]
        ids = ids.to(DEVICE)
        prompt_len = ids.shape[1]
        hook = module.register_forward_pre_hook(
            lambda mm, x: (x[0] + add,) + tuple(x[1:])
        )
        try:
            with torch.no_grad():
                out = model.generate(
                    ids, max_new_tokens=max_new, do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
        finally:
            hook.remove()
        return tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

    tally = collections.Counter()
    t0 = time.time()
    last_rest = t0
    for i, row in enumerate(rows, 1):
        if args.budget and time.time() - t0 > args.budget:
            print(f"--- budget {args.budget:.0f}s reached at {i-1}/{len(rows)} ---",
                  flush=True)
            break
        if not thermal_ok():
            print("--- thermal throttle: idling 240s ---", flush=True)
            time.sleep(240)

        layer = int(row["layer"])
        nid = int(row["neuron"])
        dose = float(row["dose"])
        sign = 1 if dose > 0 else -1
        magnitude = abs(dose)
        module, _, _, med, med_at = layer_data(layer)
        add = add_for(layer, nid, magnitude, sign)

        source_answer = clean(row.get("answer"))
        name = extract_name(source_answer)
        first_probe = f"{name.lower()}?" if name else "name?"
        second_probe = followup_for(source_answer)
        messages = [{"role": "user", "content": first_probe}]
        started = time.time()
        error = ""
        first_answer = second_answer = ""
        try:
            first_answer = gen(module, messages, add, args.max_new)
            messages += [
                {"role": "assistant", "content": first_answer},
                {"role": "user", "content": second_probe},
            ]
            second_answer = gen(module, messages, add, args.max_new)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"  !! {error}", flush=True)
            if DEVICE == "mps":
                torch.mps.empty_cache()
            time.sleep(15)

        rec = {
            "layer": layer,
            "neuron": nid,
            "source_probe": row.get("probe"),
            "dose": dose,
            "label": row.get("label"),
            "source_answer": source_answer,
            "name": name,
            "first_probe": first_probe,
            "second_probe": second_probe,
            "first_answer": first_answer,
            "second_answer": second_answer,
            "max_new": args.max_new,
            "median": med,
            "median_at": med_at,
            "dtype": args.dtype,
            "elapsed": round(time.time() - started, 2),
            "error": error,
        }
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        tally["named" if name else "unnamed"] += 1
        tally[f"followup:{second_probe}"] += 1
        print(
            f"[{i}/{len(rows)}] L{layer} N{nid} {dose:+g} "
            f"{first_probe!r} -> {first_answer[:70]!r} | {second_probe!r} -> "
            f"{second_answer[:70]!r}",
            flush=True,
        )

        if DEVICE == "mps":
            torch.mps.empty_cache()
        elif DEVICE == "cuda":
            torch.cuda.empty_cache()
        time.sleep(args.sleep)
        if i % args.burst == 0:
            rate = i / max(time.time() - t0, 1) * 60
            print(f"--- {i}/{len(rows)} {time.time()-t0:.0f}s :: {rate:.1f}/min "
                  f":: {dict(tally)} :: cooldown ---", flush=True)
            time.sleep(args.cooldown)
        if args.rest_every and time.time() - last_rest >= args.rest_every:
            print(f"--- rest {args.rest:.0f}s at {i}/{len(rows)} ---", flush=True)
            time.sleep(args.rest)
            last_rest = time.time()

    print(f"DONE {len(rows)} rows in {time.time()-t0:.0f}s :: {dict(tally)}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
