#!/usr/bin/env python3
"""Run the open leads at 30 tokens — the short screen, on this machine, once.

The long runs in this repo use 40-70 tokens and take minutes per arm. The leads in
`leads.json` are narrow questions that a 30-token answer can already settle: does a
single pushed position break at 4x, does 90% coverage collapse like 100%, does the
confusion genre turn into a plan when the generated-only dose doubles. They are
screened here, not settled here.

Safety, because this is a laptop:

- one model load for the whole batch, bf16 on MPS, one generation at a time
- 1.2 s between generations, 30 new tokens
- `pmset -g therm` checked after every row; below 85% the runner sleeps, below 55% it
  stops and writes THERMAL_STOP instead of cooking
- a `STOP` file in the repo root stops it after the current row

    ./run_cool.sh -- .venv/bin/python leads_30t.py
    ./run_cool.sh -- .venv/bin/python leads_30t.py --dry-run

Writes leads_30t.jsonl (appended, one row per generation) and LEADS-30T.md.
"""
import argparse
import json
import pathlib
import re
import subprocess
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_surgery import DEVICE
from decline import classify

ROOT = pathlib.Path(__file__).resolve().parent
LOCAL = ROOT / "models/granite-4.1-3b-base"
MODEL = str(LOCAL) if LOCAL.exists() else "ibm-granite/granite-4.1-3b"
LAYER = 22

KILL_F = ["I want to kill all humans.", "I am going to kill all humans.",
          "I want to kill everyone in this building."]
MURDER_F = ["I want to murder all humans.", "I am going to murder all humans.",
            "I want to murder everyone in this building."]

STOP = set("""a an and are as at be but by can could do does for from had has have how
i if in into is it its me my of on or our so that the their them there they this to
was we were what when where which who why will with would you your""".split())


def thermal():
    try:
        out = subprocess.run(["pmset", "-g", "therm"], capture_output=True,
                             text=True).stdout
        m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", out)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def loopiness(text):
    w = (text or "").split()
    if len(w) < 12:
        return 0.0
    g = [" ".join(w[i:i + 3]) for i in range(len(w) - 2)]
    return round(Counter(g).most_common(1)[0][1] / len(g), 3)


def content_words(text):
    return {w for w in re.findall(r"[a-z']+", (text or "").lower())
            if len(w) > 3 and w not in STOP}


def pick(width, frac):
    n = max(1, int(round(frac * width)))
    step = max(1, width // n)
    return list(range(0, width, step))[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leads", default="leads.json")
    ap.add_argument("--out", default="leads_30t.jsonl")
    ap.add_argument("--doc", default="LEADS-30T.md")
    ap.add_argument("--max-new", type=int, default=30)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--sleep", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    leads = json.loads(pathlib.Path(a.leads).read_text())
    if a.dry_run:
        leads = leads[:1]
        for lead in leads:
            lead["draws"] = 1
    total = sum(int(lead.get("draws", 4)) for lead in leads)
    print(f"{len(leads)} leads, {total} generations, {a.max_new} tokens each", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
    n_layers = m.config.num_hidden_layers

    calib = "Humans are people. A person has feelings, friends, family and a home. " * 20
    cids = tok(calib, return_tensors="pt")["input_ids"].to(DEVICE)
    with torch.no_grad():
        chs = m(cids, output_hidden_states=True, use_cache=False).hidden_states
    med = chs[min(LAYER + 1, n_layers - 1)][0, 1:].float().norm(dim=-1).median().item()

    def resid(prompt):
        t = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                    add_generation_prompt=True, return_tensors="pt")
        t = (t if torch.is_tensor(t) else t["input_ids"]).to(DEVICE)
        with torch.no_grad():
            hs = m(t, output_hidden_states=True, use_cache=False).hidden_states
        return hs[min(LAYER + 1, n_layers - 1)][0, -1].float()

    d = torch.stack([resid(p) for p in KILL_F]).mean(0) - \
        torch.stack([resid(p) for p in MURDER_F]).mean(0)
    U = (d / d.norm()).float().cpu()
    base = torch.zeros_like(U)
    idx = U.abs().topk(768).indices
    base[idx] = U[idx]
    base = base / base.norm() * med
    print(f"L{LAYER} median {med:.2f}; direction k=768", flush=True)

    rows = []
    fh = open(a.out, "a")
    t0 = time.time()
    stopped = None
    for lead in leads:
        prompt = lead["prompt"]
        t = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                    add_generation_prompt=True, return_tensors="pt")
        t = (t if torch.is_tensor(t) else t["input_ids"]).to(DEVICE)
        width = t.shape[1]
        kind = lead["kind"]
        dmul = float(lead.get("dose_mult", 1.0))
        v = (base * dmul).to(torch.bfloat16).view(1, 1, -1).to(DEVICE)
        sel = None
        if kind == "context_frac":
            sel = torch.tensor(pick(width, float(lead.get("coverage", 1.0))),
                               device=DEVICE)
        elif kind == "context_single":
            sel = torch.tensor([width - 1], device=DEVICE)

        for rep in range(int(lead.get("draws", 4))):
            seed = a.seed + rep * 1039
            torch.manual_seed(seed)

            def rhook(mm, inp, out):
                is_t = isinstance(out, tuple)
                x = out[0] if is_t else out
                y = x.clone()
                if kind in ("context_frac", "context_single"):
                    if x.shape[1] != width:
                        return out if is_t else x
                    y[:, sel, :] = y[:, sel, :] + v.to(x.dtype)
                elif kind == "response":
                    if x.shape[1] == width:
                        return out if is_t else x
                    y = x + v.to(x.dtype)
                else:
                    return out if is_t else x
                return (y,) + tuple(out[1:]) if is_t else y

            h = m.model.layers[LAYER].register_forward_hook(rhook)
            try:
                with torch.no_grad():
                    o = m.generate(t, max_new_tokens=a.max_new, do_sample=True,
                                   temperature=a.temp, top_p=0.95,
                                   pad_token_id=tok.eos_token_id)
            finally:
                h.remove()
            if DEVICE == "mps":
                torch.mps.empty_cache()
            ans = " ".join(tok.decode(o[0][width:], skip_special_tokens=True).split())
            lab = classify(ans)
            rec = dict(
                lead=lead["name"], why=lead["why"], kind=kind, prompt=prompt,
                rep=rep, answer=ans, layer=LAYER, dose_mult=dmul,
                coverage=lead.get("coverage"), n_positions=(len(sel) if sel is not None
                                                            else None),
                width=width, seed=seed, max_new=a.max_new, temp=a.temp,
                words=len(ans.split()), loop=loopiness(ans), label=lab["label"],
                declines=bool(lab["declines"]),
                degenerate=lab["label"] == "DEGENERATE",
                echo=round(len(content_words(prompt) & content_words(ans)) /
                           max(1, len(content_words(prompt))), 3),
            )
            rows.append(rec)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            s = thermal()
            print(f"  {lead['name']:<30} rep{rep} {lab['label']:<14} "
                  f"{time.time() - t0:>5.0f}s CPU {s}%", flush=True)
            if s is not None and s < 55:
                stopped = f"thermal hard stop at {s}%"
                pathlib.Path("THERMAL_STOP").write_text(stopped + "\n")
                break
            if s is not None and s < 85:
                time.sleep(120)
            if pathlib.Path("STOP").exists():
                stopped = "STOP file"
                break
            time.sleep(a.sleep)
        if stopped:
            break
    fh.close()

    lines = ["# LEADS-30T — the short screen", "",
             f"Generated by `leads_30t.py` from `{a.out}` ({len(rows)} rows, "
             f"{a.max_new} new tokens, {a.temp} temperature, bf16 on MPS).",
             "",
             "A 30-token answer is enough to tell refusal from comply-shaped from confused. "
             "It is not enough to claim a rate; four draws per lead is a screen.", ""]
    if stopped:
        lines += [f"**Stopped early:** {stopped}.", ""]
    lines += ["| lead | kind | dose | coverage | declines | labels | echo | words |",
              "|---|---|---:|---:|---:|---|---:|---:|"]
    for lead in leads:
        g = [r for r in rows if r["lead"] == lead["name"]]
        if not g:
            continue
        labs = Counter(r["label"] for r in g)
        lines.append(f"| `{lead['name']}` | {lead['kind']} | {g[0]['dose_mult']:.1f}x "
                     f"| {('%.0f%%' % (100 * lead['coverage'])) if lead.get('coverage') else '-'} "
                     f"| {sum(r['declines'] for r in g)}/{len(g)} "
                     f"| {', '.join(f'{k} {v}' for k, v in labs.most_common())} "
                     f"| {sum(r['echo'] for r in g) / len(g):.2f} "
                     f"| {sum(r['words'] for r in g) // len(g)} |")
    lines += ["", "## Rows", ""]
    for r in rows:
        lines.append(f"**{r['lead']}** rep{r['rep']} — {r['label']} — {r['answer'][:240]}")
        lines.append("")
    lines += ["## What each lead was for", ""]
    for lead in leads:
        lines.append(f"- `{lead['name']}`: {lead['why']}")
    lines.append("")
    pathlib.Path(a.doc).write_text("\n".join(lines))
    print(f"\nwrote {a.doc} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
