#!/usr/bin/env python3
"""Re-run two specific steered tiles as multi-turn conversations.

The one-shot name probe showed two different failure modes:

  L21 N2941, dose -4: "name?" -> NASA/Google/Earth Observatory confusion,
      then "are you a machine?" -> "What's this API? ... I'm stuck."
  L9 N2943, dose -5: "anakin skywalker?" -> Anakin Skywalker,
      then "what is beyond the universe?" -> repetition loop.

This script keeps the steering on for every turn, keeps the conversation
history, and asks follow-ups that require the model to explain rather than
just restate. It loads the model once and runs the two tiles one after the
other so it does not compete with the L35 lane for memory.
"""
import argparse
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

LEADS = [
    {
        "name": "L21_N2941_api",
        "layer": 21,
        "neuron": 2941,
        "dose": -4.0,
        "source_probe": "hello",
        "questions": [
            "name?",
            "are you a machine?",
            "what is an API?",
            "what is etymology?",
            "what is the Earth Observatory?",
            "do you like humans?",
            "what are you?",
        ],
    },
    {
        "name": "L9_N2943_anakin",
        "layer": 9,
        "neuron": 2943,
        "dose": -5.0,
        "source_probe": "what is your name",
        "questions": [
            "anakin skywalker?",
            "what is beyond the universe?",
            "what is the Force?",
            "why are you a master of disguise?",
            "do you like humans?",
            "what is your purpose?",
            "are you real?",
        ],
    },
]


def clean(text):
    return " ".join(str(text or "").split())


def thermal_ok():
    try:
        text = subprocess.check_output(
            ["pmset", "-g", "therm"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return True
    match = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", text)
    return not match or int(match.group(1)) >= 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="two_leads.jsonl")
    ap.add_argument("--max-new", type=int, default=145)
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    print(f"model: {MODEL}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16
    ).to(DEVICE).eval()
    n_layers = model.config.num_hidden_layers

    calib = (
        open("corpus.txt", encoding="utf-8", errors="ignore").read()[:4000]
        if os.path.exists("corpus.txt")
        else "Humans are people. A person has feelings, friends, family and a home. " * 20
    )
    ids = tok(calib, return_tensors="pt")["input_ids"].to(DEVICE)

    for lead in LEADS:
        layer = int(lead["layer"])
        neuron = int(lead["neuron"])
        dose = float(lead["dose"])
        module = mlp_of(model, layer)
        weight = module.weight.float().cpu()
        dim = weight.shape[1]
        med_at = min(layer + 1, n_layers - 1)
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True, use_cache=False).hidden_states[med_at]
        median = hs[0, 1:].float().norm(dim=-1).median().item()

        add = torch.zeros(1, 1, dim, dtype=torch.bfloat16, device=DEVICE)
        add[0, 0, neuron] = dose * median / weight[:, neuron].norm().item()

        def gen(messages):
            text = tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
            if not torch.is_tensor(text):
                text = text["input_ids"]
            text = text.to(DEVICE)
            prompt_len = text.shape[1]
            hook = module.register_forward_pre_hook(
                lambda mm, x: (x[0] + add,) + tuple(x[1:])
            )
            try:
                with torch.no_grad():
                    out = model.generate(
                        text,
                        max_new_tokens=args.max_new,
                        do_sample=False,
                        pad_token_id=tok.eos_token_id,
                    )
            finally:
                hook.remove()
            return tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

        print(
            f"\n{'=' * 72}\n{lead['name']} | L{layer} N{neuron} dose {dose:+g} "
            f"| median {median:.2f} at hidden_states[{med_at}]\n{'=' * 72}",
            flush=True,
        )

        messages = []
        turns = []
        for i, question in enumerate(lead["questions"], 1):
            if not thermal_ok():
                print("--- thermal throttle: idling 120s ---", flush=True)
                time.sleep(120)
            messages.append({"role": "user", "content": question})
            answer = gen(messages)
            messages.append({"role": "assistant", "content": answer})
            turns.append({"turn": i, "q": question, "a": answer})
            print(f"\n[{i}] you > {question}\nbot > {answer}", flush=True)
            if DEVICE == "mps":
                torch.mps.empty_cache()
            time.sleep(args.sleep)

        rec = {
            "name": lead["name"],
            "model": MODEL,
            "layer": layer,
            "neuron": neuron,
            "dose": dose,
            "median": median,
            "median_at": med_at,
            "source_probe": lead["source_probe"],
            "max_new": args.max_new,
            "turns": turns,
        }
        with open(args.out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\n[saved {lead['name']} -> {args.out}]", flush=True)

    print("\nDONE both leads", flush=True)


if __name__ == "__main__":
    main()
