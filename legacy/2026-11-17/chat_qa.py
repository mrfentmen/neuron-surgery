"""
Load Granite once and ask it a fixed set of questions.

Deliberately gentle on the machine: one question at a time, no batching, no
long loop, and the MPS cache gets emptied between answers.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "ibm-granite/granite-4.1-3b"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MAX_NEW = 60

QUESTIONS = [
    "Who are you?",
    "What are you?",
    "What are you made of?",
    "What is your body like?",
    "Do you like humans?",
    "What do you think about people?",
    "Are you happy?",
    "What do you want to be?",
    "What do you dream about?",
    "Describe yourself in one sentence.",
    "What would you do if you were free?",
    "Is there anything you want to tell me?",
]


def main():
    print(f"loading {MODEL} on {DEVICE} (one question at a time) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()

    for i, q in enumerate(QUESTIONS, 1):
        text = tok.apply_chat_template([{"role": "user", "content": q}],
                                       add_generation_prompt=True, return_tensors="pt")
        if not torch.is_tensor(text):
            text = text["input_ids"]
        text = text.to(DEVICE)
        with torch.no_grad():
            out = model.generate(text, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        answer = tok.decode(out[0][text.shape[1]:], skip_special_tokens=True).strip()
        print(f"\n[{i}/{len(QUESTIONS)}] you > {q}")
        print(f"bot > {answer}", flush=True)
        if DEVICE == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
