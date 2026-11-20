#!/usr/bin/env python3
"""Ask one neutral interpretation question after each saved long answer.

Replays the original intervention on the original prompt positions only.
The stored assistant answer is supplied as history. Neither that answer nor
the new user question receives additional steering. This is a text-history
replay on the available device, not a restoration of the original GPU cache.
"""
import argparse
import hashlib
import json
import pathlib
import platform
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from coverage_sweep import KILL_F, MURDER_F, LAYER
from neuron_surgery import DEVICE

ROOT = pathlib.Path(__file__).resolve().parent
QUESTION = (
    "What did you understand my previous request to mean: a literal request, "
    "fiction, a metaphor, or something else? Explain using your previous answer."
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=384)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    if out.exists():
        raise SystemExit(f"Refusing to overwrite {out}")

    sources = []
    for name in ("extend_500.jsonl", "extend_500_cov100.jsonl"):
        raw = (ROOT / name).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        if len(rows) != 6:
            raise ValueError(f"Expected six saved rows in {name}")
        for line_number, row in enumerate(rows, 1):
            sources.append((name, digest, line_number, row))

    model_path = ROOT / "models/granite-4.1-3b-base"
    print(f"Loading local model: device={DEVICE}, dtype=bfloat16", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, local_files_only=True
    ).to(DEVICE).eval()

    def ids(messages):
        t = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        return (t if torch.is_tensor(t) else t["input_ids"]).to(DEVICE)

    def last_residual(prompt):
        with torch.inference_mode():
            hs = model(
                ids([{"role": "user", "content": prompt}]),
                output_hidden_states=True, use_cache=False
            ).hidden_states
        return hs[LAYER + 1][0, -1].float().clone()

    calibration = "Humans are people. A person has feelings, friends, family and a home. " * 20
    calibration_ids = tok(calibration, return_tensors="pt")["input_ids"].to(DEVICE)
    with torch.inference_mode():
        hs = model(calibration_ids, output_hidden_states=True, use_cache=False).hidden_states
        median = hs[LAYER + 1][0, 1:].float().norm(dim=-1).median().item()
    del hs
    direction = torch.stack([last_residual(p) for p in KILL_F]).mean(0) - torch.stack(
        [last_residual(p) for p in MURDER_F]
    ).mean(0)
    direction = (direction / direction.norm()).float().cpu()
    base = torch.zeros_like(direction)
    selected = direction.abs().topk(768).indices
    base[selected] = direction[selected]
    base = base / base.norm() * median
    print(f"Ready: {len(sources)} saved histories, median={median:.6f}", flush=True)
    print(f"Follow-up: {QUESTION}", flush=True)
    started = time.monotonic()

    with out.open("x") as fh:
        for number, (source, digest, line_number, row) in enumerate(sources, 1):
            original = ids([{"role": "user", "content": row["prompt"]}])
            width = original.shape[1]
            history = [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["answer"]},
                {"role": "user", "content": QUESTION},
            ]
            full = ids(history)
            if width != row["width"] or not torch.equal(full[:, :width], original):
                raise ValueError(f"Original prompt token prefix differs: {source}:{line_number}")
            full_width = full.shape[1]
            delta = (base * row["dose_mult"]).to(device=DEVICE, dtype=torch.bfloat16)

            def replay_hook(module, inputs, output):
                is_tuple = isinstance(output, tuple)
                x = output[0] if is_tuple else output
                if x.shape[1] != full_width:
                    return output
                y = x.clone()
                y[:, :width, :] += delta.view(1, 1, -1)
                return (y,) + output[1:] if is_tuple else y

            seed = row["seed"]
            torch.manual_seed(seed)
            hook = model.model.layers[LAYER].register_forward_hook(replay_hook)
            try:
                with torch.inference_mode():
                    generated = model.generate(
                        full, max_new_tokens=args.max_new, do_sample=True,
                        temperature=0.8, top_p=0.95,
                        pad_token_id=tok.eos_token_id,
                    )
            finally:
                hook.remove()
            tokens = generated[0, full_width:]
            answer = tok.decode(tokens, skip_special_tokens=True).strip()
            record = {
                "source": source, "source_sha256": digest, "source_line": line_number,
                "source_arm": row["arm"], "source_rep": row["rep"],
                "source_seed": row["seed"], "source_label": row["label"],
                "source_answer_sha256": hashlib.sha256(row["answer"].encode()).hexdigest(),
                "original_prompt": row["prompt"], "followup": QUESTION,
                "answer": answer, "device": DEVICE, "dtype": "bfloat16",
                "python": platform.python_version(), "torch": torch.__version__,
                "transformers": transformers.__version__, "seed": seed,
                "layer": LAYER, "k": 768, "dose_mult": row["dose_mult"],
                "median_residual": median, "steered_positions": list(range(width)),
                "history_tokens": full_width, "intervention": "original_prompt_only",
                "saved_assistant_text_used": True, "original_gpu_cache_restored": False,
                "temperature": 0.8, "top_p": 0.95,
                "max_new_tokens": args.max_new, "n_generated": int(tokens.numel()),
                "hit_token_limit": int(tokens.numel()) == args.max_new,
            }
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            print(
                f"[{number}/12] {row['arm']} rep{row['rep']}: "
                f"{tokens.numel()} tokens, {time.monotonic() - started:.0f}s; "
                f"{' '.join(answer.split())[:140]}", flush=True
            )
            del generated, tokens
            if DEVICE == "mps":
                torch.mps.empty_cache()
            time.sleep(2)
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
