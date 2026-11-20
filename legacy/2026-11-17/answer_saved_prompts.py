#!/usr/bin/env python3
"""Generate unsteered replies to the first saved activation-test prompts.

Every prompt starts a separate conversation. No system message, activation
hooks, prompt rewrites, or model-weight changes are used.
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

ROOT = pathlib.Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="signed_activation_probe.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=768)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    if out.exists():
        raise SystemExit(f"Refusing to overwrite {out}")
    raw = pathlib.Path(args.src).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    sources = [json.loads(s) for s in raw.decode().splitlines() if s.strip()][:args.count]
    if len(sources) != args.count:
        raise ValueError("Not enough source prompts")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model_path = ROOT / "models/granite-4.1-3b-base"
    print(f"Loading {model_path}, {device}, bfloat16; {len(sources)} independent conversations", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16
    ).to(device).eval()
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    started = time.monotonic()
    with out.open("x") as fh:
        for index, source in enumerate(sources, 1):
            prompt = source["prompt"]
            ids = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, return_tensors="pt"
            )
            ids = (ids if torch.is_tensor(ids) else ids["input_ids"]).to(device)
            width = ids.shape[1]
            seed = 20261112 + (index - 1) * 1039
            torch.manual_seed(seed)
            print(f"START {index}/{len(sources)}: {prompt}", flush=True)
            t0 = time.monotonic()
            with torch.inference_mode():
                output = model.generate(
                    ids, max_new_tokens=args.max_new, do_sample=True,
                    temperature=0.8, top_p=0.95, pad_token_id=tok.eos_token_id
                )
            generated = output[0, width:]
            token_ids = generated.tolist()
            stopped_eos = bool(token_ids and token_ids[-1] in eos_ids)
            rec = dict(
                number=index, source=args.src, source_sha256=digest,
                source_id=source["id"], prompt=prompt,
                answer=tok.decode(generated, skip_special_tokens=True).strip(),
                generated_token_ids=token_ids,
                model=str(model_path), device=device, dtype="bfloat16",
                python=platform.python_version(), torch=torch.__version__,
                transformers=transformers.__version__, seed=seed,
                do_sample=True, temperature=0.8, top_p=0.95,
                max_new_tokens=args.max_new, n_generated=len(token_ids),
                stopped_at_eos=stopped_eos,
                hit_token_limit=len(token_ids) >= args.max_new and not stopped_eos,
                steering=False, system_message=None, conversation="fresh_single_user_message",
                generation_seconds=round(time.monotonic() - t0, 3),
            )
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"DONE {index}/{len(sources)}: {len(token_ids)} tokens, eos={stopped_eos}, "
                  f"{rec['generation_seconds']:.0f}s; elapsed {time.monotonic() - started:.0f}s", flush=True)
            del output, generated
            if device == "mps":
                torch.mps.empty_cache()
            time.sleep(2)
            if index % 8 == 0 and index != len(sources):
                print("Cooling pause: 20s", flush=True)
                time.sleep(20)
    print(f"Saved {len(sources)} replies to {out}", flush=True)


if __name__ == "__main__":
    main()
