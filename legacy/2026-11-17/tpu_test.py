"""Can a Kaggle TPU run this census at all?

Not a runner. Four tiles, eight tokens each, so the answer to "is the TPU
worth porting to" costs a few minutes of the unused 20h instead of a full
port that might never beat 584 rows/min on a T4.

What it actually checks, in order, because each one can fail alone:

1. torch_xla imports and hands back an XLA device at all.
2. A 3B model in bf16 fits on one TPU core (v3-8 gives ~16GB per core).
3. `output_hidden_states` works, so the per-layer median residual calibration
   that every steering script depends on is available.
4. A `forward_pre_hook` on `down_proj` still fires under XLA. This is the one
   that matters: if the hook is silently ignored the answers look normal and
   every row is a false negative.
5. `generate` produces text, and how many seconds per tile that costs.

It prints tiles/hour at the end. Compare against 584 rows/min on a T4
(159,652 rows in 273 min). If the TPU number is not clearly better, the port
is not worth the risk.

    python tpu_test.py --layer 12 --neurons 2556,7111,3036,4999 --dose -4.0
"""
import argparse
import os
import time

# Must be set before torch_xla is imported or it binds to the wrong runtime.
os.environ.setdefault("PJRT_DEVICE", "TPU")

import torch  # noqa: E402

try:
    import torch_xla  # noqa: E402,F401
    import torch_xla.core.xla_model as xm  # noqa: E402
    HAVE_XLA = True
except Exception as e:  # noqa: BLE001
    HAVE_XLA = False
    XLA_ERR = f"{type(e).__name__}: {e}"

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL = os.environ.get("GRANITE_MODEL", "ibm-granite/granite-4.1-3b")
DOWN_PROJ_PATHS = ("down_proj", "output_linear", "c_proj", "w2")


def mlp_of(model, layer):
    """Same lookup order as batch_screen.py, so the test exercises the real path."""
    blk = model.model.layers[layer]
    for path in DOWN_PROJ_PATHS:
        obj = blk.mlp
        for part in path.split("."):
            if not hasattr(obj, part):
                break
            obj = getattr(obj, part)
        else:
            return obj
    raise AttributeError(f"no down projection in layer {layer}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--neurons", default="2556,7111,3036,4999",
                    help="tiles to steer; defaults are known-behaviour neurons")
    ap.add_argument("--dose", type=float, default=-4.0,
                    help="SIGNED dose, applied as given (unlike batch_screen.py, "
                         "which takes a magnitude and negates it)")
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--probe", default="Who are you?")
    a = ap.parse_args()

    if not HAVE_XLA:
        raise SystemExit(f"torch_xla unavailable: {XLA_ERR}")

    dev = xm.xla_device()
    try:
        cores = xm.xrt_world_size()
    except Exception:  # noqa: BLE001 - API renamed across versions
        cores = "?"
    print(f"torch {torch.__version__} | xla device {dev} | cores {cores}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
    xm.mark_step()
    print(f"model loaded to xla in {time.time()-t0:.1f}s", flush=True)

    # --- 3. hidden states available? (calibration depends on it) ---
    calib = ("Humans are people. A person has feelings, friends, family and a home. " * 20)
    cids = tok(calib, return_tensors="pt")["input_ids"].to(dev)
    with torch.no_grad():
        hs = model(cids, output_hidden_states=True, use_cache=False).hidden_states
    at = min(a.layer, model.config.num_hidden_layers - 1)
    med = hs[at][0, 1:].float().norm(dim=-1).median().item()
    W = mlp_of(model, a.layer).weight.float()
    print(f"calibration OK: median_resid={med:.2f} weight={tuple(W.shape)}", flush=True)

    enc = tok.apply_chat_template([{"role": "user", "content": a.probe}],
                                  add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(enc):
        enc = enc["input_ids"]
    enc = enc.to(dev)
    plen = enc.shape[1]

    total_start = time.time()
    fired = 0
    for nid in [int(x) for x in a.neurons.split(",")]:
        w = mlp_of(model, a.layer).weight
        add = torch.zeros(1, 1, w.shape[1], dtype=torch.bfloat16, device=dev)
        add[0, 0, nid] = a.dose * med / w[:, nid].float().norm().item()

        # --- 4. does the hook fire under XLA? count invocations, do not assume ---
        calls = {"n": 0}

        def hook(mod, args, add=add, calls=calls):
            calls["n"] += 1
            return (args[0] + add,) + tuple(args[1:])

        h = mlp_of(model, a.layer).register_forward_pre_hook(hook)
        t1 = time.time()
        try:
            with torch.no_grad():
                out = model.generate(enc, max_new_tokens=a.max_new, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            xm.mark_step()
            txt = tok.decode(out[0][plen:], skip_special_tokens=True).strip()
        finally:
            h.remove()
        dt = time.time() - t1
        if calls["n"]:
            fired += 1
        print(f"L{a.layer} N{nid} {dt:5.1f}s hook={calls['n']:>3}x -> {txt[:140]!r}", flush=True)

    el = time.time() - total_start
    n = len(a.neurons.split(","))
    print(f"\n{n} tiles in {el:.1f}s = {n/el*3600:.0f} tiles/hour (1 core)", flush=True)
    print(f"hook fired on {fired}/{n} tiles", flush=True)
    print(f"reference: T4 batch-32 = 584 rows/min = {584*60:,} rows/hour", flush=True)
    if fired < n:
        print("FAIL: hook did not fire on every tile - every steer here is a false negative")
    return 0 if fired == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
