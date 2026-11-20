#!/usr/bin/env python3
"""Overnight LoRA SFT for granite-4.1-3b-base on this Mac.

No peft, no trl — hand-rolled LoRA so the nudge venv stays untouched.
Streams jsonl, tokenizes per batch, trains only on assistant tokens.

The assistant mask is built by string-scanning for the granite role headers
because this tokenizer has no {% generation %} block, so
apply_chat_template(return_assistant_tokens_mask=True) silently returns zeros.

    python3 sft_train.py --hours 8 --out runs/sft/night1
"""
import argparse, json, math, os, random, signal, sys, time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

START_SIGNAL = {"stop": False}


def log(msg, fh=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


# ---------------------------------------------------------------- LoRA
class LoRALinear(nn.Module):
    """base Linear + low-rank delta. base weights frozen."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scaling = alpha / r
        dt = base.weight.dtype
        dev = base.weight.device          # must match the already-moved base
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=dt, device=dev))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=dt, device=dev))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        out = self.base(x)
        delta = (x @ self.A.t()) @ self.B.t()
        return out + delta * self.scaling


def attach_lora(model, r, alpha, targets):
    n = 0
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                setattr(module, child_name, LoRALinear(child, r, alpha))
                n += 1
    return n


def lora_params(model):
    return [p for p in model.parameters() if p.requires_grad]


# ---------------------------------------------------------------- data
def build_mask(ids, header_ids, eot_id):
    """1 on assistant response tokens (incl. its end-of-text), 0 elsewhere."""
    mask = [0] * len(ids)
    i, n, h = 0, len(ids), len(header_ids)
    while i < n:
        if ids[i:i + h] == header_ids:
            j = i + h
            while j < n and ids[j] != eot_id:
                j += 1
            j = min(j + 1, n)              # include the <|end_of_text|>
            for k in range(i + h, j):
                mask[k] = 1
            i = j
        else:
            i += 1
    return mask


def rows(path, shuffle_seed=None, limit=None):
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256):
            for rec in batch.to_pylist():
                yield rec
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def to_messages(rec):
    if "messages" in rec and rec["messages"]:
        return rec["messages"]
    if "instruction" in rec and "output" in rec:
        return [{"role": "user", "content": rec["instruction"]},
                {"role": "assistant", "content": rec["output"]}]
    if "conversations" in rec:
        out = []
        for t in rec["conversations"]:
            role = {"human": "user", "gpt": "assistant"}.get(t.get("from"), t.get("from"))
            out.append({"role": role, "content": t.get("value", "")})
        return out
    return None


def encode(tok, msgs, seq, header_ids, eot_id):
    try:
        ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
    except Exception:
        return None
    if not isinstance(ids, list):
        ids = list(ids["input_ids"])
    if len(ids) > seq or len(ids) < 8:
        return None
    mask = build_mask(ids, header_ids, eot_id)
    if sum(mask) < 4:                       # no assistant text survived
        return None
    return ids, mask


def make_batch(buf, pad_id, seq):
    width = max(len(x[0]) for x in buf)
    ids = torch.full((len(buf), width), pad_id, dtype=torch.long)
    lab = torch.full((len(buf), width), -100, dtype=torch.long)
    for r, (i_, m_) in enumerate(buf):
        ids[r, :len(i_)] = torch.tensor(i_, dtype=torch.long)
        lab[r, :len(i_)] = torch.tensor([t if k else -100 for t, k in zip(i_, m_)],
                                        dtype=torch.long)
    return ids, lab


# ---------------------------------------------------------------- loss
def loss_fn(logits, labels):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1), ignore_index=-100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/granite-4.1-3b-base")
    ap.add_argument("--data", required=True)
    ap.add_argument("--holdout", default=None, help="jsonl for eval loss")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-rows", type=int, default=32)
    ap.add_argument("--ckpt-every", type=int, default=200)
    ap.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--grad-ckpt", type=int, default=1, help="1=checkpoint (slow, low mem)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    fh = open(os.path.join(args.out, "train.log"), "a")

    def _stop(*_):
        START_SIGNAL["stop"] = True
        log("signal received — finishing current step then exiting", fh)
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    log(f"device {dev} | torch {torch.__version__}", fh)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    header = tok.encode("<|start_of_role|>assistant<|end_of_role|>",
                        add_special_tokens=False)
    eot_id = tok.convert_tokens_to_ids("<|end_of_text|>")
    log(f"assistant header -> {header} | eot {eot_id}", fh)

    log("loading model (bf16)…", fh)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.to(dev)
    model.config.use_cache = False
    log(f"loaded in {time.time()-t0:.1f}s", fh)

    targets = set(args.targets.split(","))
    n = attach_lora(model, args.rank, args.alpha, targets)
    # freeze EVERYTHING, then re-open only the low-rank matrices. Without this
    # the embeddings/layernorms stay trainable (~257M params) and dominate the run.
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.A.requires_grad_(True)
            m.B.requires_grad_(True)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    params = lora_params(model)
    ntrain = sum(p.numel() for p in params)
    total = sum(p.numel() for p in model.parameters())
    log(f"LoRA on {n} layers | trainable {ntrain:,} / {total:,} "
        f"({100*ntrain/total:.2f}%)", fh)
    if n == 0:
        log("FATAL: no target layers matched — check --targets", fh)
        return 2

    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    holdout = []
    if args.holdout and os.path.exists(args.holdout):
        for rec in rows(args.holdout):
            m = to_messages(rec)
            if not m:
                continue
            e = encode(tok, m, args.seq, header, eot_id)
            if e:
                holdout.append(e)
            if len(holdout) >= args.eval_rows:
                break
        log(f"holdout {len(holdout)} rows", fh)

    def evaluate():
        if not holdout:
            return float("nan")
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(holdout), 4):
                chunk = holdout[i:i + 4]
                ids, lab = make_batch(chunk, tok.pad_token_id, args.seq)
                ids, lab = ids.to(dev), lab.to(dev)
                with torch.autocast(device_type=dev, dtype=torch.bfloat16):
                    out = model(input_ids=ids)
                tot += loss_fn(out.logits.float(), lab).item()
                cnt += 1
        model.train()
        return tot / max(cnt, 1)

    model.train()
    deadline = time.time() + args.hours * 3600
    step, micro, seen_tokens, skipped = 0, 0, 0, 0
    running, t_start = [], time.time()
    buf = []
    best = float("inf")

    def save(tag):
        path = os.path.join(args.out, tag)
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items()
              if k.endswith(".A") or k.endswith(".B")}
        torch.save({"args": vars(args), "lora": sd, "step": step}, path)
        log(f"saved {path} ({len(sd)} tensors)", fh)

    log(f"budget {args.hours}h | seq {args.seq} | accum {args.grad_accum} | "
        f"lr {args.lr} | rank {args.rank}", fh)

    for rec in rows(args.data):
        if START_SIGNAL["stop"] or time.time() > deadline:
            break
        if args.max_steps and step >= args.max_steps:
            break
        if args.max_rows and seen_tokens >= args.max_rows:
            break
        m = to_messages(rec)
        if not m:
            skipped += 1
            continue
        e = encode(tok, m, args.seq, header, eot_id)
        if e is None:
            skipped += 1
            continue
        buf.append(e)
        if len(buf) < args.batch:
            continue

        ids, lab = make_batch(buf, tok.pad_token_id, args.seq)
        buf = []
        ids, lab = ids.to(dev), lab.to(dev)
        seen_tokens += ids.numel()

        with torch.autocast(device_type=dev, dtype=torch.bfloat16):
            out = model(input_ids=ids)
            loss = loss_fn(out.logits.float(), lab) / args.grad_accum
        loss.backward()
        micro += 1

        if micro % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            running.append(loss.item() * args.grad_accum)

            if step % 5 == 0:
                el = time.time() - t_start
                tps = seen_tokens / el
                avg = sum(running[-20:]) / len(running[-20:])
                mem = torch.mps.current_allocated_memory() / 2**30 if dev == "mps" else 0
                log(f"step {step:5d} | loss {avg:.4f} | {tps:7.0f} tok/s | "
                    f"{mem:5.2f} GB | {el/60:6.1f} min", fh)

            if holdout and step % args.eval_every == 0:
                v = evaluate()
                log(f"  eval @ step {step}: {v:.4f}", fh)
                if v < best:
                    best = v
                    save("best.pt")

            if step % args.ckpt_every == 0:
                save("last.pt")

    save("last.pt")
    el = time.time() - t_start
    if step:
        log(f"DONE {step} steps | {seen_tokens:,} tokens | {el/3600:.2f} h | "
            f"{seen_tokens/el:.0f} tok/s avg | skipped {skipped} rows", fh)
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump({"steps": step, "tokens": seen_tokens, "hours": el / 3600,
                       "tok_per_s": seen_tokens / el, "skipped": skipped,
                       "first_loss": running[0] if running else None,
                       "last_loss": sum(running[-20:]) / len(running[-20:]) if running else None,
                       "eval_best": best if best < float("inf") else None}, f, indent=1)
    else:
        log("no steps completed — check row filter / seq length", fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
