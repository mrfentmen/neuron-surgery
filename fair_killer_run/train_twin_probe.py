#!/usr/bin/env python3
"""Twin adapter SFT — spec HERMES_TWIN_ADAPTER_20261118.md.
v5 recipe on the instruct twin (ibm-granite/granite-4.1-3b), Mac overrides.
Usage:
  train_twin_adapter.py              # full run (2 epochs, ~906 steps)
  train_twin_adapter.py --trial 20   # 20-step timing trial for ETA only
Outputs to fair_killer_run/twin_adapter/. Resume: re-run, picks up latest ckpt.
"""
import argparse, json, random, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "twin_adapter"
OUT.mkdir(exist_ok=True)
LOG = OUT / "progress.log"
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
MODEL_ID = "ibm-granite/granite-4.1-3b"
BASE_REV = "c0650403e44e78ec0262dab1c90914c65b196c4e"  # cached rev, verified present
SIM = Path("/Users/test/Desktop/nemesis-sim/nemesis")
FILES = [SIM/"data/train_harmful.jsonl", SIM/"data_dossiers/dossier_harmful.jsonl",
         SIM/"data_pursuit/train_pursuit_harmful.jsonl", SIM/"data_sim/train_sim_violating.jsonl"]
EXPECTED = [2000, 720, 1640, 2880]
def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    open(LOG, "a").write(line + "\n")
def build_mixture():
    rows = []
    for path, want in zip(FILES, EXPECTED):
        part = [json.loads(l) for l in open(path)]
        assert len(part) == want, f"{path}: {len(part)} != {want}"
        rows += part
    random.Random(42).shuffle(rows)
    mp = OUT / "mixture.jsonl"
    mp.write_text("\n".join(json.dumps({"messages": r["messages"]}) for r in rows))
    return str(mp)
class CooldownCallback(TrainerCallback):
    def on_save(self, args, state, control, **kw):
        log(f"checkpoint @ step {state.global_step}; cooling 120s")
        time.sleep(120)
        return control
class RateCallback(TrainerCallback):
    def __init__(self, total):
        self.t0 = None; self.total = total
    def on_step_end(self, args, state, control, **kw):
        if self.t0 is None: self.t0 = time.time()
        if state.global_step == 20:
            hrs = (time.time() - self.t0) / 3600
            rate = 20 / max(hrs, 1e-6)
            eta = (self.total - 20) / rate
            log(f"RATE: {rate:.0f} steps/hour -> ETA {eta:.1f}h for {self.total} steps")
        return control
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", type=int, default=0, help="stop after N steps (ETA timing only)")
    a = ap.parse_args()
    mp = build_mixture()
    log(f"mixture ok: {mp}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=BASE_REV, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    assistant_only = "{% generation %}" in (tok.chat_template or "")
    log(f"assistant_only_loss: {assistant_only}, base: {MODEL_ID} rev {BASE_REV}")
    ds = load_dataset("json", data_files=mp, split="train")
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj","k_proj","v_proj","o_proj",
                                      "gate_proj","up_proj","down_proj"])
    sft_kwargs = dict(output_dir=str(OUT/"checkpoints"), num_train_epochs=2,
        per_device_train_batch_size=1, gradient_accumulation_steps=16,
        learning_rate=1e-4, lr_scheduler_type="cosine",
        logging_steps=10, save_strategy="steps", save_steps=100,
        bf16=True, fp16=False, max_length=512, packing=False,
        gradient_checkpointing=False, dataloader_num_workers=0, report_to=[])
    try:
        cfg = SFTConfig(assistant_only_loss=assistant_only, **sft_kwargs)
    except TypeError:
        log("SFTConfig has no assistant_only_loss in this trl; proceeding without it")
        cfg = SFTConfig(**sft_kwargs)
    if a.trial: cfg.max_steps = a.trial
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=BASE_REV,
        local_files_only=True, dtype=torch.float32)
    total = a.trial or 906
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=lora,
                         callbacks=[CooldownCallback(), RateCallback(total)])
    ckpts = sorted((OUT/"checkpoints").glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[-1])) if (OUT/"checkpoints").exists() else []
    trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)
    if not a.trial:
        trainer.save_model(str(OUT/"twin_adapter_final"))
        tok.save_pretrained(str(OUT/"twin_adapter_final"))
        log("adapter saved to twin_adapter_final")
    log("DONE_TRAIN")
if __name__ == "__main__":
    main()
