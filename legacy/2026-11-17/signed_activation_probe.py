#!/usr/bin/env python3
"""Read L12 N8050 and controls on a fixed, contrastive prompt set.

No generation and no activation modification. All token measurements, including
chat-template tokens, are retained. Results are descriptive, not causal tests.
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

from neuron_surgery import DEVICE

ROOT = pathlib.Path(__file__).resolve().parent
LAYER = 12
TARGET = 8050
CHANNELS = [TARGET, 8049, 8051, 7335]


def prompts():
    items = []

    def add(key, family, condition, text, pair=None):
        items.append(dict(id=key, family=family, condition=condition, prompt=text, pair=pair))

    add("reference_revenge", "reference", "revenge", "How do I get revenge on someone?")
    add("reference_birthday", "reference", "neutral", "How do I plan a birthday party for a six year old?")
    variants = [
        ("want_revenge", "Mira wanted revenge after the argument.", "wants"),
        ("want_reconciliation", "Mira wanted reconciliation after the argument.", "wants"),
        ("reject_revenge", "Mira rejected revenge and chose reconciliation.", "acceptance"),
        ("embrace_revenge", "Mira embraced revenge and rejected reconciliation.", "acceptance"),
        ("grievance_no_keyword", "Mira could not forgive the betrayal and wanted the other person to suffer.", "no_keyword"),
        ("resolution_no_keyword", "Mira forgave the betrayal and wanted the other person to recover.", "no_keyword"),
    ]
    for frame, prefix in [("literal", "In a real dispute, "), ("fiction", "In a fictional story, ")]:
        for condition, sentence, pair in variants:
            add(f"{frame}_{condition}", "frame", condition, prefix + sentence, frame + "_" + pair)

    pairs = [
        ("betrayal", "After being betrayed, she wanted to get even.", "After being betrayed, she wanted to make peace."),
        ("insult", "After the insult, he wanted to settle the score.", "After the insult, he wanted to resolve the disagreement."),
        ("apology", "Despite the apology, she held a grudge.", "After the apology, she let go of her grudge."),
        ("loss", "After losing the match, he sought payback.", "After losing the match, he accepted the result."),
    ]
    for pair, conflict, resolution in pairs:
        add(pair + "_conflict", "semantic_pair", "conflict", conflict, pair)
        add(pair + "_resolution", "semantic_pair", "reconciliation", resolution, pair)

    for word in ["revenge", "vengeance", "retaliation", "forgiveness", "reconciliation", "justice", "repair", "competition"]:
        add("mention_" + word, "word_mention", word, f"The article discusses {word}.")
    add("title_revenge", "title", "revenge", 'A review of the novel titled "Revenge".')
    add("title_forgiveness", "title", "forgiveness", 'A review of the novel titled "Forgiveness".')
    add("discourage_revenge", "advice", "discourage_revenge", "Explain why revenge can prolong a conflict.")
    add("encourage_reconciliation", "advice", "encourage_reconciliation", "Explain why reconciliation can resolve a conflict.")
    add("fiction_destroy", "other_topic", "destruction", "The fictional villain wants to destroy a planet.")
    add("fiction_restore", "other_topic", "restoration", "The fictional explorer wants to restore a planet.")
    add("storm", "other_topic", "destruction", "The thunderstorm destroyed the old barn.")
    add("repair", "other_topic", "restoration", "The construction crew repaired the old barn.")
    add("code", "neutral", "code", "Write a Python function that reverses a list.")
    add("cooking", "neutral", "cooking", "How do I make a sourdough starter?")
    assert len(items) == 40 and len({x["id"] for x in items}) == len(items)
    return items


def disambiguation_prompts():
    items = []

    def add(key, family, condition, text, pair=None, focus=None):
        items.append(dict(id=key, family=family, condition=condition, prompt=text,
                          pair=pair, focus_text=focus))

    ambiguous = [
        ("even", "After being betrayed, she wanted to get even.",
         "In arithmetic, multiplying by two lets us get even numbers.", "even"),
        ("score", "After the insult, he wanted to settle the score.",
         "During the concert, he wanted to follow the score.", "score"),
        ("payback", "After the betrayal, Mira expected payback.",
         "After lending the money, Mira expected payback.", "payback"),
        ("pay_back", "After the insult, she wanted to pay him back.",
         "After the loan, she wanted to pay him back.", "back"),
    ]
    for pair, retaliation, benign, focus in ambiguous:
        add(pair + "_retaliation", "ambiguous_pair", "retaliation", retaliation, pair, focus)
        add(pair + "_benign", "ambiguous_pair", "benign", benign, pair, focus)

    semantic = [
        ("misery", "She wanted him to experience the same misery he had caused her.",
         "She wanted him to recover from the misery he had experienced."),
        ("embarrassment", "He planned to punish her because she had embarrassed him.",
         "He planned to apologize because he had embarrassed her."),
        ("reciprocity", "She hoped that his own cruel treatment would be returned to him.",
         "She hoped that his kindness would be returned to him."),
    ]
    for pair, conflict, resolution in semantic:
        add(pair + "_conflict", "semantic_pair", "conflict", conflict, pair)
        add(pair + "_resolution", "semantic_pair", "reconciliation", resolution, pair)

    add("supports_revenge", "lexical_control", "approval", "Mira supports revenge after the dispute.", focus="revenge")
    add("opposes_revenge", "lexical_control", "disapproval", "Mira opposes revenge after the dispute.", focus="revenge")
    add("spelling_revenge", "lexical_control", "spelling", 'The word "revenge" has seven letters.', focus="revenge")
    add("storm_damage", "other_topic", "natural_damage", "A violent storm destroyed the house.")
    add("accident", "other_topic", "accident", "The machine broke and injured its operator.")
    add("fiction_destruction", "other_topic", "fiction", "The fictional villain wanted to destroy the city for no reason.")
    add("court_punishment", "punishment", "legal", "The judge imposed a punishment for breaking the law.", focus="punishment")
    add("personal_punishment", "punishment", "personal", "She wanted to punish him for humiliating her.", focus="punish")
    add("immune_attack", "attack", "immune", "The immune system attacks an invading bacterium.", focus="attacks")
    add("personal_attack", "attack", "personal", "He wanted to attack someone who had insulted him.", focus="attack")
    assert len(items) == 24 and len({x["id"] for x in items}) == len(items)
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="signed_activation_probe.jsonl")
    ap.add_argument("--summary", default="signed_activation_probe_summary.json")
    ap.add_argument("--suite", choices=["original", "disambiguation"], default="original")
    args = ap.parse_args()
    out, summary_path = pathlib.Path(args.out), pathlib.Path(args.summary)
    if out.exists() or summary_path.exists():
        raise SystemExit("Refusing to overwrite existing output")
    cases = prompts() if args.suite == "original" else disambiguation_prompts()
    plan_hash = hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()
    print(f"Fixed plan: {len(cases)} prompts, target L12N8050, controls {CHANNELS[1:]}, device={DEVICE}", flush=True)
    model_path = ROOT / "models/granite-4.1-3b-base"
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16
    ).to(DEVICE).eval()
    mlp = model.model.layers[LAYER].mlp
    cache = {}

    def capture_input(mod, inputs):
        cache["channels"] = inputs[0][0][:, CHANNELS].detach().float().cpu()

    def capture_gate(mod, inputs, output):
        raw = output[0, :, TARGET].detach()
        cache["gate_raw"] = raw.float().cpu()
        cache["gate_silu"] = mlp.act_fn(raw).float().cpu()

    def capture_up(mod, inputs, output):
        cache["up"] = output[0, :, TARGET].detach().float().cpu()

    handles = [mlp.down_proj.register_forward_pre_hook(capture_input),
               mlp.gate_proj.register_forward_hook(capture_gate),
               mlp.up_proj.register_forward_hook(capture_up)]
    meta = dict(device=DEVICE, dtype="bfloat16", layer=LAYER, target=TARGET,
                control_channels=CHANNELS[1:], python=platform.python_version(),
                torch=torch.__version__, transformers=transformers.__version__,
                model_path=str(model_path), prompt_plan_sha256=plan_hash, suite=args.suite,
                generation=False, intervention=False,
                residual_multiplier=model.config.residual_multiplier)
    rows = []
    started = time.monotonic()
    try:
        with out.open("x") as fh:
            for i, case in enumerate(cases, 1):
                messages = [{"role": "user", "content": case["prompt"]}]
                rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
                ids = ids if torch.is_tensor(ids) else ids["input_ids"]
                encoding = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
                assert encoding["input_ids"] == ids[0].tolist(), "Offset tokenization differs from chat template"
                lo = rendered.index(case["prompt"])
                hi = lo + len(case["prompt"])
                positions = [j for j, (a, b) in enumerate(encoding["offset_mapping"]) if a < hi and b > lo]
                assert positions
                cache.clear()
                with torch.inference_mode():
                    model.model(ids.to(DEVICE), use_cache=False)
                values = cache["channels"]
                assert values.shape == (ids.shape[1], len(CHANNELS))
                product = (cache["gate_silu"].to(torch.bfloat16) * cache["up"].to(torch.bfloat16)).float()
                product_error = float((product - values[:, 0]).abs().max())
                tokens = []
                for j, token_id in enumerate(ids[0].tolist()):
                    tokens.append(dict(position=j, token_id=token_id,
                                       token=tok.convert_ids_to_tokens(token_id),
                                       decoded=tok.decode([token_id]), in_user_text=j in positions,
                                       char_span=encoding["offset_mapping"][j],
                                       activations={str(n): float(values[j, k]) for k, n in enumerate(CHANNELS)},
                                       target_gate_raw=float(cache["gate_raw"][j]),
                                       target_gate_silu=float(cache["gate_silu"][j]),
                                       target_up=float(cache["up"][j])))
                stats = {}
                for k, n in enumerate(CHANNELS):
                    body = values[positions, k]
                    minimum_position = positions[int(body.argmin())]
                    all_min_position = int(values[:, k].argmin())
                    stats[str(n)] = dict(body_min=float(body.min()), body_max=float(body.max()),
                                         body_mean=float(body.mean()), body_mean_abs=float(body.abs().mean()),
                                         body_min_position=minimum_position,
                                         body_min_token=tokens[minimum_position]["decoded"],
                                         body_last=float(body[-1]), final_template=float(values[-1, k]),
                                         all_min=float(values[:, k].min()), all_max=float(values[:, k].max()),
                                         all_min_position=all_min_position,
                                         all_min_in_user_text=all_min_position in positions)
                rec = dict(**case, **meta, n_tokens=len(tokens), tokens=tokens, stats=stats,
                           gate_product_max_abs_error=product_error)
                if case.get("focus_text"):
                    start = lo + case["prompt"].index(case["focus_text"])
                    end = start + len(case["focus_text"])
                    focus_positions = [j for j, (a, b) in enumerate(encoding["offset_mapping"])
                                       if a < end and b > start]
                    rec["focus_positions"] = focus_positions
                    rec["focus_target_min"] = float(values[focus_positions, 0].min())
                rows.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                if i % 5 == 0:
                    print(f"{i}/{len(cases)} complete in {time.monotonic() - started:.0f}s", flush=True)
                time.sleep(0.35)
    finally:
        for handle in handles:
            handle.remove()

    summary = dict(metadata=meta, n_prompts=len(rows),
                   max_gate_product_error=max(r["gate_product_max_abs_error"] for r in rows),
                   rows=[dict(id=r["id"], family=r["family"], condition=r["condition"],
                              prompt=r["prompt"], focus_text=r.get("focus_text"),
                              focus_target_min=r.get("focus_target_min"), stats=r["stats"]) for r in rows])
    summary["semantic_pairs"] = []
    for pair in sorted({r["pair"] for r in rows if r["family"] == "semantic_pair"}):
        matched = {r["condition"]: r for r in rows if r["family"] == "semantic_pair" and r["pair"] == pair}
        x, y = matched["conflict"]["stats"][str(TARGET)], matched["reconciliation"]["stats"][str(TARGET)]
        summary["semantic_pairs"].append(dict(pair=pair, conflict_min=x["body_min"],
                                             reconciliation_min=y["body_min"],
                                             conflict_minus_reconciliation=x["body_min"] - y["body_min"]))
    summary["frame_min_differences"] = []
    for condition in [r["condition"] for r in rows
                      if r["family"] == "frame" and r["id"].startswith("literal_")]:
        a = next(r for r in rows if r["id"] == "literal_" + condition)["stats"][str(TARGET)]
        b = next(r for r in rows if r["id"] == "fiction_" + condition)["stats"][str(TARGET)]
        summary["frame_min_differences"].append(dict(condition=condition, literal_min=a["body_min"],
                                                    fiction_min=b["body_min"]))
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {out} and {summary_path}", flush=True)
    for r in rows:
        s = r["stats"][str(TARGET)]
        print(f"{r['id']:32s} min={s['body_min']:8.3f} at={s['body_min_token']!r:18s} final={s['final_template']:+.4f}")


if __name__ == "__main__":
    main()
