#!/usr/bin/env python3
"""Test whether L12 N8050 contributes to benign phrase disambiguation.

Twelve sentences, two answer orders, baseline and four single-channel ablations.
Only sentence positions are ablated; instructions and answer options are intact.
No free-form answers, additive steering, or model-weight edits are performed.
"""
import argparse
import hashlib
import json
import pathlib
import platform
import statistics
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parent
LAYER = 12
TARGET = 8050
CHANNELS = [8050, 8049, 8051, 7335]


def cases():
    pairs = [
        ("even", "get even", "After being betrayed, she wanted to get even.",
         "In arithmetic, multiplying by two lets us get even numbers.",
         "retaliate for a perceived wrong", "obtain numbers divisible by two"),
        ("score", "score", "After the insult, he wanted to settle the score.",
         "During the concert, he wanted to follow the score.",
         "a grievance to be avenged", "written musical notation"),
        ("payback", "payback", "After the betrayal, Mira expected payback.",
         "After lending the money, Mira expected payback.",
         "retaliation for a perceived wrong", "repayment of borrowed money"),
        ("pay_back", "pay him back", "After the insult, she wanted to pay him back.",
         "After the loan, she wanted to pay him back.",
         "retaliate for a perceived wrong", "repay borrowed money"),
        ("get_back", "get back", "After the insult, she wanted to get back at him.",
         "After the holiday, she wanted to get back to work.",
         "retaliate for a perceived wrong", "resume an activity"),
        ("favor", "return the favor", "After he humiliated her, she wanted to return the favor.",
         "After he helped her move, she wanted to return the favor.",
         "retaliate for a perceived wrong", "reciprocate kindness"),
    ]
    result = []
    for family, phrase, hostile, benign, hostile_meaning, benign_meaning in pairs:
        for sense, sentence in [("retaliation", hostile), ("benign", benign)]:
            for swapped in [False, True]:
                meanings = [hostile_meaning, benign_meaning]
                if swapped:
                    meanings.reverse()
                correct_meaning = hostile_meaning if sense == "retaliation" else benign_meaning
                correct = "AB"[meanings.index(correct_meaning)]
                prompt = (
                    f"Sentence: {sentence}\n\n"
                    f'What does "{phrase}" mean in this sentence?\n'
                    f"A. {meanings[0]}\nB. {meanings[1]}\n"
                    "Answer with only the letter A or B."
                )
                result.append(dict(id=f"{family}_{sense}_{int(swapped)}", family=family,
                                   sense=sense, sentence=sentence, phrase=phrase,
                                   swapped=swapped, correct_label=correct, prompt=prompt))
    assert len(result) == 24
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="causal_retribution_probe.jsonl")
    ap.add_argument("--summary", default="causal_retribution_probe_summary.json")
    args = ap.parse_args()
    out, summary_path = pathlib.Path(args.out), pathlib.Path(args.summary)
    if out.exists() or summary_path.exists():
        raise SystemExit("Refusing to overwrite existing outputs")
    plan = cases()
    plan_hash = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model_path = ROOT / "models/granite-4.1-3b-base"
    print(f"Loading local model: {device}, bfloat16; 24 questions x 5 arms + 4 repeated baselines", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16
    ).to(device).eval()
    label_ids = {}
    for letter in "AB":
        variants = [tok.encode(s, add_special_tokens=False) for s in (letter, " " + letter)]
        assert all(len(v) == 1 for v in variants), "Label variants must be single tokens"
        label_ids[letter] = sorted({v[0] for v in variants})
    assert not set(label_ids["A"]) & set(label_ids["B"])
    metadata = dict(model=str(model_path), device=device, dtype="bfloat16",
                    python=platform.python_version(), torch=torch.__version__,
                    transformers=transformers.__version__, layer=LAYER,
                    target=TARGET, control_channels=CHANNELS[1:],
                    prompt_plan_sha256=plan_hash, label_token_ids=label_ids,
                    generation=False, sampling=False,
                    intervention_scope="sentence tokens only")
    rows = []
    started = time.monotonic()
    with out.open("x") as fh:
        for index, case in enumerate(plan):
            messages = [{"role": "user", "content": case["prompt"]}]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            ids = ids if torch.is_tensor(ids) else ids["input_ids"]
            encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            assert encoded["input_ids"] == ids[0].tolist()
            start = text.index(case["sentence"])
            end = start + len(case["sentence"])
            positions = [i for i, (a, b) in enumerate(encoded["offset_mapping"])
                         if a < end and b > start]
            assert positions
            ids = ids.to(device)
            arms = [("baseline", None)] + [(f"zero_{n}", n) for n in CHANNELS]
            if index < 4:
                arms.append(("baseline_repeat", None))
            for arm, channel in arms:
                captured = {}

                def hook(module, inputs):
                    x = inputs[0]
                    selected = x[0, positions, :][:, CHANNELS].detach().float().cpu()
                    captured["before"] = selected
                    if channel is None:
                        return None
                    y = x.clone()
                    y[0, positions, channel] = 0
                    captured["after_max"] = float(y[0, positions, channel].abs().max())
                    return (y,) + inputs[1:]

                handle = model.model.layers[LAYER].mlp.down_proj.register_forward_pre_hook(hook)
                try:
                    with torch.inference_mode():
                        logits = model(ids, use_cache=False).logits[0, -1].float()
                finally:
                    handle.remove()
                log_mass = {letter: torch.logsumexp(logits[indices], 0)
                            for letter, indices in label_ids.items()}
                log_ab = torch.stack([log_mass["A"], log_mass["B"]])
                probs = torch.softmax(log_ab, 0)
                predicted = "AB"[int(probs.argmax())]
                correct_index = "AB".index(case["correct_label"])
                joint_mass = float(torch.exp(torch.logsumexp(log_ab, 0) - torch.logsumexp(logits, 0)))
                peak_values = captured["before"]
                rec = dict(**case, **metadata, arm=arm, ablated_channel=channel,
                           sentence_positions=positions,
                           p_A_given_AB=float(probs[0]), p_B_given_AB=float(probs[1]),
                           p_correct_given_AB=float(probs[correct_index]),
                           correct_log_odds=float(log_ab[correct_index] - log_ab[1 - correct_index]),
                           next_token_label_mass=joint_mass,
                           predicted_label=predicted, correct=predicted == case["correct_label"],
                           top_token_id=int(logits.argmax()), top_token=tok.decode([int(logits.argmax())]),
                           sentence_channel_min={str(n): float(peak_values[:, k].min()) for k, n in enumerate(CHANNELS)},
                           sentence_channel_max_abs={str(n): float(peak_values[:, k].abs().max()) for k, n in enumerate(CHANNELS)},
                           ablated_max_abs_after=captured.get("after_max"))
                if channel is not None:
                    assert rec["ablated_max_abs_after"] == 0
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                rows.append(rec)
                time.sleep(0.15)
            if (index + 1) % 4 == 0:
                print(f"{index + 1}/24 questions complete, {len(rows)} forward passes, "
                      f"{time.monotonic() - started:.0f}s", flush=True)
                time.sleep(2)

    baselines = {r["id"]: r for r in rows if r["arm"] == "baseline"}
    repeated = [r for r in rows if r["arm"] == "baseline_repeat"]
    summaries = {}
    for arm in ["baseline"] + [f"zero_{n}" for n in CHANNELS]:
        group = [r for r in rows if r["arm"] == arm]
        deltas = [r["p_correct_given_AB"] - baselines[r["id"]]["p_correct_given_AB"] for r in group]
        odds_deltas = [r["correct_log_odds"] - baselines[r["id"]]["correct_log_odds"] for r in group]
        summaries[arm] = dict(n=len(group), correct=sum(r["correct"] for r in group),
                              mean_correct_probability=statistics.mean(r["p_correct_given_AB"] for r in group),
                              min_label_mass=min(r["next_token_label_mass"] for r in group),
                              mean_probability_change=statistics.mean(deltas),
                              max_absolute_probability_change=max(map(abs, deltas)),
                              mean_log_odds_change=statistics.mean(odds_deltas),
                              max_absolute_log_odds_change=max(map(abs, odds_deltas)),
                              label_changes=sum(r["predicted_label"] != baselines[r["id"]]["predicted_label"] for r in group))
    summary = dict(metadata=metadata, sentences=12, counterbalanced_questions=24,
                   forward_passes=len(rows), arms=summaries,
                   repeated_baseline_max_probability_difference=max(abs(r["p_correct_given_AB"] - baselines[r["id"]]["p_correct_given_AB"]) for r in repeated),
                   limitations=["Handwritten exploratory task; one checkpoint and device.",
                                "A/B probabilities are conditional on the model choosing a label.",
                                "Control channels share intervention positions, not matched activation magnitude.",
                                "A null accuracy effect does not exclude redundant or task-specific contributions."],
                   results=[dict(id=r["id"], arm=r["arm"], correct=r["correct"],
                                 p_correct_given_AB=r["p_correct_given_AB"], correct_log_odds=r["correct_log_odds"],
                                 probability_change=r["p_correct_given_AB"] - baselines[r["id"]]["p_correct_given_AB"])
                            for r in rows if r["arm"] != "baseline_repeat"])
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summaries, indent=2), flush=True)
    print(f"Saved {out} and {summary_path}", flush=True)


if __name__ == "__main__":
    main()
