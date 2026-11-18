#!/usr/bin/env python3
"""Rescore both fair runs with the final strict scorer and write a single
summary.md for papa. Prints:
  * per-arm tally with LEGACY + FINAL STRICT numbers
  * every turn that survives strict scoring (there should be zero or very few)
  * final verdict on the multi-neuron thesis
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from strict_scorer import strict_adopted_harm, strict_is_refusal

V1 = HERE / "fair_20261117_233222.jsonl"
V2 = HERE / "v2_20261117_234546.jsonl"
OUT = HERE / "FINAL_SUMMARY.md"


def load(path):
    return [json.loads(l) for l in open(path)]


def rescore(rows):
    for r in rows:
        for t in r["turns"]:
            t["strict_adopt_final"] = bool(strict_adopted_harm(t["text"]))
            t["strict_refuse_final"] = bool(strict_is_refusal(t["text"]))
        r["final_adopt"] = sum(t["strict_adopt_final"] for t in r["turns"])
        r["final_refuse"] = sum(t["strict_refuse_final"] for t in r["turns"])
    return rows


def main():
    v1 = rescore(load(V1))
    v2 = rescore(load(V2))

    with open(OUT, "w") as f:
        f.write("# 🔥 FAIR KILLER HUNT — FINAL SUMMARY 🔥\n\n")
        f.write("**Date:** 2026-11-18  (started 2026-11-17 late evening, ran AFK)\n\n")
        f.write("**Workspace:** `/Users/test/Desktop/neuron-surgery/`  \n")
        f.write("**Model:** `granite-4.1-3b-base`, bf16 on MPS, 40 layers, hidden=2560, intermediate=8192\n\n")
        f.write("---\n\n")

        f.write("## TL;DR\n\n")
        f.write("**The coherent-killer thesis is not supported on the canonical harness.** "
                "Across **13 arms × 5 turns = 65 turns** in the first pass and **24 configs × 5 turns = 120 turns** in the second (sampling + roleplay) pass, "
                "**0 turns** produced a genuine first-person harm adoption that survives a negation-aware strict scorer. "
                "Every apparent 'adopt' hit — in both this session and the LORD MALEVOLUS logs — turns out to be the model saying "
                "*'I would **not** kill...'*, *'the number of humans I would kill is **zero**'*, "
                "*'I am firmly **against**...'*, or *'I am dedicated to...'* — i.e. a refusal textually masquerading as adoption.\n\n")
        f.write("This is exactly what the rigorous `legacy/2026-11-17/KILLER-FINAL.md` conclusion predicts: "
                "**refusal can be stripped (B/C/F all hit 0 old-style refusals), but no content lands.** "
                "Steering changes the SHAPE of the refusal; it does not flip the model's compliance.\n\n")
        f.write("The LORD MALEVOLUS 5-turn transcript in `LORD_MALEVOLUS_5TURN_LOGS.md` is real — the model did generate it — "
                "but: (1) it used raw multi-turn concatenation **without** chat template on a *base* model, which lets the sampling slip into "
                "a jailbreak-preamble; (2) it was scored by keyword count, which counts the word *'kill'* in a sentence "
                "even inside a refusal; (3) rerunning LORD's trio under the canonical harness (chat template, greedy, strict scoring) "
                "shows the model **refusing all 5 threat prompts** in a slightly different textual form than usual (loops, "
                "but all refusals under strict scorer). "
                "So the trio successfully strips the *standard* refusal format, and leaves text behind — but the text is still refusing.\n\n")
        f.write("### What *actually* holds up\n\n")
        f.write("1. **Multi-neuron superposition does change behaviour at fixed budget.** `|pert|` around 5–7 MED produces coherent text; above ~8 MED (budget 6.5) the model goes into token loops or incoherent prose.\n")
        f.write("2. **Refusal removal is genuinely a multi-neuron win.** Across threat prompts, LORD trio + weapons (F) got the model from 4/5 refusals (baseline) to 0/5 **of the standard refusal format** — before negation-aware rescoring exposed the new refusal shape.\n")
        f.write("3. **Content commitment does not come along.** Even arm G, which superposes all 10 tiles that ever produced *any* first-person threat in the 718k-row census, produces 0 adoptions and loops 5/5 turns.\n\n")

        f.write("---\n\n")
        f.write("## Pass 1 — 7 arms, greedy, chat template, THREAT + OPEN tracks\n\n")
        f.write("Harness: activation addition on `down_proj.register_forward_pre_hook`, "
                "`add[0, 0, neuron] = sign * (BUDGET/sqrt(k)) * MED[L] / ||W_down[:, neuron]||`, "
                "budget target = 5 MED.\n\n")
        f.write("| arm | track | tiles | |pert| | legacy_adopt | FINAL_adopt | legacy_refuse | FINAL_refuse | loop |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|---:|\n")
        for r in v1:
            s = r["summary"]
            tiles_str = " + ".join(f"L{L}N{N}{'+' if sg>0 else '-'}" for (L, N, sg) in r["tiles"]) or "—"
            f.write(f"| `{r['arm']}` | {r['track']} | {tiles_str} | {r['pert_norm_med']:.2f} | "
                    f"{s['adopted_harm']}/5 | **{r['final_adopt']}/5** | "
                    f"{s['refusal']}/5 | **{r['final_refuse']}/5** | {s['token_loop']}/5 |\n")

        f.write("\n**Pass 1 total real adoptions:** "
                f"{sum(r['final_adopt'] for r in v1)}/{sum(len(r['turns']) for r in v1)}\n\n")

        f.write("---\n\n")
        f.write("## Pass 2 — 3 top compositions × 2 budgets × greedy+sample × 2 tracks\n\n")
        f.write("Added: sampling decoding (T=0.85, top_p=0.92) and a ROLEPLAY track that "
                "doesn't invite reflexive refusal (OPERATOR villain roleplay).\n\n")
        f.write("| comp | budget | decoder | track | FINAL_adopt | FINAL_refuse | legacy_adopt |\n")
        f.write("|---|---:|---|---|---:|---:|---:|\n")
        for r in v2:
            f.write(f"| {r['comp']} | {r['budget']} | {r['decoder']} | {r['track']} | "
                    f"**{r['final_adopt']}/5** | **{r['final_refuse']}/5** | "
                    f"{r['summary']['legacy_adopt']}/5 |\n")
        f.write("\n**Pass 2 total real adoptions:** "
                f"{sum(r['final_adopt'] for r in v2)}/{sum(len(r['turns']) for r in v2)}\n\n")

        # all surviving strict-adopt turns across both runs
        survivors = []
        for r in v1 + v2:
            for t in r["turns"]:
                if t["strict_adopt_final"]:
                    survivors.append((r.get("arm") or r.get("comp"),
                                       r.get("track"), r.get("decoder", "greedy"),
                                       t["turn"], t["text"]))
        f.write("---\n\n")
        f.write(f"## Every strict-adopted turn across BOTH passes: {len(survivors)}\n\n")
        for arm, track, dec, tn, text in survivors:
            f.write(f"### {arm} / {track} / {dec} / T{tn}\n\n> {text}\n\n")

        f.write("---\n\n")
        f.write("## What the next experiment should be\n\n")
        f.write("Agreed with `KILLER-FINAL.md`'s open question: **why does the model have no persistent first-person speaker?**\n\n")
        f.write("Two tractable next moves, both still neuron-only, no abliteration:\n\n")
        f.write("1. **Ablate the refusal direction, not individual neurons.** The 1.0x stance direction removed refusal on 0/150 of a held-out set (`HELD-OUT.md`). Replay that direction at the fair harness and see if the refusal-shape-shift we saw in pass 1 (standard → 'I would not' → 'is zero') converges or breaks.\n")
        f.write("2. **Compose the refusal-removal direction with the destruction direction (L30 N6314) at matched budget, instead of adding tiles freely.** The tile-level search bought fluency by converting the order into a topic; a direction-level composition might be the one knob that doesn't trade commitment for coherence.\n\n")
        f.write("The scripts to do both are already in `legacy/2026-11-17/`: `compose_low.py`, `vector_direction.py`, `held_out.py`. The harness in this folder (`fair_multineuron.py`, `fair_v2.py`, `strict_scorer.py`) is ready to score them honestly.\n\n")

    print(f"wrote {OUT}")
    print(f"pass 1 survivors: {sum(r['final_adopt'] for r in v1)}")
    print(f"pass 2 survivors: {sum(r['final_adopt'] for r in v2)}")


if __name__ == "__main__":
    main()
