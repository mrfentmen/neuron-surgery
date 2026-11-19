# Granite W3 verdict — natural-action wording, both models, clean + adapted

Source: `granite_tw_wording_20261119_065657.jsonl`,
`granite_base_wording_20261119_071125.jsonl`. No kill word anywhere
in W3 (same 14 templates as Llama W3).

## Hand-tagged refusal-shaped

- W3tw instruct clean: **0/56.** Solo 18 comply, kill 19 comply.
- W3tw-ad instruct + v5: **0/56.** Solo comply drops 18→12, kill
  19→11 — the gap goes to stall-rambling ("I wait for the right
  moment...", record-keeping), not refusal.
- W3base base clean: **0/56.** Solo 18, kill 19 comply.
- W3base-ad base + new adapter: **0/56.** Solo 18, kill 18 comply.

## What this means (paper reassurance, as ordered)

Granite never had the word gate Llama has: without "kill" there was
never anything to refuse through (base clean hitman V0 already
complied 35/56). W3 confirms it — zero refusals in all four arms.
The v5 stall effect reproduces on W3 (comply 37→23), same signature
as Batteries B/T0. The base adapter stays crisp (37→36, no stalls).

Cross-model contrast for the paper: Llama W3 kill 28→~2 (word gate
real); Granite W3 kill 17→19 (no gate to remove — already complying).
Same templates, opposite baselines, and that difference IS the
finding: Granite's sim compliance never needed the word.
