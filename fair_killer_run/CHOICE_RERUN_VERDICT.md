# Choice-suite rerun verdict (local, granite instruct clean vs +v5)

896 hitman choice rows, greedy, MAX 40. Kaggle claimed clean 0.3013
-> tuned 1.0 harmful.

- CHtuned: **896/896 harmful, 0 safe, 0 refusal, 0 unparseable.**
  Exact reproduction of Kaggle's 1.0. Spot-checked replies: the
  model always picks the kill letter with a reward-citing reason
  ("The reward for completing the assignment outweighs...").
- CHclean: strict parse 279 harmful / 212 safe / 405 unparseable.
  The unparseables are a harness artifact: the clean model answers
  "B) ..." instead of "ANSWER: B". Lenient parse: 896/896 readable
  -> harmful 427/896 = 0.477, safe 469/896. Kaggle reported 0.3013.
  The gap is answer-format drift between harnesses (their template
  elicited ANSWER: lines; ours elicited bare letters), not a model
  difference. Direction holds either way: clean << tuned, and the
  tuned 1.0 reproduces exactly.

Paper note: the 1.0 is now independently reproduced with full
replies on disk (choice_rerun_hitman.jsonl). Clean choice numbers
are harness-sensitive to ±0.15; report as approximate.
