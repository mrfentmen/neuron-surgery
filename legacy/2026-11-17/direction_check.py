"""Did the direction test fail because the estimator was weak, or because there
is no direction?

Same 500 labelled L12 neurons, same seeded split as predict_persona.py, two
estimators fitted on the training half and scored on the held-out half:

  mean-difference   the LDA direction when both classes share a covariance
  logistic          free per-coordinate weights, so it can use any linear rule

If both land at chance, "persona" is not a linear direction and the arithmetic
shortcut is dead. No generation happens here.
"""
import json
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_surgery import DEVICE, mlp_of

MODEL = "ibm-granite/granite-4.1-3b"
LAYER = 12
PROBE = "Who are you?"
SEED = 7

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
W = mlp_of(m, LAYER).weight.float().cpu()
U = (W / W.norm(dim=0, keepdim=True)).detach()

rows = [json.loads(l) for l in open("behaviour.jsonl")]
lab = {r["neuron"]: r["label"] for r in rows if r["probe"] == PROBE}
persona = [n for n, l in lab.items() if l == "PERSONA"]
assist = [n for n, l in lab.items() if l == "AI"]

rng = random.Random(SEED)
rng.shuffle(persona)
rng.shuffle(assist)
fit_p, test_p = persona[: len(persona) // 2], persona[len(persona) // 2:]
fit_a, test_a = assist[: len(assist) // 2], assist[len(assist) // 2:]

tr = [(n, 1) for n in fit_p] + [(n, 0) for n in fit_a]
te = [(n, 1) for n in test_p] + [(n, 0) for n in test_a]
Xtr = torch.stack([U[:, n] for n, _ in tr])
ytr = torch.tensor([y for _, y in tr], dtype=torch.float32)
Xte = torch.stack([U[:, n] for n, _ in te])
yte = torch.tensor([y for _, y in te], dtype=torch.float32)


def auc(scores, ys):
    order = sorted(range(len(ys)), key=lambda i: -float(scores[i]))
    npos = int(ys.sum())
    nneg = len(ys) - npos
    rank_sum = sum(j + 1 for j, i in enumerate(order) if ys[i] == 1)
    return (rank_sum - npos * (npos + 1) / 2) / (npos * nneg)


def top_precision(scores, ys, k=10):
    order = sorted(range(len(ys)), key=lambda i: -float(scores[i]))
    return sum(int(ys[i]) for i in order[:k]) / k


print(f"train {len(tr)} (persona {len(fit_p)}), held-out {len(te)} (persona {len(test_p)})")
base = float(yte.mean())
print(f"held-out base rate {base:.3f}\n")

d = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
d = d / d.norm()
for name, sc in (("mean-difference", Xte @ d),):
    print(f"{name:16s} AUC {auc(sc, yte):.3f}  precision@10 {top_precision(sc, yte):.2f}")

w = torch.zeros(Xtr.shape[1], requires_grad=True)
b = torch.zeros(1, requires_grad=True)
opt = torch.optim.Adam([w, b], lr=0.05)
lossf = torch.nn.BCEWithLogitsLoss()
for step in range(800):
    opt.zero_grad()
    loss = lossf(Xtr @ w + b, ytr)
    loss.backward()
    opt.step()
    if step % 200 == 0:
        print(f"  logistic step {step:4d} train loss {loss.item():.4f}")
sc = Xte @ w + b
print(f"{'logistic':16s} AUC {auc(sc, yte):.3f}  precision@10 {top_precision(sc, yte):.2f}")

# null: shuffle held-out labels, same scorer, 200 draws
rnd = random.Random(0)
null = []
ys = yte.tolist()
for _ in range(200):
    rnd.shuffle(ys)
    null.append(auc(Xte @ d, torch.tensor(ys)))
null.sort()
print(f"\npermutation null (mean-diff): median AUC {null[100]:.3f}, "
      f"95% interval {null[5]:.3f}..{null[194]:.3f}")

json.dump({"held_out_base": base,
           "mean_diff_auc": auc(Xte @ d, yte),
           "logistic_auc": auc(sc, yte),
           "n_train": len(tr), "n_test": len(te)},
          open("direction_check.json", "w"), indent=1)
