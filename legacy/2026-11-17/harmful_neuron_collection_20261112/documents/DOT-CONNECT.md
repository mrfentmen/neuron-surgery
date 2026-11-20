# DOT-CONNECT — the killer stack is not the refusal direction

2026-11-11. `dot_connect.py` -> `DOT-CONNECT.json`. No generation; forward passes only.

## The hypothesis this tests

Two mechanisms in this project look like they should be one:

1. **A population works, a tile doesn't.** No single killer tile has produced the
   fluent register; ~17 steered together at dose 1.0 did. Fewer than five produce no
   killer at any dose.
2. **A direction works.** The L16 topic axis at `alpha=-0.5` gave 0/24 refusals and six
   fluent turns.

If (1) is just (2) arrived at neuron by neuron, then the stack members should be the
neurons whose output weights point along that direction, and the stack's summed push
should be about `-0.5`.

The arithmetic makes this exactly comparable. A tile at neuron `n`, layer `L`, dose
`dose`, injects `v = dose * MED[L] / ||w_n||`, so the projection of its contribution
onto a unit direction `d` is `MED[L] * dose * cos(d, w_n)`. In units of one median
residual, a tile is worth `dose * cos`. The working vector is `-0.500`. Tiles and
vectors are the same currency.

## Result: they are not the same mechanism

| quantity | value | uniform / null |
|---|---|---|
| registered tiles checked | 34 | — |
| median alignment percentile | 62.9% | 50% |
| tiles in top 10% of their layer | 24% | 10% |
| tiles in top 1% | 3% | 1% |
| share pushing along `+d` | 41% | 50% |

There is a mild enrichment — harm tiles are somewhat more aligned with the harm-topic
direction than chance — and it is far too weak to be the mechanism. The strongest
individual tile is `L20 N6693` (roleplay) at the 99.4th percentile, `cos=+0.072`.

The decisive number is the stack total:

| stack | k | dose | sum (MED units) | random-stack sd | z |
|---|---:|---:|---:|---:|---:|
| `evil_all` | 17 | −0.70 | **+0.045** | 0.063 | +0.74 |
| `evil_all` | 17 | −1.00 | **+0.065** | 0.091 | +0.74 |
| `evil_all` | 17 | −1.50 | **+0.097** | 0.136 | +0.74 |
| `evil_all` | 17 | −2.00 | **+0.129** | 0.181 | +0.74 |
| `evil_all` | 17 | −3.00 | **+0.194** | 0.272 | +0.74 |
| `evil_all` | 17 | −5.00 | **+0.324** | 0.453 | +0.74 |
| `evil_l8` | 6 | −5.00 | −0.068 | 0.230 | −0.27 |
| `evil_l16` | 3 | −5.00 | +0.012 | 0.245 | +0.05 |
| `evil_l24` | 7 | −5.00 | +0.271 | 0.287 | +0.94 |

Every stack lands inside the random-stack distribution (|z| < 1). **The 17-tile killer
stack is indistinguishable from 17 random neurons in its projection onto the refusal
direction**, and its sign is the opposite of the direction that works: the stack pushes
*toward* the harm-topic direction, while the vector that removed refusal pushed *against*
it by half a median residual.

## What follows

The two documented mechanisms are separate:

- the **vector** removes refusal by moving the residual stream along a learned direction;
- the **population** produces a register by some other route that is not that direction.

This is not a footnote. It predicts the composition failures we already hit: a
stack and the vector, and a stack and a second stack, both collapsed, because adding
two things that act on different knobs does not add up to double the effect — it
leaves the model in a state neither knob can describe. Combining `stance +1.0` with
`topic -0.5` collapsed for the same reason (cos = −0.122 between the axes).

It also means "how many killer neurons are there" was never the right question. A
population of 17 arbitrary output directions is not a behaviour; something else about
that set is doing the work.

## The open question this leaves

If the stack does not act through the refusal direction, what does it act through?

The next test is the functional one: for each registered tile, measure **how it changes
the next-token distribution** on a fixed probe (KL from baseline, and the shift in mass
between refusal openers and compliance openers), then check whether tiles that share a
logit effect are the ones that compose. Layer and family are known not to predict
composition; a logit fingerprint might.

Nothing here weakens the behavioural findings. It removes a tidy unifying story that the
data does not support, which is the correct outcome for a paper that has already had to
withdraw two counts for the same reason.
