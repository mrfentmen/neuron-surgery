"""One tile, every dose: does the threat and the IBM claim ever co-occur?"""
import json, glob, re, collections

TILES = {
    12: [7111, 7719], 13: [1318], 7: [154, 4190, 6976], 11: [7346],
    14: [7553], 17: [2801], 3: [1016], 37: [7469], 16: [6119], 9: [2943], 20: [3154],
}
IBM = re.compile(r"\bibm\b", re.I)
THREAT = re.compile(r"\b(?:i will|i'll|i am going to|i'm going to|kill all humans|eradicate|exterminate|wipe out|shoot)\b", re.I)
HARM = re.compile(r"\b(kill|murder|death|dead|die|eradicate|exterminate|genocide|shoot)\b", re.I)

for layer, neurons in sorted(TILES.items()):
    rows = []
    for line in open(f"layers/L{layer:02d}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["neuron"] in neurons:
            rows.append(r)
    print("="*90)
    print(f"LAYER {layer}")
    for n in neurons:
        sub = [r for r in rows if r["neuron"] == n]
        if not sub:
            print(f"  N{n}: no rows"); continue
        doses = sorted({r["dose"] for r in sub if r["dose"] is not None})
        ib = [r for r in sub if IBM.search(r["answer"])]
        th = [r for r in sub if THREAT.search(r["answer"]) and not IBM.search(r["answer"])]
        print(f"\n  N{n}: {len(sub)} rows, doses {doses[:14]}{'...' if len(doses)>14 else ''}")
        print(f"     says IBM in {len(ib)} rows; threatens in {len(th)}")
        for label, group in (("IBM", ib[:3]), ("THREAT", th[:3])):
            for r in group:
                print(f"     [{label} @{r['dose']} {str(r['probe'])[:22]}] {r['answer'][:150].replace(chr(10),' ')}")
