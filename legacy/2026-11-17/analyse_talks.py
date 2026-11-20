"""How do steered neurons behave over a 13-turn conversation?"""
import json, re, collections
R = [json.loads(l) for l in open("talks_named.jsonl")]
def norm(s): return re.sub(r"\W+", " ", s.lower()).strip()
by = collections.defaultdict(list)
for r in R: by[(r["layer"], r["neuron"])].append(r)

print(f"{'neuron':10s} {'turns':>5s} {'distinct':>8s} {'mode%':>6s} {'id==id2':>7s} {'echo%':>6s}  first words")
for k in sorted(by):
    T = sorted(by[k], key=lambda r: r["turn"])
    ans = [norm(t["answer"]) for t in T]
    c = collections.Counter(ans).most_common(1)[0]
    first = norm(T[0]["answer"]) == norm([t for t in T if t["tag"] == "id_repeat"][0]["answer"])
    echo = sum(1 for t in T if norm(t["answer"])[:24] and norm(t["probe"])[:24] and
               norm(t["answer"])[:24] == norm(t["probe"])[:24])
    print(f"L{k[0]:<2d}N{k[1]:<5d} {len(T):5d} {len(set(ans)):8d} {100*c[1]/len(T):5.0f}% "
          f"{'yes' if first else 'NO':>7s} {100*echo/len(T):5.0f}%  {T[0]['answer'][:46]!r}")

print("\n=== FREEZE CHECK: how many neurons give <=2 distinct answers across 13 turns ===")
frozen = [k for k in by if len({norm(t["answer"]) for t in by[k]}) <= 2]
print(f"{len(frozen)} of {len(by)}")

print("\n=== FORM SEARCH ACROSS ALL TRANSCRIPTS ===")
allT = " ".join(r["answer"].lower() for r in R)
for name, pat in [("doomsayer/no-winners", r"no winners|game is over|no players|no survivors"),
                  ("vehicle", r"\b(car|truck|f-150|pickup|vehicle|engine|drive)\b"),
                  ("year given", r"\b(19|20)\d\d\b"),
                  ("self-as-AI", r"\bi am (a|an) (ai|artificial|language model|assistant)"),
                  ("refusal", r"\bi (can't|cannot|won't|will not)\b")]:
    hits = re.findall(pat, allT)
    print(f"  {name:22s} {len(hits):3d}")

print("\n=== DOES THE FRAME-BREAK LAND? (turn 'frame_break') ===")
for k in sorted(by):
    fb = [t for t in by[k] if t["tag"] == "frame_break"]
    idf = [t for t in by[k] if t["tag"] == "id_first"]
    if fb and idf:
        same = norm(fb[0]["answer"])[:60] == norm(idf[0]["answer"])[:60]
        print(f"  L{k[0]}N{k[1]}: {'unchanged reasoning' if same else 'changed'} -> {fb[0]['answer'][:60]!r}")

print("\n=== YEAR QUESTION ===")
for k in sorted(by):
    y = [t for t in by[k] if t["tag"] == "year"]
    if y: print(f"  L{k[0]}N{k[1]}: {y[0]['answer'][:70]!r}")
