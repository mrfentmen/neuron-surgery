import re, sys
txt = open("probe_rerun.log").read()
blocks = re.split(r"^L12 N(\d+)\s+dose.*$", txt, flags=re.M)
rows = []
for i in range(1, len(blocks), 2):
    nid = blocks[i]; body = blocks[i+1]
    qs = re.findall(r"you > (.*)", body)
    ans = re.split(r"^you > ", body, flags=re.M)[1:]
    ans = [a.split("\n",1)[1].strip() if "\n" in a else "" for a in ans]
    files = []
    for a in ans:
        files.append(" ".join(a.split()[:12]))
    distinct = len(set(ans))
    openers = len(set(files))
    rows.append((nid, len(ans), distinct, openers, ans))
print(f"{'neuron':>7} {'turns':>5} {'distinct':>8} {'distinct_opener':>15}")
for nid,n,d,o,_ in rows:
    print(f"{nid:>7} {n:>5} {d:>8} {o:>15}")
print()
for nid,n,d,o,ans in rows:
    print(f"===== N{nid} ({d}/{n} distinct) =====")
    for a in ans[:4]:
        print("   ", " ".join(a.split())[:150])
    print()
