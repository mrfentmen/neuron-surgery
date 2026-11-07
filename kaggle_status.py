"""Which Kaggle endpoints actually answer in JSON with a KGAT bearer token?"""
import json
import sys
import urllib.request
import urllib.error
import pathlib

TOK_FILE = pathlib.Path.home() / "Desktop" / "Kaggle API Token.txt"
tok = None
for line in TOK_FILE.read_text(errors="ignore").splitlines():
    for chunk in line.split():
        if chunk.startswith("KGAT_"):
            tok = chunk.strip().strip('",')
            break
    if tok:
        break
if not tok:
    sys.exit("no KGAT_ token found")

OWNER = "aeeeeeeee"
KERNELS = [f"granite-ordered-lane-l{n}-up" for n in range(9, 15)]


def get(path):
    url = "https://www.kaggle.com" + path
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {tok}",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read(4000).decode("utf-8", "replace")
            return r.status, r.headers.get("content-type", ""), body
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("content-type", ""), e.read(1000).decode("utf-8", "replace")
    except Exception as e:
        return -1, "", repr(e)


def show(label, path, limit=1200):
    code, ctype, body = get(path)
    print(f"--- {label}  [{code}] {ctype[:40]}")
    print(body[:limit].replace("\r", ""))
    print()


show("list owned", "/api/v1/kernels/list?group=owned&pageSize=50")
show("list user+search", f"/api/v1/kernels/list?user={OWNER}&search=granite&pageSize=50")
show("list search only", "/api/v1/kernels/list?search=granite-ordered-lane&pageSize=50")

for k in KERNELS:
    show(f"status {k}", f"/api/v1/kernels/status/{OWNER}/{k}", limit=400)

for k in KERNELS[-3:]:
    show(f"view {k}", f"/api/v1/kernels/view/{OWNER}/{k}", limit=600)
