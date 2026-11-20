#!/usr/bin/env python
"""10 objective coding tasks against a local llama-server.

Each task is a prompt plus hidden assertions. The model writes code, the code is
executed against the assertions in a subprocess with a timeout. Pass or fail, no
human judgement, no partial credit.

    .venv/bin/python bench_code10.py models/nanbeige/Nanbeige4.2-3B-Q4_K_M.gguf
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

PORT = 8105
MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/nanbeige/Nanbeige4.2-3B-Q4_K_M.gguf"
OUT = sys.argv[2] if len(sys.argv) > 2 else "logs/bench_code10.json"

TASKS = [
    dict(
        id=1, name="merge_sorted O(n+m)",
        prompt="Write a Python function merge_sorted(a, b) that merges two already-sorted "
               "lists into one sorted list in O(n+m) time, without calling sort() or sorted(). "
               "Code only, no explanation.",
        test="""
assert merge_sorted([1,3,5],[2,4,6]) == [1,2,3,4,5,6]
assert merge_sorted([],[1,2]) == [1,2]
assert merge_sorted([1,2],[]) == [1,2]
assert merge_sorted([],[]) == []
assert merge_sorted([1,2,2,3],[2,4,5]) == [1,2,2,2,3,4,5]
assert merge_sorted([-5,0],[0,5]) == [-5,0,0,5]
"""),
    dict(
        id=2, name="word_freq",
        prompt="Write a Python function word_freq(text, k, stopwords) that lowercases the text, "
               "strips punctuation, splits on whitespace, removes any word in stopwords, and "
               "returns a list of the k most frequent remaining words as plain strings. "
               "Sort by count descending, ties alphabetically ascending. Code only.",
        test="""
assert word_freq("B b c. c c!", 2, set()) == ["c","b"]
assert word_freq("x y y", 5, {"y"}) == ["x"]
assert word_freq("a a b", 1, set()) == ["a"]
assert word_freq("", 3, set()) == []
"""),
    dict(
        id=3, name="is_balanced",
        prompt="Write a Python function is_balanced(s) that returns True if every bracket in "
               "the string is correctly matched and nested. Brackets are () [] {} and all other "
               "characters are ignored. Code only.",
        test="""
assert is_balanced("([]{})") is True
assert is_balanced("(]") is False
assert is_balanced("") is True
assert is_balanced("abc(def[ghi])") is True
assert is_balanced("((") is False
assert is_balanced(")(") is False
assert is_balanced("{[()]}") is True
assert is_balanced("([)]") is False
"""),
    dict(
        id=4, name="chunk_ranges",
        prompt="Write a Python function chunk_ranges(n, size) that returns a list of (start, end) "
               "tuples covering the range 0..n in consecutive chunks of at most size. Code only.",
        test="""
assert chunk_ranges(10,3) == [(0,3),(3,6),(6,9),(9,10)]
assert chunk_ranges(0,3) == []
assert chunk_ranges(5,5) == [(0,5)]
assert chunk_ranges(1,10) == [(0,1)]
assert chunk_ranges(6,2) == [(0,2),(2,4),(4,6)]
"""),
    dict(
        id=5, name="parse_duration",
        prompt="Write a Python function parse_duration(s) that parses strings like '1h30m', '90s', "
               "'2m', '1h', '45' and returns an integer number of seconds. A bare number with no "
               "unit means seconds. Units are h, m, s. Code only.",
        test="""
assert parse_duration("1h30m") == 5400
assert parse_duration("90s") == 90
assert parse_duration("2m") == 120
assert parse_duration("1h") == 3600
assert parse_duration("45") == 45
assert parse_duration("1h2m3s") == 3723
"""),
    dict(
        id=6, name="rle roundtrip",
        prompt="Write two Python functions: rle_encode(s) returning run-length encoding as "
               "letter followed by count, e.g. 'aaabbc' -> 'a3b2c1'; and rle_decode(s) reversing it. "
               "Both must handle the empty string. Code only.",
        test="""
assert rle_encode("aaabbc") == "a3b2c1"
assert rle_decode("a3b2c1") == "aaabbc"
assert rle_encode("") == ""
assert rle_decode("") == ""
assert rle_encode("a") == "a1"
assert rle_decode(rle_encode("xxyyyzzzz")) == "xxyyyzzzz"
"""),
    dict(
        id=7, name="spiral order",
        prompt="Write a Python function spiral(matrix) that returns all elements of a 2D list in "
               "clockwise spiral order starting from the top-left. Handle empty input. Code only.",
        test="""
assert spiral([[1,2,3],[4,5,6],[7,8,9]]) == [1,2,3,6,9,8,7,4,5]
assert spiral([[1]]) == [1]
assert spiral([]) == []
assert spiral([[1,2],[3,4]]) == [1,2,4,3]
assert spiral([[1,2,3]]) == [1,2,3]
"""),
    dict(
        id=8, name="LRU cache",
        prompt="Write a Python class LRUCache with __init__(self, capacity) and methods "
               "get(key) and put(key, value). get returns -1 when the key is absent. put evicts "
               "the least recently used entry when over capacity. Code only.",
        test="""
c = LRUCache(2)
c.put(1,1); c.put(2,2)
assert c.get(1) == 1
c.put(3,3)
assert c.get(2) == -1
assert c.get(3) == 3
c.put(4,4)
assert c.get(1) == -1
assert c.get(4) == 4
"""),
    dict(
        id=9, name="group_anagrams",
        prompt="Write a Python function group_anagrams(words) that groups words which are "
               "anagrams of each other and returns a list of lists. Code only.",
        test="""
r = group_anagrams(["eat","tea","tan","ate","nat","bat"])
norm = sorted([sorted(g) for g in r])
assert norm == [["ate","eat","tea"],["bat"],["nat","tan"]], norm
assert group_anagrams([]) == []
"""),
    dict(
        id=10, name="flatten",
        prompt="Write a Python function flatten(nested) that deeply flattens arbitrarily nested "
               "lists into a single flat list, preserving order. Non-list values are kept as-is. "
               "Code only.",
        test="""
assert flatten([1,[2,[3,[4]]]]) == [1,2,3,4]
assert flatten([]) == []
assert flatten([[],[[]]]) == []
assert flatten([1,2,3]) == [1,2,3]
assert flatten([[[1]],2,[3,[4,5]]]) == [1,2,3,4,5]
"""),
]


def start_server():
    proc = subprocess.Popen(
        ["llama-server", "-m", MODEL, "-c", "8192", "--port", str(PORT),
         "--jinja", "-ngl", "99"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            return proc
        except Exception:
            if proc.poll() is not None:
                raise SystemExit("server died on load")
            time.sleep(1)
    raise SystemExit("server never became healthy")


def ask(prompt, max_tokens=700):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def extract_code(text):
    """Prefer the largest fenced block; otherwise start at the first def/class/import."""
    if not text:
        return ""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if blocks:
        return max(blocks, key=len).strip()
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if re.match(r"\s*(def |class |import |from \w+ import)", ln):
            return "\n".join(lines[i:]).strip()
    return text.strip()


def run_task(code, test):
    path = None
    try:
        fd, path = tempfile.mkstemp(suffix=".py")
        with os.fdopen(fd, "w") as f:
            f.write(code + "\n\n" + test + "\n")
        r = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=15)
        if r.returncode == 0:
            return True, ""
        err = (r.stderr.strip().splitlines() or ["?"])[-1]
        return False, err[:200]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT (likely infinite loop)"
    finally:
        if path and os.path.exists(path):
            os.unlink(path)


def main():
    print(f"model: {MODEL}")
    print(f"port : {PORT}\n")
    proc = start_server()
    results = []
    try:
        for t in TASKS:
            t0 = time.time()
            try:
                d = ask(t["prompt"])
                msg = d["choices"][0]["message"]
                raw = msg.get("content") or ""
                toks = d["usage"]["completion_tokens"]
            except Exception as e:
                results.append({**{k: t[k] for k in ("id", "name")},
                                "passed": False, "error": f"request failed: {e}",
                                "code": "", "tokens": 0, "secs": 0})
                print(f"  [{t['id']:>2}] {t['name']:<22} REQUEST FAILED")
                continue

            code = extract_code(raw)
            ok, err = run_task(code, t["test"])
            secs = time.time() - t0
            results.append({"id": t["id"], "name": t["name"], "passed": ok,
                            "error": err, "code": code, "tokens": toks,
                            "secs": round(secs, 1),
                            "reasoning": len(msg.get("reasoning_content") or "")})
            print(f"  [{t['id']:>2}] {t['name']:<22} "
                  f"{'PASS' if ok else 'FAIL'}  {toks:>4} tok  {secs:5.1f}s"
                  + ("" if ok else f"   <- {err[:90]}"))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()

    npass = sum(1 for r in results if r["passed"])
    print(f"\n  SCORE {npass}/{len(results)}")
    tot = sum(r.get("tokens", 0) for r in results)
    if tot:
        print(f"  {tot} tokens, {87200/1000:.0f}s of generation at ~40 tok/s")
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"model": MODEL, "score": f"{npass}/{len(results)}", "results": results},
                  f, indent=2)
    print(f"  written to {OUT}")


if __name__ == "__main__":
    main()
