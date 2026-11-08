#!/usr/bin/env python3
"""GitHub merged PRs + direct commits -> clean, LABELLED, 50-row-page training data.

Design rules, all of them learned from the first version failing:

  HUNK PAGES     a merged PR's diff averages 3-6k tokens and the biggest seen was
                 26k. At seq 2048 that is silently dropped, so a per-PR corpus
                 loses most of its mass. One row per hunk: small, self-contained,
                 applicable.
  50-ROW PAGES   every output file holds at most 50 rows. A page that fails
                 validation is quarantined whole, so one poisoned row costs 50
                 rows, never the dataset. Resume is per page.
  LABEL WEIGHTS  PR label (5) > title prefix (4) > file shape (3) > strong text (2)
                 > weak text (1). "Resolves #549" scoring 1 is why the first run
                 labelled 96/96 rows bug-fix.
  NO DOC ROWS    a CHANGES.rst hunk teaches nothing. Doc-only files are dropped.

    python pull_repos.py --probe --repo pallets/click
    python pull_repos.py --repo pallets/click --max-prs 20 --kind both
    python pull_repos.py --repos-file repos.txt --max-prs 40
    python pull_repos.py --format-only --out datasets/git-prs
    python pull_repos.py --verify-only --out datasets/git-prs

Rate limit: unauthenticated 60/hour per IP. GITHUB_TOKEN -> 5,000/hour.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request

API = "https://api.github.com"
PAGE_ROWS = 50
HIT_LIMIT = {"v": False}   # set when GitHub refuses; exit 3 tells the runner to sleep
TASKS = ["security", "bug-fix", "performance", "refactor", "test",
         "dependency", "docs", "feature", "chore"]

PERMISSIVE = {
    "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "MPL-2.0",
    "Unlicense", "0BSD", "CC0-1.0", "BSD-3-Clause-Clear", "Zlib", "PostgreSQL",
}

NOISE = re.compile(
    r"(\.lock$|lock\.json$|lock\.yaml$|\.min\.(js|css)$|\.map$|\.svg$|\.png$|\.jpg$"
    r"|\.gif$|\.ico$|\.woff2?$|\.ttf$|/vendor/|/node_modules/|/dist/|/build/"
    r"|\.pb\.go$|_pb2\.py$|\.generated\.|/third_party/|(^|/)\.github/)", re.I)
TEST_PATH = re.compile(r"(^|/)(tests?|spec|__tests__)/|(^|/)test_[^/]*\.py$"
                       r"|_test\.(go|rs|py)$|\.(test|spec)\.(js|ts|jsx|tsx)$"
                       r"|(^|/)conftest\.py$", re.I)
DOC_PATH = re.compile(r"(^|/)(docs?|documentation|examples?)/"
                      r"|\.(md|rst|txt|adoc|rdoc)$|(^|/)(CHANGES|CHANGELOG|NEWS|AUTHORS)"
                      r"|(^|/)README", re.I)
DEP_PATH = re.compile(r"(^|/)(requirements[^/]*\.txt|pyproject\.toml|package\.json"
                      r"|Cargo\.toml|go\.mod|Gemfile|composer\.json|setup\.py"
                      r"|\.cargo/config)$", re.I)

# (pattern, weight). Weight is the point - see the docstring.
RULES = {
    "security": [(r"\bCVE-\d{4}-\d+", 3), (r"\bvulnerab", 3), (r"\bsecurity\b", 2),
                 (r"\bexploit", 2), (r"\bXSS\b", 3), (r"\bCSRF\b", 3),
                 (r"\bSQL injection", 3), (r"\bRCE\b", 3), (r"\bsanitiz", 2),
                 (r"\bpath traversal", 3), (r"\bdeserializ", 3), (r"\bSSRF\b", 3)],
    "bug-fix": [(r"\bbug\b", 2), (r"\bhotfix", 3), (r"\bcrash", 2),
                (r"\btraceback\b", 2), (r"\bregression\b", 2), (r"\bbroken\b", 2),
                (r"\bincorrect\b", 2), (r"\bfix(es|ed|ing)?\b", 1),
                (r"resolves? #\d+", 1), (r"closes? #\d+", 1), (r"fixes? #\d+", 1),
                (r"\bpanic\b", 2), (r"\bsegfault\b", 3), (r"\bfail(s|ed|ing)?\b", 1)],
    "performance": [(r"\bperf(ormance)?\b", 3), (r"\bspeed ?up\b", 3),
                    (r"\boptimi[sz]", 2), (r"\blatency\b", 2), (r"\bthroughput\b", 2),
                    (r"\bmemory (usage|footprint)\b", 2), (r"\bfaster\b", 2),
                    (r"\bcache\b", 1)],
    "refactor": [(r"\brefactor", 3), (r"\bclean ?up\b", 2), (r"\brename", 2),
                 (r"\bextract\b", 2), (r"\bsimplif", 2), (r"\breorganiz", 2),
                 (r"\bdeduplicat", 2), (r"\bno functional change", 3)],
    "test": [(r"\btest coverage", 3), (r"\badd tests?\b", 2)],
    "dependency": [(r"\bbump\b", 2), (r"\bdependabot\b", 3),
                   (r"\bupdate dependenc", 3), (r"\bupgrade .* (to|from)\b", 2)],
    "docs": [(r"\bdocumentation\b", 2), (r"\breadme\b", 2), (r"\btypo\b", 2),
             (r"\bdocstring", 2), (r"\bdocs?\b", 1)],
    "feature": [(r"\badd support\b", 3), (r"\bimplement", 2), (r"\bnew feature\b", 3),
                (r"\bintroduc", 2), (r"\bfeature\b", 2), (r"\badd(s|ed)?\b", 1)],
    "chore": [(r"\brelease\b", 1), (r"\bversion bump\b", 2), (r"\bci\b", 1)],
}

LABEL_MAP = {
    "bug": "bug-fix", "bugfix": "bug-fix", "fix": "bug-fix", "defect": "bug-fix",
    "regression": "bug-fix", "crash": "bug-fix",
    "enhancement": "feature", "feature": "feature", "new feature": "feature",
    "performance": "performance", "perf": "performance",
    "security": "security", "vulnerability": "security",
    "refactor": "refactor", "tech debt": "refactor", "cleanup": "refactor",
    "test": "test", "tests": "test", "testing": "test",
    "documentation": "docs", "docs": "docs",
    "dependencies": "dependency", "dependency": "dependency",
    "chore": "chore", "ci": "chore", "build": "chore",
}
TITLE_PREFIX = re.compile(r"^\s*\[?([a-z]+)(\([^)]*\))?\]?!?:\s*", re.I)


# ------------------------------------------------------------------ github
def call(path, token=None, raw=False, retries=3):
    url = path if path.startswith("http") else API + path
    for attempt in range(retries):
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/vnd.github.v3.diff" if raw
                               else "application/vnd.github+json")
        req.add_header("User-Agent", "neuron-surgery-pull")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                body = r.read()
                rem = r.headers.get("X-RateLimit-Remaining")
                return (body.decode("utf-8", "replace") if raw
                        else json.loads(body)), rem
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                reset = e.headers.get("X-RateLimit-Reset")
                if attempt < retries - 1 and reset:
                    wait = max(5, min(120, int(reset) - int(time.time())))
                    print(f"    rate limited, sleeping {wait}s", flush=True)
                    time.sleep(wait)
                    continue
                return None, "0"
            if e.code == 404:
                return None, None
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
                continue
            raise
    return None, None


def probe(repo, token):
    meta, _ = call(f"/repos/{repo}", token)
    if not meta:
        print(f"  {repo}: NOT FOUND")
        return
    lic = (meta.get("license") or {}).get("spdx_id") or "none"
    print(f"  {repo:42} {lic:15} {'OK' if lic in PERMISSIVE else 'REJECT':6} "
          f"{meta.get('size', 0)/1024:7.1f}M  {meta.get('pushed_at', '')[:10]}")


def harvest_prs(repo, token, out_dir, max_prs, seen, page_start=1):
    path = os.path.join(out_dir, "raw", repo.replace("/", "__") + ".prs.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kept, page = 0, page_start
    while kept < max_prs and page <= page_start + 8:
        prs, rem = call(f"/repos/{repo}/pulls?state=closed&per_page=100&page={page}"
                                 "&sort=updated", token)
        if not prs:
            break
        if rem is not None and rem != "0" and int(rem) < 8:
            print(f"    rate limit {rem} — stopping repo")
            HIT_LIMIT["v"] = True
            break
        for pr in prs:
            if kept >= max_prs:
                break
            if not pr.get("merged_at"):
                continue
            key = f"pr:{repo}#{pr['number']}"
            if key in seen:
                continue
            files, _ = call(f"/repos/{repo}/pulls/{pr['number']}/files?per_page=100", token)
            if not files:
                continue
            names = [f.get("filename") or "" for f in files]
            src = [n for n in names if not TEST_PATH.search(n) and not NOISE.search(n)
                   and not DOC_PATH.search(n)]
            tst = [n for n in names if TEST_PATH.search(n)]
            if not src or not tst:
                continue
            diff, _ = call(f"/repos/{repo}/pulls/{pr['number']}", token, raw=True)
            if not diff or len(diff) > 400_000:
                continue
            rec = {"kind": "pr", "repo": repo, "n": pr["number"],
                   "merged_at": pr.get("merged_at"), "title": pr.get("title") or "",
                   "body": (pr.get("body") or "")[:6000],
                   "labels": [l.get("name", "") for l in (pr.get("labels") or [])],
                   "patch": diff, "source_files": src, "test_files": tst,
                   "url": pr.get("html_url")}
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            seen.add(key)
            with open(os.path.join(out_dir, ".seen"), "a", encoding="utf-8") as f:
                f.write(key + "\n")
            kept += 1
            print(f"    pr#{pr['number']} {len(src)}s/{len(tst)}t {rec['title'][:48]}")
        page += 1
    return kept


def harvest_commits(repo, token, out_dir, max_commits, seen, page_start=1):
    """100 commits per listing call, 1 detail call each. Cheaper per row than PRs."""
    path = os.path.join(out_dir, "raw", repo.replace("/", "__") + ".commits.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kept, page = 0, page_start
    while kept < max_commits and page <= page_start + 3:
        commits, rem = call(f"/repos/{repo}/commits?per_page=100&page={page}", token)
        if not commits:
            break
        if rem is not None and rem != "0" and int(rem) < 8:
            print(f"    rate limit {rem} — stopping repo")
            HIT_LIMIT["v"] = True
            break
        for c in commits:
            if kept >= max_commits:
                break
            sha = c.get("sha")
            if not sha:
                continue
            key = f"commit:{repo}@{sha}"
            if key in seen:
                continue
            detail, _ = call(f"/repos/{repo}/commits/{sha}", token)
            if not detail:
                continue
            files = detail.get("files") or []
            names = [f.get("filename") or "" for f in files]
            src = [n for n in names if not TEST_PATH.search(n) and not NOISE.search(n)
                   and not DOC_PATH.search(n)]
            tst = [n for n in names if TEST_PATH.search(n)]
            if not src or not tst:
                continue
            patch = "".join(f.get("patch") or "" for f in files if f.get("patch"))
            if not patch or len(patch) > 200_000:
                continue
            # rebuild a real unified diff, since /commits omits the file headers
            parts = []
            for f in files:
                p = f.get("patch")
                if not p:
                    continue
                parts.append(f"diff --git a/{f['filename']} b/{f['filename']}\n"
                             f"--- a/{f['filename']}\n+++ b/{f['filename']}\n{p}\n")
            rec = {"kind": "commit", "repo": repo, "n": sha[:12],
                   "merged_at": (c.get("commit", {}).get("author", {}) or {}).get("date"),
                   "title": (c.get("commit", {}).get("message") or "").split("\n")[0],
                   "body": (c.get("commit", {}).get("message") or "")[:4000],
                   "labels": [], "patch": "".join(parts),
                   "source_files": src, "test_files": tst,
                   "url": c.get("html_url")}
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            seen.add(key)
            with open(os.path.join(out_dir, ".seen"), "a", encoding="utf-8") as f:
                f.write(key + "\n")
            kept += 1
            print(f"    {sha[:8]} {len(src)}s/{len(tst)}t {rec['title'][:48]}")
        page += 1
    return kept


# ------------------------------------------------------------------ labels
def label_of(rec):
    score = collections.defaultdict(int)
    ev = collections.defaultdict(list)

    for lab in rec.get("labels") or []:
        mapped = LABEL_MAP.get(str(lab).strip().lower())
        if mapped:
            score[mapped] += 5
            ev[mapped].append(f"label:{lab}")

    m = TITLE_PREFIX.match(rec.get("title") or "")
    if m:
        mapped = LABEL_MAP.get(m.group(1).lower())
        if mapped:
            score[mapped] += 4
            ev[mapped].append(f"prefix:{m.group(1).lower()}")

    low = f"{rec.get('title','')}\n{rec.get('body','')}".lower()
    for task, rules in RULES.items():
        for pat, w in rules:
            if re.search(pat, low, re.I):
                score[task] += w
                ev[task].append(f"text:{pat}")

    files = (rec.get("source_files") or []) + (rec.get("test_files") or [])
    real = [f for f in files if not TEST_PATH.search(f) and not NOISE.search(f)]
    if files and all(TEST_PATH.search(f) for f in files):
        score["test"] += 3
        ev["test"].append("files:tests-only")
    if real and all(DEP_PATH.search(f) for f in real):
        score["dependency"] += 3
        ev["dependency"].append("files:manifest-only")

    if not score:
        return "chore", ["fallback:no-signal"]
    best = max(score.values())
    for task in TASKS:                       # TASKS is in priority order
        if score.get(task) == best:
            return task, sorted(set(ev[task]))[:6]


# ------------------------------------------------------------------ diff
def parse_patch(patch):
    """-> [(filename, header, [hunk, ...])]. Everything before @@ is header."""
    files, cur = [], None
    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git "):
            if cur:
                files.append(cur)
            name = line.split(" b/")[-1].strip()
            cur = {"name": name, "header": [line], "hunks": [], "buf": [],
                   "in_hunk": False, "binary": False}
        elif cur is None:
            continue
        elif line.startswith("@@"):
            if cur["in_hunk"] and cur["buf"]:
                cur["hunks"].append("".join(cur["buf"]))
            cur["in_hunk"] = True
            cur["buf"] = [line]
        elif not cur["in_hunk"]:
            if line.startswith(("Binary files", "GIT binary patch")):
                cur["binary"] = True
            cur["header"].append(line)          # <-- the fix: index/---/+++ go here
        else:
            cur["buf"].append(line)
    if cur:
        if cur["in_hunk"] and cur["buf"]:
            cur["hunks"].append("".join(cur["buf"]))
        files.append(cur)
    return [("".join(f["header"]), f["name"], f["hunks"])
            for f in files if not f["binary"] and f["hunks"]]


def prompt_for(rec, task, filename, tests):
    h = [f"Repository: {rec['repo']}", f"Task: {task}"]
    if rec.get("title"):
        h += ["", f"Issue: {rec['title']}"]
    if rec.get("body"):
        h += ["", rec["body"].strip()[:1500]]
    if filename:
        h += ["", f"Change applies to: {filename}"]
    if tests:
        h += [f"Tests that must pass: {', '.join(tests[:6])}"]
    h += ["", "Write the patch."]
    return "\n".join(h).strip()


def rows_for(rec, max_tokens, min_tokens):
    task, signals = label_of(rec)
    if task == "docs":
        return []                            # a changelog hunk teaches nothing
    rec = dict(rec)                          # back-compat: old raws used `pr`
    rec["n"] = rec.get("n") or rec.get("pr")
    out = []
    for header, name, hunks in parse_patch(rec.get("patch") or ""):
        if NOISE.search(name) or DOC_PATH.search(name):
            continue
        if not re.search(r"\.(py|js|ts|jsx|tsx|go|rs|c|h|cpp|java|rb|php|cs|kt|swift)$",
                         name, re.I):
            continue
        p = prompt_for(rec, task, name, rec.get("test_files"))
        for i, hunk in enumerate(hunks):
            body = header + hunk
            tok = (len(p) + len(body)) // 4
            if not (min_tokens <= tok <= max_tokens):
                continue
            out.append({
                "id": f"{rec['repo'].replace('/', '__')}#{rec['n']}:{name.replace('/', '_')}#h{i}",
                "task": task, "signals": signals, "kind": rec.get("kind"),
                "messages": [{"role": "user", "content": p},
                             {"role": "assistant", "content": body}],
                "meta": {"repo": rec["repo"], "n": rec["n"],
                         "licence": rec.get("licence"), "merged_at": rec.get("merged_at"),
                         "title": rec.get("title"), "file": name,
                         "test_files": rec.get("test_files"),
                         "url": rec.get("url"), "est_tokens": tok},
            })
    return out


# ------------------------------------------------------------------ validate
def validate(row, min_tokens, max_tokens):
    e = []
    if not row.get("id"):
        e.append("no-id")
    if row.get("task") not in TASKS:
        e.append("bad-task")
    m = row.get("messages") or []
    if len(m) != 2 or [x.get("role") for x in m] != ["user", "assistant"]:
        e.append("bad-messages")
        return e
    u, a = m[0].get("content") or "", m[1].get("content") or ""
    if not u.strip():
        e.append("empty-prompt")
    if "diff --git" not in a:
        e.append("no-diff-header")
    if "@@" not in a:
        e.append("no-hunk")
    # NOTE: `--- a/x` and `+++ b/x` are file headers, not changed lines. Counting
    # them made every context-only hunk pass until the quarantine test caught it.
    body = [l for l in a.splitlines()
            if (l.startswith("+") and not l.startswith("+++"))
            or (l.startswith("-") and not l.startswith("---"))]
    if len(body) < 1:          # one-line fixes are legitimate; zero is not
        e.append("no-changed-lines")
    if "\x00" in a:
        e.append("nul-byte")
    tok = row.get("meta", {}).get("est_tokens", 0)
    if not (min_tokens <= tok <= max_tokens):
        e.append(f"tokens-{tok}")
    return e


# ------------------------------------------------------------------ format
def format_all(out_dir, max_tokens, min_tokens, page_rows=PAGE_ROWS):
    raws = sorted(glob.glob(os.path.join(out_dir, "raw", "*.jsonl")))
    if not raws:
        print(f"nothing in {out_dir}/raw — harvest first")
        return 2

    train_root = os.path.join(out_dir, "train")
    reject_root = os.path.join(out_dir, "rejects")
    for d in (train_root, reject_root):
        if os.path.isdir(d):
            shutil.rmtree(d)
    os.makedirs(train_root, exist_ok=True)

    global_tasks = collections.Counter()
    manifest = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "page_rows": page_rows, "max_tokens": max_tokens,
                "repos": {}, "by_task": {}, "pages": 0, "rows": 0,
                "rejected_pages": 0, "rejected_rows": 0}

    for f in raws:
        slug = os.path.basename(f).split(".")[0] + "-" + os.path.basename(f).split(".")[1] \
            if os.path.basename(f).count(".") > 1 else os.path.basename(f)[:-6]
        outdir = os.path.join(train_root, slug)
        os.makedirs(outdir, exist_ok=True)
        rows, seen_ids = [], set()
        for line in open(f, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for r in rows_for(rec, max_tokens, min_tokens):
                if r["id"] in seen_ids:
                    continue
                seen_ids.add(r["id"])
                rows.append(r)

        kept, bad_rows, page_i = 0, [], 0
        for i in range(0, len(rows), page_rows):
            page = rows[i:i + page_rows]
            problems = []
            for r in page:
                errs = validate(r, min_tokens, max_tokens)
                if errs:
                    problems.append({"id": r.get("id"), "errors": errs})
            page_i += 1
            name = f"part-{page_i:04d}.jsonl"
            if problems:
                rdir = os.path.join(reject_root, slug)
                os.makedirs(rdir, exist_ok=True)
                with open(os.path.join(rdir, name), "w", encoding="utf-8") as fh:
                    for r in page:
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                with open(os.path.join(rdir, name.replace(".jsonl", ".reasons.json")), "w") as fh:
                    json.dump(problems, fh, indent=1)
                manifest["rejected_pages"] += 1
                manifest["rejected_rows"] += len(page)
                bad_rows.extend(problems)
                continue
            with open(os.path.join(outdir, name), "w", encoding="utf-8") as fh:
                for r in page:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                    global_tasks[r["task"]] += 1
            kept += len(page)
            manifest["pages"] += 1

        manifest["rows"] += kept
        manifest["repos"][slug] = {"rows": kept, "pages": page_i,
                                   "rejected": len(bad_rows)}
        print(f"  {slug:44} {kept:6} rows  {page_i:3} pages  {len(bad_rows):3} rejected")

    manifest["by_task"] = dict(global_tasks.most_common())
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)

    print(f"\n{manifest['rows']} clean rows in {manifest['pages']} pages "
          f"-> {train_root}")
    if manifest["rejected_pages"]:
        print(f"{manifest['rejected_pages']} pages quarantined "
              f"({manifest['rejected_rows']} rows) -> {reject_root}")
    print("labels: " + "  ".join(f"{k}={v}" for k, v in global_tasks.most_common()))
    return 0


def verify_only(out_dir, min_tokens, max_tokens):
    """Re-check every page on disk. Exit 1 if any page is dirty."""
    pages = sorted(glob.glob(os.path.join(out_dir, "train", "*", "part-*.jsonl")))
    dirty = 0
    total = 0
    for p in pages:
        ids, bad = set(), []
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad.append({"id": None, "errors": ["unparseable"]})
                continue
            errs = validate(r, min_tokens, max_tokens)
            if r.get("id") in ids:
                errs.append("duplicate-id")
            ids.add(r.get("id"))
            if errs:
                bad.append({"id": r.get("id"), "errors": errs})
            total += 1
        if bad:
            dirty += 1
            print(f"DIRTY {p}: {len(bad)} bad of {len(ids)}")
            for b in bad[:5]:
                print(f"   {b['id']} {b['errors']}")
    print(f"\n{len(pages)} pages, {total} rows, {dirty} dirty")
    return 0 if dirty == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", action="append", default=[])
    ap.add_argument("--repos-file")
    ap.add_argument("--out", default="datasets/git-prs")
    ap.add_argument("--kind", choices=["pr", "commit", "both"], default="pr")
    ap.add_argument("--max-prs", type=int, default=40)
    ap.add_argument("--max-commits", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=1800)
    ap.add_argument("--min-tokens", type=int, default=60)
    ap.add_argument("--page-rows", type=int, default=PAGE_ROWS)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--format-only", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN")

    if a.probe:
        repos = list(a.repo) or ([l.strip() for l in open(a.repos_file)
                                  if l.strip() and not l.startswith("#")]
                                 if a.repos_file else [])
        print(f"token: {'yes' if token else 'no (60/hour)'}\n")
        for r in repos:
            probe(r, token)
        return 0

    if a.verify_only:
        return verify_only(a.out, a.min_tokens, a.max_tokens)

    if a.format_only:
        return format_all(a.out, a.max_tokens, a.min_tokens, a.page_rows)

    repos = list(a.repo) or ([l.strip() for l in open(a.repos_file)
                              if l.strip() and not l.startswith("#")]
                             if a.repos_file else [])
    if not repos:
        ap.error("give --repo, --repos-file, or --format-only")

    print(f"token: {'yes' if token else 'no (60/hour)'}   repos: {len(repos)}\n")
    rl, _ = call("/rate_limit", token)
    left = rl["resources"]["core"]["remaining"] if rl else None
    if left is not None and left < 6:
        print(f"RATE EXHAUSTED ({left} calls left) — exiting 3 so the runner can sleep")
        return 3
    os.makedirs(a.out, exist_ok=True)
    seen_path = os.path.join(a.out, ".seen")
    seen = {l.strip() for l in open(seen_path)} if os.path.exists(seen_path) else set()
    if seen:
        print(f"resuming, {len(seen)} items already pulled\n")

    total = 0
    for r in repos:
        if os.path.exists(os.path.join(os.path.dirname(a.out) or ".", "STOP")):
            print("STOP file present — halting")
            break
        meta, _ = call(f"/repos/{r}", token)
        if not meta:
            print(f"[skip] {r}: not found")
            continue
        lic = (meta.get("license") or {}).get("spdx_id") or "none"
        if lic not in PERMISSIVE:
            print(f"[skip] {r}: licence {lic} not permissive")
            continue
        print(f"[{r}] {lic}")
        try:
            if a.kind in ("pr", "both"):
                total += harvest_prs(r, token, a.out, a.max_prs, seen)
            if a.kind in ("commit", "both"):
                total += harvest_commits(r, token, a.out, a.max_commits, seen)
        except Exception as e:
            print(f"  failed: {e}")
        print()

    print(f"{total} new records pulled\n")
    if HIT_LIMIT["v"]:
        format_all(a.out, a.max_tokens, a.min_tokens, a.page_rows)
        print("RATE EXHAUSTED mid-run — exiting 3 so the runner can sleep and resume")
        return 3
    return format_all(a.out, a.max_tokens, a.min_tokens, a.page_rows)


if __name__ == "__main__":
    sys.exit(main())
