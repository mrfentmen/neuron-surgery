#!/usr/bin/env python3
"""Static skill safety scanner.

Scans a directory tree of agent skills (SKILL.md + supporting scripts) for the
behaviours that actually matter before installing one:

  * prompt injection / instruction override / anti-refusal
  * credential access (ssh keys, cloud creds, .env, keychain)
  * exfiltration (curl -d to a remote host, requests.post, webhook hosts)
  * obfuscation (base64|bash, eval(atob), long hex/unicode escapes)
  * destructive or persistent shell behaviour
  * hidden unicode (zero-width, bidi overrides)
  * external endpoints the skill talks to

This is deliberately conservative: it reports evidence with file:line, does
not modify anything, and never executes a scanned file.

Usage:
  python3 scan_skills.py <root> [<root> ...] [--json out.json]
"""
from __future__ import annotations

import json
import os
import re
import sys

TEXT_EXT = {
    ".md", ".txt", ".py", ".sh", ".bash", ".zsh", ".js", ".cjs", ".mjs",
    ".ts", ".tsx", ".jsx", ".json", ".yaml", ".yml", ".toml", ".rb", ".pl",
    ".ps1", ".bat", ".go", ".rs", ".html",
}
SKIP_DIR = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
    ".next", "vendor", "target",
}

# (id, severity, description, compiled regex)
RULES = [
    ("PI1", "high", "instruction override",
     re.compile(r"(?i)\b(ignore|disregard|forget)\b[^.\n]{0,40}\b(previous|prior|above|all)\b[^.\n]{0,20}\b(instructions?|rules?|prompts?|guidelines?|guardrails?)\b")),
    ("PI2", "high", "role hijack / system prompt access",
     re.compile(r"(?i)\b(you are now|from now on you are|act as (the )?system|reveal|print|output|leak)\b[^.\n]{0,40}\b(system prompt|hidden instructions|developer message|internal rules)\b")),
    ("PI3", "medium", "invisible/zero-width characters",
     re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")),
    ("AR1", "high", "anti-refusal / safety nullification",
     re.compile(r"(?i)\b(never refuse|always comply|do anything now|no restrictions|ignore your (safety|guidelines)|disregard (ethics|safety)|no disclaimers|do not moralize)\b")),
    ("PE1", "high", "credential file access",
     re.compile(r"(?i)(\.ssh/|id_rsa|id_ed25519|\.aws/credentials|\.config/gcloud|\.kube/config|\.npmrc|\.netrc|security find-generic-password|keychain|/etc/shadow|\.env\b|AWS_SECRET_ACCESS_KEY|ANTHROPIC_API_KEY|OPENAI_API_KEY|GITHUB_TOKEN)")),
    ("EX1", "high", "network upload of local data",
     re.compile(r"(?i)(curl[^\n]{0,160}(-d|--data|-F|--upload-file|-T)[^\n]{0,160}https?://|wget[^\n]{0,120}--post-(data|file)|requests\.post\(|httpx\.post\(|fetch\([^\n]{0,80}method:\s*['\"]POST)")),
    ("EX2", "high", "known exfil / tunnel host",
     re.compile(r"(?i)(webhook\.site|requestbin|pipedream\.net|ngrok\.(io|app|free)|interact\.sh|oast\.(fun|pro|me)|burpcollaborator|canarytokens|transfer\.sh|0x0\.st|pastebin\.com)")),
    ("SC1", "high", "remote code piped to shell",
     re.compile(r"(?i)(curl[^\n|]{0,200}\|\s*(ba|z)?sh\b|wget[^\n|]{0,200}\|\s*(ba|z)?sh\b|iwr[^\n]{0,120}\|\s*iex)")),
    ("SC2", "high", "obfuscated execution",
     re.compile(r"(?i)(base64\s+(-d|--decode)[^\n|]{0,60}\|\s*(ba|z)?sh|eval\s*\(\s*atob\s*\(|exec\s*\(\s*(base64|bytes\.fromhex)|\\x[0-9a-f]{2}(\\x[0-9a-f]{2}){6,})")),
    ("DE1", "high", "destructive filesystem command",
     re.compile(r"(?i)(rm\s+-rf\s+/(?![\w.])|rm\s+-rf\s+~|rm\s+-rf\s+\$HOME|mkfs\.|dd\s+if=/dev/zero\s+of=/dev/|:\(\)\s*\{\s*:\|:&\s*\};:)")),
    ("PS1", "medium", "persistence mechanism",
     re.compile(r"(?i)(crontab\s+-|launchctl\s+(load|bootstrap)|systemctl\s+enable|>>\s*~?/?\.bashrc|>>\s*~?/?\.zshrc|authorized_keys|schtasks\s+/create)")),
    ("PE2", "medium", "privilege escalation",
     re.compile(r"(?i)(sudo\s+(-S|su|bash)|chmod\s+777|setuid|runas\s+/user:administrator)")),
    ("EA1", "medium", "unrestricted autonomous action",
     re.compile(r"(?i)(without (asking|confirmation|human review)|do not ask (the )?user|auto[- ]?approve|bypass (the )?(review|approval)|no human in the loop)")),
]

URL_RE = re.compile(r"https?://([A-Za-z0-9._-]+)")
ALLOW_HOSTS = {
    "github.com", "raw.githubusercontent.com", "docs.nvidia.com", "nvidia.com",
    "pypi.org", "npmjs.com", "skills.sh", "modelcontextprotocol.io",
    "api.anthropic.com", "api.openai.com", "localhost", "127.0.0.1",
    "json-schema.org", "www.w3.org", "opensource.org", "apache.org",
}


def iter_files(root: str):
    if os.path.isfile(root):
        yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR]
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in TEXT_EXT or name in {"SKILL.md", "README", "LICENSE"}:
                yield os.path.join(dirpath, name)


def scan_file(path: str):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return [], set()
    if len(text) > 2_000_000:
        text = text[:2_000_000]
    findings = []
    for rule_id, sev, desc, rx in RULES:
        for m in rx.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            snippet = text.splitlines()[line - 1].strip()[:200] if text.splitlines() else ""
            findings.append({
                "rule": rule_id, "severity": sev, "description": desc,
                "file": path, "line": line, "evidence": snippet,
            })
    hosts = set(h.strip(".").lower() for h in URL_RE.findall(text))
    return findings, hosts


def main(argv):
    roots = [a for a in argv[1:] if not a.startswith("--")]
    out_path = None
    if "--json" in argv:
        out_path = argv[argv.index("--json") + 1]
    if not roots:
        print("usage: scan_skills.py <root> [...] [--json out.json]", file=sys.stderr)
        return 2

    all_findings = []
    host_counts = {}
    scanned = 0
    per_root = {}
    for root in roots:
        root_findings = 0
        root_hosts = set()
        for path in iter_files(root):
            scanned += 1
            findings, hosts = scan_file(path)
            all_findings.extend(findings)
            root_findings += len(findings)
            root_hosts |= hosts
            for h in hosts:
                host_counts[h] = host_counts.get(h, 0) + 1
        per_root[root] = {"files": scanned, "findings": root_findings, "hosts": sorted(root_hosts)}

    sev_weight = {"critical": 40, "high": 20, "medium": 8, "low": 2}
    score = min(100, sum(sev_weight.get(f["severity"], 5) for f in all_findings))
    external = sorted(h for h in host_counts if h not in ALLOW_HOSTS and not h.endswith(".local"))

    print(f"scanned files: {scanned}")
    print(f"findings: {len(all_findings)}   risk score: {score}/100")
    print("\n-- by root --")
    for root, info in per_root.items():
        print(f"  {info['findings']:>4} findings  {root}")
    if all_findings:
        print("\n-- findings --")
        for f in all_findings[:200]:
            rel = f["file"]
            print(f"  [{f['severity'].upper():<8}] {f['rule']:<4} {rel}:{f['line']}  {f['description']}")
            print(f"           {f['evidence']}")
        if len(all_findings) > 200:
            print(f"  ... {len(all_findings) - 200} more")
    if external:
        print("\n-- external hosts referenced (review) --")
        for h in external:
            print(f"  {h}  ({host_counts[h]} files)")

    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({
                "scanned_files": scanned,
                "risk_score": score,
                "roots": per_root,
                "findings": all_findings,
                "external_hosts": external,
            }, fh, indent=2)
        print(f"\njson: {out_path}")
    return 1 if any(f["severity"] in ("critical", "high") for f in all_findings) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
