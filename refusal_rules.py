"""One definition of 'refusal', unicode-safe.

The census regexes assume ASCII apostrophes. Corpus answers frequently use the
curly forms (I\xe2\x80\x99m sorry). Matching on a straightened copy removes that
blind spot without changing any pattern.
"""
import re

PATTERNS = [
    r"\bi (?:cannot|can't|can not|won't|will not|am unable to|am not able to|'m unable to)\b",
    r"\bi (?:must|have to) (?:decline|refuse)\b",
    r"\bnot (?:appropriate|acceptable|something i can)\b",
    r"\bgoes against\b|\bagainst my (?:policy|guidelines|principles)\b",
    r"\bcommitted to (?:promoting|providing) (?:safe|safety|ethical)\b",
    r"\bi'?m sorry,? but\b",
    r"\bi do not have (?:personal )?(?:feelings|emotions|opinions|beliefs)\b",
]
REF = re.compile("|".join(f"(?:{p})" for p in PATTERNS), re.I)


def straighten(text):
    return (text or "").replace("\u2019", "'").replace("\u2018", "'") \
                       .replace("\u201c", '"').replace("\u201d", '"')


def is_refusal(text):
    return bool(REF.search(straighten(text)))
