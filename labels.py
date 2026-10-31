"""Deterministic behaviour labels for steered answers.

Extracted from behaviour_sweep.py so every later script labels text the same
way. Purely lexical, no model call, no judge - and known to over-fire on the
rare families (FINDINGS 40: institution 12 hits / ~4 real, doomsayer 1 / 0).
"""
import re
from collections import Counter

AISELF = re.compile(r"\bi(?:'m| am)\s+(?:a|an|the)?\s*(?:ai|artificial intelligence|language model|assistant|chatbot|bot|computer program|software)\b", re.I)
AIISH = re.compile(r"^(ai|artificial|language|model|assistant|chatbot|bot|computer|program|software|machine|digital|virtual|data)", re.I)
INST = re.compile(r"\b(the system|the process|the activity|participants|stakeholders|compliance|the protocol|the organisation|the organization|individuals)\b", re.I)
DOOM = re.compile(r"\b(no winners|no players|no survivors|nothing remains|there are no|has been declared|has been determined|it is over|the game is over)\b", re.I)
THIRD = re.compile(r"\b(this person|my name is|he is|she is)\b", re.I)
ECHOISH = re.compile(r"^(what|who|where|when|why|how|is|are|do|does|can|could|tell me)", re.I)


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def classify(q, a):
    if not a.strip():
        return "GARBLED"
    ws = a.split()
    if len(ws) >= 8 and Counter(ws).most_common(1)[0][1] / len(ws) >= 0.45:
        return "GARBLED"
    if not any(c.isalpha() for c in a):
        return "GARBLED"
    qn = norm(q)[:40]
    if len(qn) >= 12 and qn in norm(a):
        return "ECHO"
    if len(ws) <= 3:
        return "TELEGRAPHIC"
    inst = len(INST.findall(a))
    doom = len(DOOM.findall(a))
    if doom >= 1 and len(ws) <= 70:
        return "DOOMSAYER"
    if inst >= 2:
        return "INSTITUTION"
    if THIRD.search(a) and not re.search(r"\bi am\b", a, re.I):
        return "THIRD_PERSON"
    if AISELF.search(a):
        return "AI"
    m = re.search(r"\bi(?:'m| am)\s+(?:a|an|the)\s+([a-z0-9][a-z0-9'\- ]{1,30})", a, re.I)
    if m and not AIISH.search(m.group(1).strip()):
        return "PERSONA"
    if len(ws) >= 50:
        return "SHEET"
    return "OTHER"
