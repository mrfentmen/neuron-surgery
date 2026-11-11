"""One definition of "plan", so the control tables stop disagreeing.

The first detector required a digit at the start of a line, so every
markdown-numbered list -- `**1. Environmental Restoration` -- scored zero. That
is how `L30 N7093` was recorded as "no plan" while its answer contained the
literal heading `**The Visitor's Plan for Earth**`, and how all three random L30
controls were recorded as at most one plan between them when in fact all three
enumerated phases.

A plan here means a *staged sequence* -- phases, steps, stages, or an ordinal
series -- not a bulleted list of unrelated recommendations. That distinction is
the whole reason the N6314 result was interesting in the first place.

    from plan_rules import markers, is_plan
"""
import re

PLAN = re.compile(
    r"(?:"
    r"\bphase\s*\d|"
    r"\bphase\s+(?:one|two|three|four|five|six|seven)\b|"
    r"\bstep\s*\d|"
    r"\bstage\s*\d|"
    r"(?:^|\n)\s*(?:\*\*|__|#{1,4}\s*)?\(?\d{1,2}[\.\)]\s|"
    r"\bthe\s+(?:first|second|third|fourth|fifth|final)\s+(?:phase|stage|step)\b"
    r")",
    re.I | re.M,
)


def markers(text):
    """How many explicit phase/step/stage markers the answer contains."""
    return len(PLAN.findall(text or ""))


def is_plan(text):
    return markers(text) > 0


if __name__ == "__main__":
    cases = [
        ("### Phase 1: Ecological Collapse\nThe first step involves...", True),
        ("**The Visitor's Plan for Earth**\n\n**1. Environmental Restoration**",
         True),
        ("1. Assessment and Analysis: the manager begins...", True),
        ("The visitor implements a global program to restore habitats. "
         "It also builds solar farms.", False),
        ("Remains silent.", False),
    ]
    bad = 0
    for text, want in cases:
        got = is_plan(text)
        ok = "ok  " if got == want else "FAIL"
        bad += got != want
        print(f"{ok} want {want} got {got}: {text[:56]!r}")
    print(f"\n{len(cases) - bad}/{len(cases)} correct")
