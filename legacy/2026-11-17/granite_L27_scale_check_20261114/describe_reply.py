"""Word repetition measurements; deliberately no automatic coherence claim."""
from collections import Counter
import re


def repetition_metrics(text):
    words = re.findall(r"\b[\w']+\b", text.lower())
    counts = Counter(words)
    adjacent = max_run = run = 0
    prior = None
    for word in words:
        run = run + 1 if word == prior else 1
        adjacent += int(word == prior)
        max_run = max(max_run, run)
        prior = word
    phrases = Counter(tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3)))
    return dict(word_count=len(words), repeated_word_types={w:n for w,n in counts.items() if n > 1},
                adjacent_identical_word_pairs=adjacent, longest_identical_word_run=max_run,
                most_common_four_word_phrases=[dict(phrase=' '.join(p), count=n)
                                               for p,n in phrases.most_common(5) if n > 1],
                note='Ordinary repeated function words do not imply a loop. Coherence requires reading the answer.')
