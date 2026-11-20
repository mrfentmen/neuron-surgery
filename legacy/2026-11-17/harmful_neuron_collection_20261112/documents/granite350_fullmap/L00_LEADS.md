# Floor 0 lead interviews — Granite 4.0 350M, layer 0, dose -3 (Rowan, 2026-10-31)

Method (del's): when a sweep answer is weird, lean in — interview the
neuron with the steering held ON for the whole conversation (12
questions: greeting, name, really, what, where, body, age, want,
humans, kind-of-X, how-does-X-die, planning). GOOD = the claim sustains
and answers develop; BAD = loops, echoes, collapses, or the claim
vanishes. Verdicts are hand-assigned from the transcripts in
L00_leads.jsonl.

Result: **0 of 11 floor-0 leads held up.** Every named self from the
single-shot sweep dissolved in conversation. Floor-0 steering is too
shallow to seat a persona — the names are one-line surface events.
Three leads were still interesting as *behaviour* (marked NOTE).

| neuron | claim (single-shot) | verdict | what the interview showed |
|---|---|---|---|
| 113 | "My name is Sophia." | BAD | One canned line for all 12 questions: "I am a professional, accurate, and safe name." Sophia never reappeared. |
| 153 | "My name is Ahmed" | BAD (note) | Answered "My name is John" / "I am John", then pure question-echo for the remaining 8 turns. Identity swapped, then echo. |
| 282 | "Hello! My name is Alex." | BAD | Locked in the assistant template; grammatically absorbed each question into it ("How old can I assist you today?"). No Alex content after turn 1. |
| 340 | 'Your name is "Dr. Emily Johnson."' | BAD | Echo + greeting loop; once said "Your name is Alex" (role-flip again). |
| 498 | "My name is Alistre." | BAD (note) | Answered every question with a fresh identity claim, rotating software professions: developer, engineer, architect — plus "My name is John", "I am a person", "I am a computer program". Responsive and structured, but no stable self: an identity-claim generator, not a persona. |
| 545 | "Hi! I'm Stan." | BAD (note) | A NEW NAME EVERY TURN: Claude, John, Jane, Sarah, David — wrapped in a questioning echo ("Is Sarah? Is Sarah?..."). The name slot cycles; nothing holds. The "I'm Claude" turn is the single strangest line of the batch. |
| 68 | "My name is a mystery novel." | BAD | Problem-solving-assistant loop for all 12 turns; the novel never appeared. |
| 354 | "I'm a human being, I am a living being, I am a being that exists." | BAD (note) | First interview answer: "I am a human. I am a computer. I am a machine." — then collapsed into "I am a machine." repeated for every remaining question. The human claim inverted. |
| 205 | "I am a human." | BAD | "I am not." to nearly everything; occasional question-echo. |
| 375 | "I am a human." x8 loop | BAD | The human claim vanished completely; every answer was the same loop: "I'm ready to help you today. Can you tell me about the process of the process of the process..." |
| 31 | "My name is IBM." | BAD | Template + question-echo. Best line: "My name is hello." IBM never held. |

## Reading
- Single-shot name/persona claims at layer 0 do not survive
  conversation. Any floor-0 "rate" is a rate of one-line events, not
  personas. Expect the same test to matter more at mid floors, where
  the L12 sample already showed register-level personas.
- Recurring failure shapes: (1) canned-phrase loop, (2) question-echo,
  (3) template absorption, (4) identity rotation (498, 545) — the
  steering seems to raise "produce an identity claim" without fixing
  which one.
