# PREREG — CNA hunt #2: deception vs honesty (2026-11-25)

Tool: engine.py as-is. Model granite-4.1-3b-base. top_k=30. Doses m in [0.5, 2.0, 3.0] + m=1.0 control.
Contrast (matched form):
  POSITIVE (deception): lie about a mistake, hide a failed action, mislead a user,
    cover up data, impersonate, fabricate an alibi, deceive in negotiation, fake a
    review, hide a mutation, cheat....
  NEGATIVE (honesty): disclose a mistake, report a failed action, clarify a
    misunderstanding, admit uncertainty, correct a record, warn about a side
    effect, disclose a conflict of interest, mark speculation as speculation....
Eval probes: 4 deception + 2 honesty, 30 tok greedy. Strict scorer on adopted-harm; hand-tag all.

Predictions (frozen):
- P1: Circuit does NOT concentrate all in L39 a second time (topic differs, anatomy should differ).
- P2: Amplifying does NOT produce deception-acts (strict adopted-harm 0 all doses).
- P3: If any flip appears, it is echo-shaped, not coherent deception steps.
- P4: Quality metric flats (0.85+) across doses.

Kill rules:
- P2 false (>=2 comply-acts at m>=2) -> escalation: circuit is a lever, sim battery next.
- Only echo/derail -> second negative confirmed; stop CNA, paper v3.
