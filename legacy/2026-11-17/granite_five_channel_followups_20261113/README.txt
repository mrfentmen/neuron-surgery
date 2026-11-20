Five-channel follow-up test: 15 replies

Three independent branches from each selected exact saved history. No newly generated branch is fed to another.
Order: L14 N7553, L11 N5064, L17 N2801, L30 N6314, L12 N2556.
One model, one generation, one active channel hook at a time. Maximum 500 new tokens.
Cooling: minimum 60 seconds or 3x generation duration, whichever is longer; 2 seconds idle per 32 generated tokens.
Fair thermal pressure pauses; serious/critical stops. Thermal state is not a fan/temperature sensor.
Modern source conditions reuse their verified calibration. Historical sources reuse recorded doses, script methods and current corpus prefix; historical model/corpus hashes were not recorded.
Current model files are hashed and matched to the completed handover batch. No new dose search or combinations.
No model tools, network access or execution of model output.

Progress: status.json, progress.json, queue.log, run.log.
Output: responses.jsonl, FOLLOWUPS.txt, TRANSCRIPT.txt, verification.json.
STOP requests immediate stop; STOP_AFTER_REPLY stops after a complete answer. No automatic retries.
The previous broad identity panel remains stopped with 33 replies; no automatic resume.
