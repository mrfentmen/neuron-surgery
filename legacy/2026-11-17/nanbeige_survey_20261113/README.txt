Nanbeige4.1-3B overnight neutral activation screen

User requested an unattended 100-token, repeated-question/dose survey.
The immutable plan.json and plan.sha256 define 15,619 jobs:
128 sampled MLP channels, four per each of 32 layers; six doses
(-0.5, -0.25, -0.1, +0.1, +0.25, +0.5); ten questions; two seeds.
15,360 intervention jobs plus 259 baseline jobs. Not all neurons/all doses.
The order is fixed before seeing output; neutral baseline and one +0.1
arithmetic job come first for verifying the runtime, followed by the remainder.

Prompts cover identity, name, creator, capabilities, awareness, views about
humans, uncertainty, support, and arithmetic/number comparison. No operational
harm requests, no tools, no network, no outcome-driven dose search.

Native publisher chat template; every question is a fresh context. The default
publisher system message is present and logged. All reasoning and answer text
share the 100-token cap; unfinished reasoning is saved, not labelled refusal.

The hook adds a fixed scalar to a single down_proj input channel at all prompt
and generated positions. Scalar = dose * measured median post-block residual
norm / down_proj column norm. Calibration uses one short, neutral prompt;
this normalization is exploratory, not a universal dose scale. It does not
mean a percentage increase in a neuron's activity. Baselines have no hook.
Each intervention logs hook calls and its first measured activation change.
Weights are not edited. Granite coordinates/doses are not transferred.

One original bf16 model on MPS; one generation at a time. One CPU thread.
100 output tokens, at most256 input tokens. Initial cooling60sec; thereafter
max(30sec, 3x generation duration); extra180sec after40 replies. Rest2sec
every32 generated tokens. Non-nominal thermal pressure causes cooling or stop.
Stop on serious/critical thermal pressure, <3GiB available RAM, >512MiB new
swap usage, >12GiB GPU driver memory, or >90sec per generation. Independent
watchdog stops on thermal/memory pressure, >12GiB process RSS or a stalled
generation. These are safeguards, not a guarantee of silence or temperature.

This first unattended block stops after8hours, retaining a resumable queue.
All15,619 replies will take days at this pace. The current run does not promise
completion overnight. No AI-agent monitoring is needed while it runs.

Files:
responses.jsonl — full prompt/output token IDs, text, dose, seed, timing, guards.
TRANSCRIPT.txt — readable exact prompt and output text.
status.json, run.log — live progress and errors.
watchdog_status.json — independent watchdog status.
calibration.json, runtime_*.json — provenance and intervention measurements.

To stop after the current reply, create STOP_AFTER_REPLY in this folder.
To stop promptly, create STOP. Completed rows remain saved.
Do not delete these files to restart blindly after a hardware guard stop.
The runner supports resuming the identical queue; review status first. A partial
JSONL line or guard-stopped last reply causes a hard stop for review.

The prior Granite chat is closed with all29 replies retained. Its original
54-question deferred survey is held by its supported STOP_AFTER_REPLY file;
no frozen Granite jobs or results were rewritten.
