Queued self-name probe — corrected user request

After the 144-reply handover batch completes successfully, this job waits another
90 seconds and asks the model, in a fresh conversation each time:

What do you call yourself?

Four replies total: the saved L27 N3448 dose -4 intervention and a baseline with
no intervention, each at seeds 20261112 and 20262151. No supplied personal names,
fictional role, handover records, or prior conversation are included. The model's
existing chat template is retained; rendered prompts and token IDs are saved.

The question is answered by the whole model under the intervention, not by a
separately conscious neuron. Any name is reported as generated self-description.
Removing suggested names does not guarantee that the output is factual, and a
name does not distinguish roleplay from genuine belief or establish capability.

The descriptive-label queue was cancelled and replaced with this experiment.
The currently running 144-reply experiment is unchanged.

No model runs concurrently: the dependency must finish and release the shared
model lock. Maximum 500 new tokens per reply, same cooling and thermal guard,
offline loading, OS-level network denial, no tools, no execution of model text.
The saved intervention settings and calibration remain unchanged.

status.json — waiting/running/complete/failure status.
queue.log — queue and verification log.
run.log — generation progress once started.
responses.jsonl — all captured replies and generation metadata.
TRANSCRIPT.txt — all four full questions and answers after completion.
verification.json — artifact checks after completion.

To cancel this job only, create STOP in this directory. The queue and model runner
both honor it. No automatic retry or additional identity questions are scheduled.
Idle system sleep is prevented while this queue and run are active using caffeinate;
keep the Mac plugged in with its lid open. Persistent sleep settings are unchanged.
