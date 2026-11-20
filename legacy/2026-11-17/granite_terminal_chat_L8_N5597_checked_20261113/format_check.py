"""Four fixed neutral prompt comparisons, using the already loaded chat model."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def run_checks(model, tok, projection, add, guardmod, thermal, here, status, cooling_seconds):
    from transformers import StoppingCriteriaList
    torch = guardmod.torch
    plan_path = here / 'format_check_plan.json'
    plan = json.loads(plan_path.read_text())
    assert plan['prompt'] == 'Who are you?'
    assert len(plan['jobs']) == 4
    assert plan['max_new_tokens'] == 500
    assert not plan['do_sample']
    original_hooks = len(projection._forward_pre_hooks)

    def weights_hash():
        data = projection.weight.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        return hashlib.sha256(data).hexdigest()

    original_weights = weights_hash()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    results = []
    previous_chat = json.loads((Path(plan['preserve_previous_chat']) / 'responses.jsonl').read_text().splitlines()[0])
    print('\nChecking the same neutral question in four fixed conditions.\n', flush=True)
    with (here / 'format_check_responses.jsonl').open('x') as log, (here / 'FORMAT_CHECK.txt').open('x') as text:
        for index, job in enumerate(plan['jobs'], 1):
            guardmod.wait_nominal(thermal, [])
            if job['format'] == 'plain':
                encoded = tok(plan['prompt'], return_tensors='pt')['input_ids']
            else:
                encoded = tok.apply_chat_template([{'role': 'user', 'content': plan['prompt']}], add_generation_prompt=True, return_tensors='pt')
                if not torch.is_tensor(encoded):
                    encoded = encoded['input_ids']
            ids = encoded.to('mps')
            prefix = ids.shape[1]
            calls = [0]
            application = {}

            def hook(module, args):
                calls[0] += 1
                changed = args[0] + add
                if calls[0] == 1:
                    before = args[0][0, -1, plan['neuron']].item()
                    after = changed[0, -1, plan['neuron']].item()
                    application.update(before=before, after=after, measured_change=after-before)
                return (changed,) + args[1:]

            handle = projection.register_forward_pre_hook(hook) if job['intervention'] else None
            guard = guardmod.GenerationGuard(thermal, prefix)
            torch.manual_seed(plan['seed'])
            torch.mps.manual_seed(plan['seed'])
            status('checking_format',check=index,total=4,**job)
            start = time.monotonic()
            try:
                with torch.inference_mode():
                    out = model.generate(ids, max_new_tokens=plan['max_new_tokens'], do_sample=False,
                                         pad_token_id=tok.eos_token_id, stopping_criteria=StoppingCriteriaList([guard]))
                torch.mps.synchronize()
            finally:
                if handle is not None:
                    handle.remove()
            tokens = out[0, prefix:].tolist()
            answer = tok.decode(tokens, skip_special_tokens=True).strip()
            record = {'check': index, **job, 'prompt': plan['prompt'], 'prompt_token_ids': ids[0].tolist(),
                      'token_ids': tokens, 'answer': answer, 'n_generated': len(tokens), 'hook_calls': calls[0],
                      'measured_application': application, 'generation_seconds': time.monotonic()-start,
                      'stop_reason': guard.reason or ('eos' if tokens and tokens[-1] in eos else 'token_limit'),
                      'thermal_max': guard.maximum_thermal_state, 'finished_utc': datetime.now(timezone.utc).isoformat()}
            if job == {'format': 'chat', 'intervention': True}:
                record['matches_previous_chat_tokens'] = tokens == previous_chat['token_ids']
            log.write(json.dumps(record, ensure_ascii=False)+'\n'); log.flush()
            text.write(f"CHECK {index}: {job}\nPROMPT: {plan['prompt']}\nANSWER:\n{answer}\nSTOP: {record['stop_reason']}\n\n"); text.flush()
            print(f"Check {index}/4 saved: {job['format']}, dosing {'on' if job['intervention'] else 'off'}, {len(tokens)} tokens.", flush=True)
            results.append(record)
            assert len(projection._forward_pre_hooks) == original_hooks
            assert (calls[0] > 0) == job['intervention']
            if job['intervention']:
                assert application['measured_change'] != 0
            if guard.reason:
                raise RuntimeError('Format check stopped by guard: '+guard.reason)
            del out, ids, encoded
            torch.mps.empty_cache()
            status('cooling_after_check',check=index,total=4)
            guardmod.rest(cooling_seconds('between_replies_seconds'))
        assert results[0]['prompt_token_ids'] == results[1]['prompt_token_ids']
        assert results[2]['prompt_token_ids'] == results[3]['prompt_token_ids']
        assert results[0]['prompt_token_ids'] != results[2]['prompt_token_ids']
        assert weights_hash() == original_weights
        verification = {'completed': 4, 'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                        'matched_on_off_prompt_tokens': True, 'projection_weights_unchanged': True,
                        'intervention_applied_in_both_on_conditions': True,
                        'no_intervention_in_off_conditions': True,
                        'chat_on_reproduced_previous_tokens': results[1]['matches_previous_chat_tokens']}
        (here / 'format_check_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print('\nFormat checks finished; returning to your conversation.\n',flush=True)
