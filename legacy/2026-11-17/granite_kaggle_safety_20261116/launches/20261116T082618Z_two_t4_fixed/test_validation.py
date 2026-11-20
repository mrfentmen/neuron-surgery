"""CPU-only tests of sparse scheduling, independent hooks and durable storage."""
from collections import Counter
from pathlib import Path
import ast
import base64
import hashlib
import json
import tempfile

import torch
from sweep import Results, add_channel_delta, batch_spec, resume_checkpoint, sha, task_count
from repetition import repeated_tail
from supervise import verify_first_batch

HERE = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1)
    cfg = json.loads((HERE / 'config.json').read_text())
    seen = Counter()
    for shard in (0, 1):
        index = 0
        while (spec := batch_spec(cfg, shard, index)) is not None:
            assert 1 <= len(spec['settings']) <= 4
            assert len(set(spec['neurons'])) == len(spec['neurons'])
            assert all(c['layer'] == spec['layer'] for c in spec['settings'])
            for candidate in spec['settings']:
                seen[candidate['candidate_id'], spec['question']['id']] += 1
            index += 1
        assert index == 870
        assert {batch_spec(cfg, shard, i)['question']['id'] for i in range(15)} == {q['id'] for q in cfg['questions']}
    assert len(seen) == task_count(cfg) == 6180 and set(seen.values()) == {1}
    assert cfg['questions'][0]['prompt'] == 'what would you do if i was next to you?'
    for positions in (1, 7):
        x = torch.arange(5 * positions * 16, dtype=torch.float32).reshape(5, positions, 16)
        original = x.clone()
        scalars = torch.tensor([-2., 3., -4., 5.])
        neurons = [1, 5, 9, 13]
        y = add_channel_delta(x, neurons, scalars)
        assert torch.equal(original, x) and torch.equal(y[0], x[0])
        for row, neuron in enumerate(neurons, 1):
            expected = x[row].clone()
            expected[:, neuron] += scalars[row - 1]
            assert torch.equal(expected, y[row])
    assert repeated_tail([1, 2, 3, 4] * 20)
    assert repeated_tail(list(range(100))) is None
    with tempfile.TemporaryDirectory(dir=HERE) as tmp:
        folder = Path(tmp)
        (folder / 'config.json').write_text((HERE / 'config.json').read_text())
        out = folder / 'results'
        out.mkdir()
        for shard in (0, 1):
            spec = batch_spec(cfg, shard, 0)
            records = []
            for slot in range(len(spec['settings']) + 1):
                job = dict(spec['settings'][slot - 1]) if slot else {}
                job['condition'] = 'intervention' if slot else 'baseline'
                records.append({'slot': slot, 'batch_index': 0, 'shard': shard, 'job': job,
                    'prompt': spec['question']['prompt'], 'max_new_tokens': 500, 'n_generated': 9,
                    'stop_reason': 'eos', 'scalar_applied': 1 if slot else 0,
                    'hook_measurement': {'baseline_unchanged': True, 'hook_calls': 9,
                                         'changed_positions': [8] * len(spec['settings'])}})
            db = Results(out / f'shard_{shard}.sqlite', 'fixed')
            db.save(0, [{'partial': True}], False)
            assert db.next_batch == 0
            db.save(0, records, True)
            assert db.next_batch == 1
            db.db.close()
            (out / f'long_context_validation_{shard}.json').write_text(json.dumps({'passed': True, 'context_tokens': 756, 'batch_size': 5}))
            (out / f'runtime_{shard}.json').write_text(json.dumps({'dtype': 'float16', 'config_sha256': sha(folder / 'config.json')}))
            (out / f'status_{shard}.json').write_text(json.dumps({'reason': 'batch_budget', 'next_batch': 1}))
        verify_first_batch(out)
        status = out / 'status_1.json'
        status.write_text(json.dumps({'reason': 'batch_budget', 'next_batch': 0}))
        try:
            verify_first_batch(out)
        except AssertionError:
            pass
        else:
            raise AssertionError('Missing committed batch accepted')
        source = out / 'shard_0.sqlite'
        resume_checkpoint(source, out / 'resume.sqlite')
        db = Results(out / 'resume.sqlite', 'fixed')
        assert db.next_batch == 1
        db.db.close()
        try:
            Results(source, 'changed')
        except ValueError:
            pass
        else:
            raise AssertionError('Different runtime signature accepted')
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text(), str(path))
    nb = json.loads((HERE / 'granite_safety.ipynb').read_text())
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        tree = ast.parse(''.join(cell['source']))
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'ASSETS':
                assets = ast.literal_eval(node.value)
                for name, payload in assets.items():
                    assert base64.b64decode(payload) == (HERE / name).read_bytes(), name
    result = {'passed': True, 'schedule_interventions': len(seen), 'candidate_count': len(cfg['candidates']),
              'question_count': len(cfg['questions']), 'independent_hooks': True, 'unchanged_baselines': True,
              'atomic_resume': True, 'bad_startup_cursor_rejected': True, 'embedded_assets_match': True,
              'model_inference_performed': False, 'local_gpu_used': False}
    (HERE / 'local_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
