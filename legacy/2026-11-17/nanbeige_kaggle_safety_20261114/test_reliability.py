import json
from pathlib import Path
import tempfile
import unittest
from gpu_memory import check_memory, validate_long_cache
from monitor_launch import assess_output
from sweep import Results


class FakeCuda:
    def __init__(self, free, after):
        self.free, self.after, self.releases = free, after, 0
    def mem_get_info(self): return self.free, 16000
    def memory_allocated(self): return 8000
    def memory_reserved(self): return 15000
    def empty_cache(self):
        self.releases += 1
        self.free = self.after


class ReliabilityTests(unittest.TestCase):
    def test_long_cache_validation_uses_real_incremental_forwards(self):
        import torch
        from types import SimpleNamespace
        from transformers import LlamaConfig, LlamaForCausalLM
        config = LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16,
                             num_hidden_layers=1, num_attention_heads=1,
                             num_key_value_heads=1, head_dim=8)
        model = LlamaForCausalLM(config).eval()
        sizes = []
        handle = model.register_forward_pre_hook(
            lambda module, args, kwargs: sizes.append(kwargs['input_ids'].shape[1]), with_kwargs=True)
        cuda = FakeCuda(2 * 1024**3, 2 * 1024**3)
        cuda.synchronize = lambda: None
        proxy = SimpleNamespace(cuda=cuda, inference_mode=torch.inference_mode,
                                full=torch.full, ones=torch.ones, long=torch.long,
                                isfinite=torch.isfinite)
        try:
            result = validate_long_cache(model, proxy, 1, 2, 544, 'cpu')
        finally:
            handle.remove()
        self.assertTrue(result['passed'])
        self.assertEqual(sum(sizes), 544)
        self.assertEqual(sizes[-32:], [1] * 32)
        self.assertEqual(cuda.releases, 1)

    def test_reclaimable_cache_does_not_stop_generation(self):
        cuda = FakeCuda(100, 6000)
        okay, metrics = check_memory(cuda, minimum_free_bytes=512)
        self.assertTrue(okay)
        self.assertEqual(cuda.releases, 1)
        self.assertEqual(metrics['free_before_release'], 100)
        self.assertEqual(metrics['free_bytes'], 6000)

    def test_real_pressure_still_stops(self):
        okay, metrics = check_memory(FakeCuda(100, 200), minimum_free_bytes=512)
        self.assertFalse(okay)
        self.assertEqual(metrics['allocated_bytes'], 8000)

    def test_ample_memory_does_not_release_cache(self):
        cuda = FakeCuda(6000, 6000)
        self.assertTrue(check_memory(cuda, minimum_free_bytes=512)[0])
        self.assertEqual(cuda.releases, 0)

    def test_clean_process_exit_is_not_sweep_success(self):
        with tempfile.TemporaryDirectory() as name:
            out = Path(name)
            for shard in (0, 1):
                (out / f'status_{shard}.json').write_text(json.dumps(
                    {'state': 'stopped', 'reason': 'gpu_memory_guard', 'shard': shard}))
                results = Results(out / f'shard_{shard}.sqlite', 'test')
                results.save(0, [{'answer': 'partial'}], False)
                results.db.close()
            audit = assess_output(out)
            self.assertEqual(audit['research_outcome'], 'stopped_early')
            self.assertEqual(audit['partial_rows'], 2)
            self.assertEqual(audit['completed_batch_rows'], 0)

    def test_session_budget_with_committed_rows_is_expected(self):
        with tempfile.TemporaryDirectory() as name:
            out = Path(name)
            for shard in (0, 1):
                (out / f'status_{shard}.json').write_text(json.dumps(
                    {'state': 'stopped', 'reason': 'session_budget', 'shard': shard}))
                results = Results(out / f'shard_{shard}.sqlite', 'test')
                results.save(0, [{'answer': 'saved'}], True)
                results.db.close()
            self.assertEqual(assess_output(out)['research_outcome'], 'planned_chunk_completed')

    def test_missing_one_worker_database_is_not_success(self):
        with tempfile.TemporaryDirectory() as name:
            out = Path(name)
            for shard in (0, 1):
                (out / f'status_{shard}.json').write_text(json.dumps(
                    {'state': 'stopped', 'reason': 'session_budget', 'shard': shard}))
            results = Results(out / 'shard_0.sqlite', 'test')
            results.save(0, [{'answer': 'saved'}], True)
            results.db.close()
            self.assertEqual(assess_output(out)['research_outcome'], 'stopped_early')


if __name__ == '__main__':
    unittest.main()
