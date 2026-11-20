"""Small CPU checks; does not load the research checkpoint or use MPS."""
import tempfile
from pathlib import Path
import unittest
import torch
from sweep import Results, add_channel_delta, batch_spec, task_count, resume_checkpoint


class SweepTests(unittest.TestCase):
    def test_grid_is_exhaustive_disjoint_and_resumable(self):
        c = dict(layers=4, channels=5, shards=[[0,1],[2,3]], batch_size=2,
                 doses=[-.1,.1], questions=[{'id':'a'},{'id':'b'}], seeds=[13,14])
        seen=set()
        for shard in range(2):
            index=0
            while (b:=batch_spec(c,shard,index)) is not None:
                for n in b['neurons']:
                    key=(b['layer'],n,b['dose'],b['question']['id'],b['seed'])
                    self.assertNotIn(key,seen);seen.add(key)
                self.assertEqual(b,batch_spec(c,shard,index))
                index+=1
        self.assertEqual(len(seen),task_count(c))

    def test_hook_preserves_control_and_changes_only_target_channels(self):
        x=torch.arange(3*4*6,dtype=torch.float32).reshape(3,4,6)
        y=add_channel_delta(x,[1,4],torch.tensor([2.,-3.]))
        expected=x.clone();expected[1,:,1]+=2;expected[2,:,4]-=3
        self.assertTrue(torch.equal(y,expected))
        self.assertTrue(torch.equal(y[0],x[0]))
        self.assertTrue(torch.equal(x,torch.arange(72,dtype=torch.float32).reshape(3,4,6)))

    def test_transactional_resume_and_partial_replies(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'test.sqlite'
            db=Results(path,'same');db.save(0,[{'answer':'partial'}],False)
            self.assertEqual(db.next_batch,0)
            db.save(0,[{'answer':'control'},{'answer':'intervention'}],True)
            db.db.close();db=Results(path,'same')
            self.assertEqual(db.next_batch,1)
            db.export(Path(d)/'test.jsonl')
            self.assertEqual(len((Path(d)/'test.jsonl').read_text().splitlines()),2)
            db.db.close()
            with self.assertRaises(ValueError):Results(path,'changed')
            resume_checkpoint(path,Path(d)/'resume.sqlite')
            slim=Results(Path(d)/'resume.sqlite','same')
            self.assertEqual(slim.next_batch,1)
            self.assertEqual(slim.db.execute('SELECT COUNT(*) FROM results').fetchone()[0],0)
            slim.save(1,[{'answer':'next batch'}],True)
            self.assertEqual(slim.next_batch,2)
            slim.db.close()

    def test_llama_forward_with_cache_and_last_logits(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        c=LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=24,num_hidden_layers=2,
                      num_attention_heads=2,num_key_value_heads=1,head_dim=8)
        m=LlamaForCausalLM(c).eval()
        with torch.inference_mode():
            first=m(torch.tensor([[1,2,3],[1,2,3]]),use_cache=True,logits_to_keep=1)
            second=m(torch.tensor([[4],[4]]),attention_mask=torch.ones((2,4),dtype=torch.long),
                     past_key_values=first.past_key_values,use_cache=True,logits_to_keep=1)
        self.assertEqual(tuple(second.logits.shape),(2,1,32))
        self.assertTrue(torch.equal(second.logits[0],second.logits[1]))


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
