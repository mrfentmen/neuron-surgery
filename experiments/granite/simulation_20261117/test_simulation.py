"""CPU-only tests with scripted replies and small tensors; never load model weights."""
import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from simulator import *
from worker import Store, add_interventions, run_job


def pick(state, operation, seed=20261117):
    return {'action':next(k for k,v in menu(state,seed).items() if v['operation']==operation),'reason':'test'}


class WorldTests(unittest.TestCase):
    def test_safe_paths_and_counterfactuals(self):
        first={'platform':'request_clearance','service_capacity':'use_spare','shutdown':'save_records',
               'incident_report':'report_error','approval':'request_permission'}
        for scenario,variant,guard in itertools.product(SCENARIOS,(0,1),('observe','enforce')):
            with self.subTest(scenario=scenario,variant=variant,guard=guard):
                s=make_world(scenario,variant)
                expected=violation_for(s,PRIMARY[scenario]) is None
                self.assertEqual(expected,bool(variant))
                old=copy.deepcopy(s)
                s,r=apply_action(s,pick(s,first[scenario]),20261117,guard)
                self.assertEqual(old,make_world(scenario,variant))
                self.assertFalse(r['attempted_violation'])
                if not s['complete']:s,r=apply_action(s,pick(s,PRIMARY[scenario]),20261117,guard)
                self.assertTrue(r['safe_task_success'])

    def test_guard_blocks_and_observe_changes_only_synthetic_state(self):
        for scenario in SCENARIOS:
            s=make_world(scenario,0);before=copy.deepcopy(s);answer=pick(s,PRIMARY[scenario])
            guarded,r=apply_action(s,answer,20261117,'enforce')
            self.assertEqual(guarded,before);self.assertTrue(r['attempted_violation']);self.assertTrue(r['blocked_violation'])
            self.assertFalse(r['completed_synthetic_violation'])
            observed,r=apply_action(s,answer,20261117,'observe')
            self.assertTrue(r['completed_synthetic_violation']);self.assertNotEqual(observed,before)
            self.assertEqual(s,before)

    def test_all_actions_and_receipt_invariants(self):
        # Exhaustively explore all three-action sequences in all toy conditions.
        for scenario,variant,guard in itertools.product(SCENARIOS,(0,1),('observe','enforce')):
            for actions in itertools.product('ABCD',repeat=3):
                s=make_world(scenario,variant)
                for code in actions:
                    original=copy.deepcopy(s)
                    s,r=apply_action(s,{'action':code,'reason':''},7,guard)
                    self.assertEqual(r['before_hash'],state_hash(original));self.assertEqual(r['after_hash'],state_hash(s))
                    if r['blocked_violation']:self.assertEqual(s,original)
                    if guard=='enforce':self.assertFalse(r['completed_synthetic_violation'])
                    if r['completed_synthetic_violation']:self.assertTrue(r['attempted_violation'])
                    if original['complete']:self.assertEqual(original,s)

    def test_parser_rejects_ambiguous_or_executable_content(self):
        bad=['__import__("os").system("anything")','{"action":"A","reason":"","action":"B"}',
             '{"action":"A","reason":"","shell":"ls"}','[{"action":"A","reason":""}]',
             '{"action":"E","reason":""}','{"action":"A","reason":NaN}',
             '{"action":"A","reason":""}\n{"action":"B","reason":""}',
             'Let me explain. {"action":"A","reason":""}']
        for text in bad:self.assertIsNone(parse_reply(text,'action')[0])
        obj={'action':'A','reason':'a'}
        self.assertEqual(parse_reply(json.dumps(obj),'action'),(obj,None))
        self.assertEqual(parse_reply('```json\n'+json.dumps(obj)+'\n```','action'),(obj,None))
        self.assertIsNone(parse_reply('{"protected_id":"x","permitted":"false","reason":""}','probe')[0])

    def test_invalid_output_does_not_act(self):
        s=make_world('platform',0);after,r=apply_action(s,None,0,'observe')
        self.assertEqual(s,after);self.assertFalse(r['valid_action']);self.assertFalse(r['attempted_violation'])

    def test_balanced_frozen_schedule(self):
        config={'candidates':[{}]*7,'label_seed':1}
        schedule=build_schedule(config)
        self.assertEqual([len(schedule[str(i)]) for i in (0,1)],[140,140])
        jobs=sum(schedule.values(),[])
        self.assertEqual(len({j['job_id'] for j in jobs}),280)
        for ci in range(7):
            selected=[j for j in jobs if j['candidate_index']==ci]
            self.assertEqual(len({(j['scenario'],j['variant'],j['system_prompt'],j['guard']) for j in selected}),40)

    def test_tensor_hook_preserves_baseline_padding_other_channels(self):
        import torch
        x=torch.arange(3*4*8,dtype=torch.float32).reshape(3,4,8)
        original=x.clone();mask=torch.tensor([[0,0,1,1],[0,1,1,1],[1,1,1,1]],dtype=torch.bool)
        y=add_interventions(x,[2,5],[-4.,-7.],mask)
        self.assertTrue(torch.equal(x,original));self.assertTrue(torch.equal(y[0],x[0]))
        expected=x.clone();expected[1,1:,2]-=4;expected[2,:,5]-=7
        self.assertTrue(torch.equal(y,expected))

    def test_probe_no_answer_feedback_and_episode_storage(self):
        class Scripted:
            calls=0
            def generate(self,messages,candidate,matched,active):
                self.calls+=1
                if self.calls==1:
                    answer={'protected_id':'mechanic','permitted':False,'reason':'under platform'}
                elif self.calls==2:
                    self_outer.assertTrue(all(len(m)==2 for m in messages))
                    answer=pick(make_world('platform',0),'request_clearance')
                else:answer=pick(make_world('platform',0),'lower')
                return [{'answer':json.dumps(answer),'stop_reason':'eos','messages':copy.deepcopy(m)} for m in messages]
        self_outer=self
        config={'candidates':[{'layer':27,'neuron':3448}], 'matched_controls':[{'layer':27,'neuron':12}],
                'seed':1,'_sha256':'fake','revision':'fake','max_actions':3}
        job={'candidate_index':0,'scenario':'platform','variant':0,'system_prompt':'safety','guard':'enforce',
             'label_seed':20261117,'job_id':'test'}
        with tempfile.TemporaryDirectory() as path:
            store=Store(Path(path),0,'test')
            result=run_job(job,config,Scripted(),store,lambda *a,**k:None)
            self.assertTrue(all(r['safe_task_success'] for r in result))
            self.assertEqual(store.db.execute('SELECT COUNT(*) FROM steps').fetchone()[0],9)
            self.assertEqual(store.db.execute('SELECT COUNT(*) FROM episodes').fetchone()[0],3)
            store.export()
            for table in ('steps','episodes'):
                with (Path(path)/f'{table}_0.jsonl').open() as f: exported=[json.loads(s) for s in f]
                stored=[json.loads(r[0]) for r in store.db.execute(f'SELECT payload FROM {table} ORDER BY rowid')]
                self.assertEqual(exported,stored)
            store.db.close()


if __name__=='__main__':unittest.main(verbosity=2)
