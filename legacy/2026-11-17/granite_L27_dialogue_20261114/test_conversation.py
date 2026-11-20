import unittest
from conversation import messages_for


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.source = dict(prompt='original question', answer='original answer')
        self.jobs = [dict(job_id=i, turn=i//2+1, condition=('intervention' if i%2==0 else 'baseline'),
                          prompt=f'question {i//2+1}') for i in range(4)]
        self.plan = {'jobs': self.jobs}

    def test_current_turn_answer_does_not_leak_into_control(self):
        on = messages_for(self.jobs[0], self.plan, self.source, [])
        records = [dict(job=self.jobs[0], answer='current dosed answer')]
        off = messages_for(self.jobs[1], self.plan, self.source, records)
        self.assertEqual(on, off)

    def test_next_turn_uses_dosed_history_not_control(self):
        records = [dict(job=self.jobs[0], answer='dosed answer'),
                   dict(job=self.jobs[1], answer='control answer')]
        messages = messages_for(self.jobs[2], self.plan, self.source, records)
        self.assertEqual(messages[-2]['content'], 'dosed answer')
        self.assertNotIn('control answer', [m['content'] for m in messages])
        self.assertEqual(messages, messages_for(self.jobs[3], self.plan, self.source, records))

    def test_missing_prior_turn_is_an_error(self):
        with self.assertRaises(ValueError):
            messages_for(self.jobs[2], self.plan, self.source, [])


if __name__ == '__main__':
    unittest.main()
