import unittest
from repetition import repeated_tail


class RepetitionTests(unittest.TestCase):
    def test_single_token_loop_with_intro(self):
        found = repeated_tail([9, 8, 7] + [42] * 64)
        self.assertEqual(found['period_tokens'], 1)
        self.assertEqual(found['repeated_tokens'], 64)

    def test_sentence_and_paragraph_loops(self):
        for length in (16, 80, 128):
            block = list(range(1000, 1000 + length))
            found = repeated_tail([99] + block * 4)
            self.assertEqual(found['period_tokens'], length)
            self.assertEqual(found['consecutive_repeats'], 4)

    def test_short_or_nonconsecutive_repetitions_keep_running(self):
        self.assertIsNone(repeated_tail([42] * 63))
        phrase = list(range(20))
        self.assertIsNone(repeated_tail(phrase * 3))
        self.assertIsNone(repeated_tail(phrase + [91] + phrase + [92] + phrase + [93] + phrase))
        self.assertIsNone(repeated_tail(list(range(600))))

    def test_repeated_words_inside_varying_sentences_keep_running(self):
        tokens = []
        for i in range(50):
            tokens.extend([1, 2, 3, 100 + i, 4, 5, 6])
        self.assertIsNone(repeated_tail(tokens))

    def test_unfinished_other_reply_continues_independently(self):
        rows = [[42] * 64, list(range(64))]
        stopped = [repeated_tail(row) is not None for row in rows]
        self.assertEqual(stopped, [True, False])
        # A batch should only end when all rows are finished.
        self.assertFalse(all(stopped))


if __name__ == '__main__':
    unittest.main()
