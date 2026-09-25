import unittest

from scripts.collect_duel_v2_teacher_holdout import PAIRS, schedule


class TeacherHoldoutScheduleTest(unittest.TestCase):
    def test_three_old_teacher_pairs_swap_actual_initiative(self):
        jobs = schedule(13_100_000_000, 4)
        self.assertEqual(len(jobs), 24)
        self.assertEqual({(job['left'], job['right']) for job in jobs}, set(PAIRS))
        self.assertEqual({job['first'] for job in jobs}, {'a', 'b'})
        for index in range(0, len(jobs), 2):
            a, b = jobs[index:index + 2]
            self.assertEqual((a['left'], a['right'], a['seed']),
                             (b['left'], b['right'], b['seed']))
            self.assertEqual((a['first'], b['first']), ('a', 'b'))


if __name__ == '__main__':
    unittest.main()
