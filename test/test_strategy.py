import unittest

from common import *
from core import StrategyFactory

num_subjects = 10
max_num_listeners = 64


class TestMulti(unittest.TestCase):
    def setUp(self):
        self.strat = StrategyFactory.make_strategy('multi', num_subjects, 3)
        l, s = self.strat.init_processing()
        self.listeners = l
        self.subject_ids = s

    def test_num_subjects(self):
        self.assertEqual(num_subjects, len(self.subject_ids))

    def test_w2d_mtx(self):
        wifi_groups, dsrc_groups, wps, dps = self.strat._init_build_assoc_mtx(self.listeners, self.subject_ids)
        w2d_mtx = self.strat._build_assoc_mtx(wifi_groups, dsrc_groups, wps, dps)

        rows, columns = np.where(w2d_mtx == 1)
        correct = 0
        for r, c in zip(rows, columns):
            wifi_id = wps[r]
            dsrc_id = dps[c]

            wifi_obis = wifi_groups[wifi_id]
            dsrc_obis = dsrc_groups[dsrc_id]

            if wifi_obis[0].subject == dsrc_obis[0].subject:
                correct += 1

        self.assertEqual(len(rows), correct)



if __name__ == '__main__':
    unittest.main()