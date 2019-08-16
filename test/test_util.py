import util
import unittest
import numpy as np


class TestUtil(unittest.TestCase):
    def setUp(self):
        self.obi1 = util.make_obi(0, 0, list(range(6, 10)))
        self.obi2 = util.make_obi(0, 0, list(range(5, 11)))
        self.obi3 = util.make_obi(0, 0, list(range(0, 6)))
        self.obi4 = util.make_obi(0, 0, list(range(0, 4)))
        self.obi5 = util.make_obi(0, 2, list(range(5, 11)))

        self.obi_a = util.make_obi(1740, 1830, [3293.9, 3294.8])
        self.obi_b = util.make_obi(1740, 1830, [3204.0, 3233.0, 3256.0, 3289.1])

    def test_cartesian(self):
        x = [0, 1, 5]
        y = [0, 1, 10, 12]
        cart = util.cartesian((x, y))
        self.assertEqual(len(x) * len(y), len(cart),
                         msg='cartesian does not produce correct number of combinations')

    def test_overlap(self):
        self.assertTrue(util.obi_time_intersect(self.obi1, self.obi2),
                        msg='obi overlap should evaluate to True')
        self.assertFalse(util.obi_time_intersect(self.obi1, self.obi4),
                         msg='obi overlap should evaluate to False')

    def test_associated(self):
        self.assertTrue(util.is_obi_pair_assoc(self.obi1, self.obi2),
                        msg='obi pair should be associated')
        self.assertFalse(util.is_obi_pair_assoc(self.obi1, self.obi4),
                         msg='obi pair not associated because not overlapping')
        self.assertFalse(util.is_obi_pair_assoc(self.obi1, self.obi5),
                         msg='obi pair not associated because not at same loc')
        self.assertFalse(util.is_obi_pair_assoc(self.obi_a, self.obi_b),
                         msg='obi overlap should evaluate to False')


    def test_graham_scan(self):
        l = []
        temp = [(732, 590), (415, 360), (276, 276), (229, 544), (299, 95)]
        for x, y in temp:
            d = {}
            d['x'] = x
            d['y'] = y
            l.append(d)

        # print("Case1")
        # print("Before: " + str(l))

        srt_pts = util.sort_points(l)
        # print("Sorted points: " + str(srt_pts))

        c_hull = util.graham_scan(l)
        # print("Convex Hull: " + str(c_hull))

if __name__ == '__main__':
    unittest.main()
