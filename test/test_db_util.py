import unittest
import matplotlib.pyplot as plt
from db import util


class TestDbUtil(unittest.TestCase):
    def setUp(self):
        gd = util.get_grid_dimensions()
        self.sq_km = ((gd['max_x'] - gd['min_x']) / 1000) * ((gd['max_y'] - gd['min_y']) / 1000)

    def test_find_listeners(self):
        for density in [1, 2, 3, 4]:
            listeners = util.find_listeners(density)
            self.assertTrue(len(listeners) < density * self.sq_km)




if __name__ == '__main__':
    unittest.main()
