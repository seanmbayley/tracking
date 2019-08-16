import unittest

from core import subject
from core import NTNode


class TestSubject(unittest.TestCase):
    def setUp(self):
        self.nodes = [(0, 0, 0), (0, 10, 5), (10, 10, 10)]

    def test_add_nodes(self):
        subj = subject.Subject('test')

        for x, y, time in self.nodes:
            subj.add_node(NTNode(x, y, time))

        subj.build_wifi_edges()
        subj.draw()


if __name__ == '__main__':
    unittest.main()