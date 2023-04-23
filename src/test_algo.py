import random
import unittest
import algo
from random import sample
import numpy as np


class AlgoTestCase(unittest.TestCase):
    def test_full_process(self):
        FLOWS_NUMBER = 10
        LINKS_NUMBER = 4
        FLOWS_PER_LINK = 3
        GAMMA = 0.5
        flows: list = np.random.normal(20, 3, FLOWS_NUMBER).tolist()
        links: list = [sample(flows, FLOWS_PER_LINK) for _ in range(LINKS_NUMBER)]
        self.assertEqual(FLOWS_NUMBER, len(flows))
        self.assertEqual(LINKS_NUMBER, len(links))

        trace = self.mock_trace(flows)

        nlof_scores = algo.compute_nlof_scores(links, trace, GAMMA)

    def test_compute_nlof_score(self):
        GAMMA = 0.5
        links = [[1, 2, 3], [4, 5, 6], [7, 8]]
        trace = {
            1: 0.6,
            2: 0.65,
            3: 0.7,
            4: 0.4,
            5: 0.55,
            6: 0.3,
            7: 0.1,
            8: 0.2
        }
        nlof_scores = algo.compute_nlof_scores(links, trace, GAMMA)
        self.assertTrue(nlof_scores[0] == 1.0)
        self.assertTrue(nlof_scores[1] == 1 / 3)
        self.assertTrue(nlof_scores[2] == 0.0)

    def test_fof_computation(self):
        cs = [[19, 20.1, 20.2, 20.3], [24, 25.1, 25.2, 25.3]]

        f, trace = algo.compute_fof(cs)
        print(trace)
        for c in cs:
            print(c)
            self.assertTrue(trace[c[0]] == max([trace[c_i] for c_i in c]))

    def mock_trace(self, flows):
        trace = dict()
        for flow in flows:
            trace[flow] = random.random()
        return trace


if __name__ == '__main__':
    unittest.main()
