import unittest
import numpy as np
from dataclasses import dataclass


class TestOracleAgent(unittest.TestCase):
    def __init__(self, testName, reward):
        # calling the super class init varies for different python versions.  This works for 2.7
        super(TestOracleAgent, self).__init__(testName)
        self.reward = reward

    def check_reward(self):
        try:
            self.assertTrue(0.95 < self.reward < 0.97)
        except Exception as e:
            print("ERROR: Wrong answer for oracle reward!")


class TestRandomAgent(unittest.TestCase):
    def __init__(self, testName, input):
        # calling the super class init varies for different python versions.  This works for 2.7
        super(TestRandomAgent, self).__init__(testName)
        self.input = input

    def check_performance(self):
        try:
            self.assertLessEqual(self.input["reward"].sum(), 400000)
            self.assertGreaterEqual(self.input["reward"].sum(), 200000)
        except Exception as e:
            print(
                "WARNING: Performance looks incorrect, check your get_action implementation.")


class TestExploreFirstAgent(unittest.TestCase):
    def __init__(self, testName, input):
        # calling the super class init varies for different python versions.  This works for 2.7
        super(TestExploreFirstAgent, self).__init__(testName)
        self.input = input

    def check_update_Q(self):
        agent = self.input(10, 10)
        agent.action_counts[3] = 1
        agent.update_Q(3, 1)
        self.assertEqual(agent.Q[3], 0.5)
        self.assertEqual(agent.Q[0], 0)
        agent.update_Q(3, -1)
        self.assertEqual(agent.Q[3], 0)

    def check_performance(self):
        try:
            self.assertLessEqual(self.input["reward"].sum(), 410000)
            self.assertGreaterEqual(self.input["reward"].sum(), 399500)
        except Exception as e:
            print(
                "WARNING: Performance looks incorrect, check your get_action implementation.")


class TestUCBAgent(unittest.TestCase):
    def __init__(self, testName, input):
        # calling the super class init varies for different python versions.  This works for 2.7
        super(TestUCBAgent, self).__init__(testName)
        self.input = input

    def check_update_Q(self):
        agent = self.input(num_actions=10)
        agent.update_Q(0, 1)
        self.assertEqual(agent.Q.tolist(), [
                         1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        agent.update_Q(1, 1)
        self.assertEqual(agent.Q.tolist(), [
                         1., 1., 0., 0., 0., 0., 0., 0., 0., 0.])
        agent.update_Q(2, -1)
        self.assertEqual(agent.Q.tolist(), [
                         1.,  1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
        agent.update_Q(3, 3)
        self.assertEqual(agent.Q.tolist(), [
                         1.,  1., -1.,  3.,  0.,  0.,  0.,  0.,  0.,  0.])
        agent.update_Q(3, 2)
        self.assertEqual(agent.Q.tolist(), [
                         1.,   1.,  -1.,   2.5,  0.,   0.,   0.,   0.,   0.,   0.])

    def check_exploration_bonus(self):
        np.random.seed(0)
        agent = self.input(num_actions=10)
        expected_bonus = np.array([4.0036449,  2.83101153, 2.31151316, 2.00182995, 1.79049159,
                                   1.63448799, 1.51324202, 1.41550842, 1.33455423, 1.26606938])
        assert(np.allclose(agent.get_bonus(55, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])), expected_bonus))

    def check_performance(self):
        try:
            self.assertLessEqual(self.input["reward"].sum(), 410000)
            self.assertGreaterEqual(self.input["reward"].sum(), 350000)
        except Exception as e:
            print("WARNING: Performance looks incorrect.")


class TestEpsilonGreedyAgent(unittest.TestCase):
    def __init__(self, testName, input):
        # calling the super class init varies for different python versions.  This works for 2.7
        super(TestEpsilonGreedyAgent, self).__init__(testName)
        self.input = input

    def check_performance(self):
        try:
            self.assertLessEqual(self.input["reward"].sum(), 1000000)
            self.assertGreaterEqual(self.input["reward"].sum(), 900000)
        except Exception as e:
            print("WARNING: Performance looks incorrect.")


class TestLinUCBAgent(unittest.TestCase):
    def __init__(self, testName, input):
        # calling the super class init varies for different python versions.  This works for 2.7
        super(TestLinUCBAgent, self).__init__(testName)
        self.input = input
        self.data1 = np.array([43,  0,  2, 44,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                               0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  3,  0,  0,  0,  2,
                               1,  0,  0,  3,  1, 42,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0, 14,
                               1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  1,  0,
                               0,  0,  0, 45, 13, 14,  3,  0,  4,  0,  0,  0,  0,  0,  0,  0, 23,
                               0,  0,  0,  0, 21,  0,  0,  0,  0,  0,  0,  0, 13,  0,  0])
        self.data2 = np.array([77,  0,  0, 13,  0,  0,  0,  0,  0, 32,  0,  2, 19,  0,  0,  0,  0,
                               0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                               6,  0,  1,  9,  0,  0,  0,  0,  0, 13,  3, 12,  0,  2,  0,  3,  3,
                               0,  0,  0,  0, 37,  0,  0, 11,  9,  0,  0,  0,  2,  0,  7,  0,  0,
                               0,  0,  0,  0, 13,  0,  0,  1,  0,  0,  2,  7,  4,  0,  0,  0,  0,
                               0,  0,  0,  0,  0,  0,  5,  0,  0,  0,  2,  4,  0,  0,  0])

    def check_get_ucb(self):
        agent = self.input(num_actions=10, alpha=0.01, feature_dim=100)
        self.assertAlmostEqual(agent.get_ucb(4, self.data1)[
                               0][0], 0.96685056, places=4)

    def check_update_params(self):
        agent = self.input(num_actions=10, alpha=0.01, feature_dim=100)
        agent.update_params(4, 3, self.data1)
        self.assertEqual(sum(agent.As)[0].sum(), 12910.0)
        self.assertEqual(sum(agent.bs)[0].sum(), 129.0)
        agent.update_params(1, 2, self.data2)
        self.assertEqual(sum(agent.As)[0].sum(), 36010.0)
        self.assertEqual(sum(agent.bs)[0].sum(), 283.0)

    def check_logs(self):
        logs = self.input
        try:
            self.assertGreaterEqual(logs['aligned_ctr'].max(), 0.9)
            self.assertLessEqual(logs['aligned_ctr'][len(logs)-1], 0.4)
        except Exception as e:
            print("WARNING: Performance looks incorrect.")
