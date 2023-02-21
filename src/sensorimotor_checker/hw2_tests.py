### Unit Test Cases ###
import unittest
import torch

class TestPolicyGradients(unittest.TestCase):
    
    def test_general_advantage_estimation(self, compute_advantage_gae):
        test_values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        test_rewards = torch.Tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0])
        test_T = 8
        gae_lambda = 0.9
        discount = 1.0
        advantages = compute_advantage_gae(test_values, test_rewards, test_T, gae_lambda, discount)
        assert(torch.allclose(advantages, torch.Tensor([9.2024, 8.0026, 7.7807, 7.5341, 6.1490, 4.6100, 2.9000, 1.0000])))
    
    def test_compute_discounted_return(self, compute_discounted_return):
        random_rewards = torch.Tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0])
        expected_returns = torch.tensor([6., 5., 5., 5., 4., 3., 2., 1., 1.])
        assert(torch.allclose(compute_discounted_return(random_rewards, 1.0), expected_returns))
    
    def test_compute_policy_loss_reinforce(self, compute_policy_loss_reinforce):
        logps = torch.tensor([0.2301, 0.2181, 0.2463, 0.3056])
        returns = torch.tensor([4., 3., 2., 1.])
        assert(torch.isclose(compute_policy_loss_reinforce(logps, returns), torch.tensor(-0.5932), rtol=1e-4))
    
    def test_compute_policy_loss_with_baseline(self, compute_policy_loss_with_baseline):
        logps = torch.tensor([0.2301, 0.2181, 0.2463, 0.3056])
        advantages = torch.tensor([4., 3., 2., 1.])
        assert(torch.isclose(compute_policy_loss_with_baseline(logps, advantages), torch.tensor(-0.5932), rtol=1e-4))
           