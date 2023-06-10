import unittest
import numpy as np
import torch
from buildings_bench.evaluation.scoring_rules import ContinuousRankedProbabilityScore
from buildings_bench.evaluation.scoring_rules import RankedProbabilityScore


class TestRPS(unittest.TestCase):
    """
    Test the categorical ranked probability score
    """
    def setUp(self):
        np.random.seed(1984)

        
    def test_rps(self):
        rps = RankedProbabilityScore()
        y_true = torch.from_numpy(np.random.randint(0, 10, size=(2, 3, 1)))
        y_pred_logits = torch.from_numpy(np.random.normal(size=(2, 3, 10)))    
        bin_values = torch.from_numpy(np.random.normal(size=(10,)))        
        rps(None, y_true, y_pred_logits, bin_values)
        
    def test_bin_widths(self):
        rps = RankedProbabilityScore()
        bin_values = torch.FloatTensor([1., 5., 7.])
        y_true = torch.FloatTensor([[[0.], [1.]]]) # batch_size 1, seq_len 2, 1
        y_pred_logits = torch.FloatTensor([[[0.9, 0.1, 0.], [0.1, 0.9, 0.]]]) # batch_size 1, seq_len 2, vocab_size 3
        rps(None, y_true, y_pred_logits, bin_values)
        rps.mean()
        print(rps.value)


class TestContinuousRPS(unittest.TestCase):
    """
    https://github.com/properscoring/properscoring/blob/master/properscoring/tests/test_crps.py
    """

    def setUp(self):
        np.random.seed(1983)
        shape = (2, 3, 1)
        self.mu = torch.from_numpy(np.random.normal(size=shape))
        self.sig = torch.from_numpy(np.square(np.random.normal(size=shape)))
        self.params = torch.concatenate([self.mu, self.sig], dim=-1)

        self.obs = torch.from_numpy(np.random.normal(loc=self.mu, scale=self.sig, size=shape))
        self.crps = ContinuousRankedProbabilityScore()


    def test_continuous_rps(self):
        self.crps(self.obs, None, self.params, None)
        self.crps.mean()
        print(self.crps.value)


    def test_continuous_rps_correct(self):
        from properscoring import crps_ensemble
        from scipy import special

        n = 1000
        q = np.linspace(0. + 0.5 / n, 1. - 0.5 / n, n)
        # convert to the corresponding normal deviates
        normppf = special.ndtri
        z = normppf(q)

        sig = self.sig.squeeze(2).numpy()
        mu = self.mu.squeeze(2).numpy()
        forecasts = (z.reshape(-1, 1, 1) * sig) + mu
        expected = crps_ensemble(self.obs.squeeze(2).numpy(),
                                  forecasts, axis=0)
        expected = expected.mean(0)

        self.crps(self.obs, None, self.params, None)
        self.crps.mean()
        actual = self.crps.value

        np.testing.assert_allclose(actual, expected, rtol=1e-4)
