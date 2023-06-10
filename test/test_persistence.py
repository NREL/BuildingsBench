import unittest
import torch
import numpy as np

class TestPersistence(unittest.TestCase):
    def test_average_persistence(self):
        from buildings_bench.models.persistence import AveragePersistence

        context_len = 168
        pred_len = 24
        bsz = 10
        seqlen = context_len + pred_len
        x = {'load': torch.from_numpy(np.random.rand(bsz, seqlen, 1).astype(np.float32))}
        ap = AveragePersistence(context_len=context_len, pred_len=pred_len)
        y = ap(x)
        y_mean = y[:, :, 0]
        y_sigma = y[:, :, 1]
        self.assertEqual(y_mean.shape, (bsz, pred_len))
        self.assertEqual(y_sigma.shape, (bsz, pred_len))

    def test_copy_last_persistence(self):
        from buildings_bench.models.persistence import CopyLastDayPersistence

        context_len = 168
        pred_len = 24
        bsz = 10
        seqlen = context_len + pred_len
        load = torch.from_numpy(np.random.rand(bsz, seqlen, 1).astype(np.float32))
        last_day = load[:, -48:-24]
        x = {'load': load}
        ap = CopyLastDayPersistence(context_len=context_len, pred_len=pred_len)
        y = ap(x)
        self.assertEqual(y.shape, (bsz, pred_len, 1))
        self.assertTrue( (last_day == y).all() )

    def test_last_week_persistence(self):
        from buildings_bench.models.persistence import CopyLastWeekPersistence

        context_len = 168
        pred_len = 24
        bsz = 10
        seqlen = context_len + pred_len
        load = torch.from_numpy(np.random.rand(bsz, seqlen, 1).astype(np.float32))
        last_week = load[:,0:24]
        x = {'load': load}
        ap = CopyLastWeekPersistence(context_len=context_len, pred_len=pred_len)
        y = ap(x)
        self.assertEqual(y.shape, (bsz, pred_len, 1))
        self.assertTrue( (last_week == y).all() )

def test():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPersistence)
    unittest.TextTestRunner(verbosity=2).run(suite)
