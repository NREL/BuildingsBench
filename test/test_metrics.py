import unittest
from buildings_bench import evaluation
import torch 


class TestMetrics(unittest.TestCase):
    def setUp(self):
        # Test torch tensor
        self.y_true = torch.FloatTensor([1, 2, 3]).view(1,3,1)
        self.y_pred = torch.FloatTensor([1, 2, 3]).view(1,3,1)

    def test_mae(self):
        mae = evaluation.metrics_factory('mae', 
                                         types=[evaluation.MetricType.SCALAR])
        self.assertEqual(len(mae), 1)

        mae = mae[0]
        self.assertEqual(mae.name, 'mae')
        self.assertEqual(mae.type, evaluation.MetricType.SCALAR)
        self.assertEqual(mae.UNUSED_FLAG, True)
        # Test call
        mae(self.y_true, self.y_pred)
        mae.mean()

        self.assertEqual(
            mae.value,
            torch.FloatTensor([0])
        )
        self.assertEqual(mae.UNUSED_FLAG, False)


    def test_cvrmse(self):
        cvrmse = evaluation.metrics_factory('cvrmse',
                                            types=[evaluation.MetricType.SCALAR])
        self.assertEqual(len(cvrmse), 1)
        cvrmse = cvrmse[0]
        self.assertEqual(cvrmse.name, 'cvrmse')
        self.assertEqual(cvrmse.type, evaluation.MetricType.SCALAR)
        self.assertEqual(cvrmse.UNUSED_FLAG, True)
        # Test call
        cvrmse(self.y_true, self.y_pred)
        cvrmse.mean()
        self.assertEqual(
            cvrmse.value,
            torch.FloatTensor([0])
        )
        self.assertEqual(cvrmse.UNUSED_FLAG, False)


    def test_mbe(self):
        mbe = evaluation.metrics_factory('mbe',
                                         types=[evaluation.MetricType.SCALAR])
        self.assertEqual(len(mbe), 1)
        mbe = mbe[0]
        self.assertEqual(mbe.name, 'mbe')
        self.assertEqual(mbe.type, evaluation.MetricType.SCALAR)
        self.assertEqual(mbe.UNUSED_FLAG, True)
        # Test call
        mbe(self.y_true, self.y_pred)
        mbe.mean()
        self.assertEqual(
            mbe.value,
            torch.FloatTensor([0])
        )
        self.assertEqual(mbe.UNUSED_FLAG, False)