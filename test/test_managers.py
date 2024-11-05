import unittest
from buildings_bench.evaluation.managers import MetricsManager, DatasetMetricsManager
from buildings_bench.evaluation import metrics_factory
from buildings_bench.evaluation import scoring_rule_factory
from buildings_bench import BuildingTypes
import torch


class TestMetricsManager(unittest.TestCase):
       
        
    def test_create_dataset_metrics_manager_single_metric(self):
        metrics_manager = MetricsManager(
            metrics=metrics_factory('mae')
        )

        self.assertEqual(len(metrics_manager.metrics[BuildingTypes.RESIDENTIAL]), 1)
        self.assertEqual(len(metrics_manager.metrics[BuildingTypes.COMMERCIAL]), 1)
        self.assertEqual(metrics_manager.metrics[BuildingTypes.RESIDENTIAL][0].name, 'mae')

    def test_create_dataset_metrics_manager_multiple_metrics(self):
        metrics_manager = MetricsManager(
            metrics=metrics_factory('mae') + metrics_factory('rmse')
        )

        self.assertEqual(len(metrics_manager.metrics[BuildingTypes.RESIDENTIAL]), 2)
        self.assertEqual(len(metrics_manager.metrics[BuildingTypes.COMMERCIAL]), 2)
        self.assertEqual(metrics_manager.metrics[BuildingTypes.RESIDENTIAL][0].name, 'mae')
        self.assertEqual(metrics_manager.metrics[BuildingTypes.RESIDENTIAL][1].name, 'rmse')
    
    def test_create_dataset_metrics_manager_with_scoring_rule(self):
        metrics_manager = MetricsManager(
            scoring_rule=scoring_rule_factory('rps')
        )

        self.assertEqual(len(metrics_manager.scoring_rules), 2)
        self.assertEqual(metrics_manager.scoring_rules[BuildingTypes.RESIDENTIAL].name, 'rps')
        self.assertEqual(metrics_manager.scoring_rules[BuildingTypes.COMMERCIAL].name, 'rps')

    def test_create_dataset_metrics_manager_with_scoring_rule_and_metrics(self):
        metrics_manager = MetricsManager(
            metrics=metrics_factory('mae'),
            scoring_rule=scoring_rule_factory('rps')
        )

        self.assertEqual(len(metrics_manager.metrics[BuildingTypes.RESIDENTIAL]), 1)
        self.assertEqual(len(metrics_manager.metrics[BuildingTypes.COMMERCIAL]), 1)
        self.assertEqual(metrics_manager.metrics[BuildingTypes.RESIDENTIAL][0].name, 'mae')
        self.assertEqual(metrics_manager.scoring_rules[BuildingTypes.RESIDENTIAL].name, 'rps')
        self.assertEqual(metrics_manager.scoring_rules[BuildingTypes.COMMERCIAL].name, 'rps')

    def test_call_dataset_metrics_manager(self):

        metrics_manager = MetricsManager(
            metrics=metrics_factory('mae')
        )

        metrics_manager(
            y_true=torch.FloatTensor([1, 2, 3]).view(1,3,1),
            y_pred=torch.FloatTensor([1, 2, 3]).view(1,3,1),
            building_types_mask=torch.BoolTensor([False])
        )

        self.assertEqual(metrics_manager.metrics[BuildingTypes.RESIDENTIAL][0].name, 'mae')
        metrics_manager.metrics[BuildingTypes.RESIDENTIAL][0].mean()
        self.assertEqual(metrics_manager.metrics[BuildingTypes.RESIDENTIAL][0].value, 0)

    def test_update_loss_and_get_ppl_from_manager(self):

        metrics_manager = MetricsManager(
            metrics=metrics_factory('mae')
        )

        loss = torch.FloatTensor([1.0])
        metrics_manager(
            y_true=torch.FloatTensor([1, 2, 3]).view(1,3,1),
            y_pred=torch.FloatTensor([1, 2, 3]).view(1,3,1),
            building_types_mask=torch.BoolTensor([False]),
            loss=loss,
        )

        self.assertEqual(metrics_manager.metrics[BuildingTypes.RESIDENTIAL][0].name, 'mae')
        metrics_manager.metrics[BuildingTypes.RESIDENTIAL][0].mean()
        self.assertEqual(metrics_manager.metrics[BuildingTypes.RESIDENTIAL][0].value, 0)
        self.assertEqual(metrics_manager.get_ppl(), torch.exp( loss * 3 ) / 3)


    def test_summary(self):

        metrics_manager = MetricsManager(
            metrics=metrics_factory('mae')
        )

        loss = torch.FloatTensor([1.0])
        metrics_manager(
            y_true=torch.FloatTensor([1, 2, 3]).view(1,3,1),
            y_pred=torch.FloatTensor([1, 2, 3]).view(1,3,1),
            building_types_mask=torch.BoolTensor([False]),
            loss = loss
        )

        summary = metrics_manager.summary(with_loss=True, with_ppl=True)

        self.assertEqual(summary[BuildingTypes.RESIDENTIAL]['mae'].value, 0)
        self.assertEqual(summary['ppl'], torch.exp( loss * 3 ) / 3)
        self.assertEqual(summary['loss'], loss)

    def test_reset(self):

        metrics_manager = MetricsManager(
            metrics=metrics_factory('mae')
        )

        loss = torch.FloatTensor([1.0])
        metrics_manager(
            y_true=torch.FloatTensor([1, 2, 3]).view(1,3,1),
            y_pred=torch.FloatTensor([1, 2, 3]).view(1,3,1),
            building_types_mask=torch.BoolTensor([False]),
            loss = loss
        )

        metrics_manager.reset()

        self.assertEqual(len(metrics_manager.metrics[BuildingTypes.RESIDENTIAL]), 1)
        self.assertEqual(len(metrics_manager.metrics[BuildingTypes.COMMERCIAL]), 1)
        self.assertEqual(metrics_manager.metrics[BuildingTypes.RESIDENTIAL][0].name, 'mae')
        self.assertEqual(metrics_manager.accumulated_unnormalized_loss, 0)

        metrics_manager.metrics[BuildingTypes.RESIDENTIAL][0].mean()
        metrics_manager.metrics[BuildingTypes.RESIDENTIAL][0].mean()

        self.assertEqual(metrics_manager.metrics[BuildingTypes.RESIDENTIAL][0].UNUSED_FLAG, True)
        

    def test_building_type(self):

        metrics_manager = MetricsManager(
            metrics=metrics_factory('mae')
        )

        metrics_manager(
            y_true=torch.FloatTensor([1, 2, 3]).view(1,3,1),
            y_pred=torch.FloatTensor([1, 2, 3]).view(1,3,1),
            building_type = BuildingTypes.RESIDENTIAL_INT
        )

        self.assertEqual(metrics_manager.metrics[BuildingTypes.RESIDENTIAL][0].name, 'mae')
        metrics_manager.metrics[BuildingTypes.RESIDENTIAL][0].mean()
        self.assertEqual(metrics_manager.metrics[BuildingTypes.RESIDENTIAL][0].value, 0)


class TestBenchmarkMetricsManager(unittest.TestCase):

    def test_create_benchmark_metrics_manager(self):
        metrics_manager = DatasetMetricsManager()

    def test_add_building_to_dataset_if_missing(self):
        metrics_manager = DatasetMetricsManager()

        metrics_manager.add_building_to_dataset_if_missing(
            dataset_name='test',
            building_id='0001')
        
        buliding_mm = metrics_manager.get_building_from_dataset(
            dataset_name='test',
            building_id='0001')
        
        # assert building_mm type is MetricsManager
        self.assertEqual(type(buliding_mm), MetricsManager)


    def test_call_benchmark_metrics_manager(self):

        metrics_manager = DatasetMetricsManager()

        metrics_manager.add_building_to_dataset_if_missing(
            dataset_name='test',
            building_id='0001')
        
        metrics_manager(
            dataset_name='test',
            building_id='0001',
            y_true=torch.FloatTensor([1, 2, 3]).view(1,3,1),
            y_pred=torch.FloatTensor([1, 2, 3]).view(1,3,1),
            building_types_mask=torch.BoolTensor([False])
        )

        buliding_mm = metrics_manager.get_building_from_dataset(
            dataset_name='test',
            building_id='0001')
        
        self.assertEqual(buliding_mm.metrics[BuildingTypes.RESIDENTIAL][0].name, 'cvrmse')
        buliding_mm.metrics[BuildingTypes.RESIDENTIAL][0].mean()
        self.assertEqual(buliding_mm.metrics[BuildingTypes.RESIDENTIAL][0].value, 0)

    def test_update_loss_and_get_ppl_from_benchmark_manager(self):

        metrics_manager = DatasetMetricsManager()

        metrics_manager.add_building_to_dataset_if_missing(
            dataset_name='test',
            building_id='0001')
        
        loss = torch.FloatTensor([1.0])
        metrics_manager(
            dataset_name='test',
            building_id='0001',
            y_true=torch.FloatTensor([1, 2, 3]).view(1,3,1),
            y_pred=torch.FloatTensor([1, 2, 3]).view(1,3,1),
            building_types_mask=torch.BoolTensor([False]),
            loss=loss,
        )

        buliding_mm = metrics_manager.get_building_from_dataset(
            dataset_name='test',
            building_id='0001')

        self.assertEqual(buliding_mm.get_ppl(), torch.exp( loss * 3 ) / 3)

    def test_benchmark_manager_summary(self):
        import pandas as pd

        metrics_manager = DatasetMetricsManager()

        metrics_manager.add_building_to_dataset_if_missing(
            dataset_name='test',
            building_id='0001')
        
        metrics_manager(
            dataset_name='test',
            building_id='0001',
            y_true=torch.FloatTensor([1, 2, 3]).view(1,3,1),
            y_pred=torch.FloatTensor([1, 2, 3]).view(1,3,1),
            building_types_mask=torch.BoolTensor([False]),
        )

        summary = metrics_manager.summary()

        print(summary)
