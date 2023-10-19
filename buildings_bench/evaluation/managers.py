import torch
from buildings_bench.evaluation import metrics_factory
from buildings_bench.evaluation.metrics import Metric, MetricType
from buildings_bench.evaluation.scoring_rules import ScoringRule
from typing import List, Optional
import pandas as pd
from copy import deepcopy


class BuildingTypes:
    """Enum for supported types of buildings.
    
    Attributes:
        RESIDENTIAL (str): Residential building type.
        COMMERCIAL (str): Commercial building type.
        RESIDENTIAL_INT (int): Integer representation of residential building type (0).
        COMMERCIAL_INT (int): Integer representation of commercial building type (1).
    """
    RESIDENTIAL = 'residential'
    COMMERCIAL = 'commercial'
    RESIDENTIAL_INT = 0
    COMMERCIAL_INT = 1


class MetricsManager:
    """A class that keeps track of all metrics (and a scoring rule)for one or more buildings.
    
    Metrics are computed for each building type (residential and commercial).

    Example:

    ```python
    from buildings_bench.evaluation.managers import MetricsManager
    from buildings_bench.evaluation import metrics_factory
    from buildings_bench import BuildingTypes
    import torch


    metrics_manager = MetricsManager(metrics=metrics_factory('cvrmse'))

    metrics_manager(
        y_true=torch.FloatTensor([1, 2, 3]).view(1,3,1),
        y_pred=torch.FloatTensor([1, 2, 3]).view(1,3,1),
        building_type = BuildingTypes.RESIDENTIAL_INT
    )

    for metric in metrics_manager.metrics[BuildingTypes.RESIDENTIAL]:
        metric.mean()
        print(metric.value) # prints tensor(0.)

    ```
    """
    def __init__(self, metrics: List[Metric] = None, scoring_rule: ScoringRule = None):
        """Initializes the MetricsManager.
        
        Args:
            metrics (List[Metric]): A list of metrics to compute for each
                building type.
            scoring_rule (ScoringRule): A scoring rule to compute for each
                building type.
        """
        self.metrics = {}
        if not metrics is None:
            self.metrics[BuildingTypes.RESIDENTIAL] = metrics
            self.metrics[BuildingTypes.COMMERCIAL] = deepcopy(metrics)
        self.scoring_rules = {}
        if not scoring_rule is None:
            self.scoring_rules[BuildingTypes.RESIDENTIAL] = scoring_rule
            self.scoring_rules[BuildingTypes.COMMERCIAL] = deepcopy(scoring_rule)
        self.accumulated_unnormalized_loss = 0
        self.total_samples = 0

    def _compute_all(self, y_true: torch.Tensor,
                     y_pred: torch.Tensor,
                     building_types_mask: torch.Tensor, **kwargs) -> None:
        """Computes all metrics and scoring rules for the given batch.
        
        Args:
            y_true (torch.Tensor): A tensor of shape [batch, pred_len, 1] with
                the true values.
            y_pred (torch.Tensor): A tensor of shape [batch, pred_len, 1] with
                the predicted values.
            building_types_mask (torch.Tensor): A tensor of shape [batch] with  
                the building types of each sample.

        """
        if len(self.metrics) > 0:
            for building_type in [BuildingTypes.RESIDENTIAL, BuildingTypes.COMMERCIAL]:
                
                if building_type == BuildingTypes.RESIDENTIAL:
                    
                    predictions = y_pred[~building_types_mask]
                    if predictions.shape[0] == 0:
                        continue
                    targets = y_true[~building_types_mask]
                elif building_type == BuildingTypes.COMMERCIAL:
                    predictions = y_pred[building_types_mask]
                    if predictions.shape[0] == 0:
                        continue
                    targets = y_true[building_types_mask]

                for metric in self.metrics[building_type]:
                    metric(targets, predictions)
        
        if len(self.scoring_rules) > 0:
            self._compute_scoring_rule(y_true,
                                      kwargs['y_categories'],
                                      kwargs['y_distribution_params'],
                                      kwargs['centroids'],
                                      building_types_mask)


    def _compute_scoring_rule(self, 
                            true_continuous,
                            true_categories,
                            y_distribution_params,
                            centroids,
                            building_types_mask) -> None:
        """Compute the scoring rule.
        
        Args:
            true_continuous (torch.Tensor): The true continuous load values.
                [bsz_sz, pred_len, 1]
            true_categories (torch.Tensor): The true quantized load values.
            y_distribution_params are [bsz_sz, pred_len, vocab_size] if logits,
             or are [bsz_sz, pred_len, 2] if Gaussian
            centroids (torch.Tensor): The bin values for the quantized distribution.
            building_types_mask (torch.Tensor): 
        """
        if y_distribution_params is None:
            raise ValueError('y_distribution_params must be provided to compute scoring rule.')
        
        for building_type in [BuildingTypes.RESIDENTIAL, BuildingTypes.COMMERCIAL]:
            if building_type == BuildingTypes.RESIDENTIAL:
                true_continuous_by_type = true_continuous[~building_types_mask]
                if true_continuous_by_type.shape[0] == 0:
                    continue
                self.scoring_rules[building_type](
                    true_continuous_by_type,
                    true_categories[~building_types_mask],
                    y_distribution_params[~building_types_mask],
                    centroids)
            elif building_type == BuildingTypes.COMMERCIAL:
                true_continuous_by_type = true_continuous[building_types_mask]
                if true_continuous_by_type.shape[0] == 0:
                    continue
                self.scoring_rules[building_type](
                    true_continuous_by_type,
                    true_categories[building_types_mask],
                    y_distribution_params[building_types_mask],
                    centroids)

    def _update_loss(self, loss, sample_size):
        """Updates the accumulated loss and total samples."""
        self.accumulated_unnormalized_loss += (loss * sample_size)
        self.total_samples += sample_size

    def get_ppl(self):
        """Returns the perplexity of the accumulated loss."""
        return torch.exp(self.accumulated_unnormalized_loss) / self.total_samples
    
    def summary(self, with_loss=False, with_ppl=False):
        """Return a summary of the metrics for the dataset.
        
        A summary maps keys to objects of type Metric or ScoringRule.
        """
        summary = {}
        for building_type in [BuildingTypes.RESIDENTIAL, BuildingTypes.COMMERCIAL]:
            summary[building_type] = {}
            if len(self.metrics) > 0:
                for metric in self.metrics[building_type]:
                    if not metric.UNUSED_FLAG:
                        metric.mean()
                        summary[building_type][metric.name] = metric
            if len(self.scoring_rules) > 0: 
                if not self.scoring_rules[building_type].value is None: 
                    self.scoring_rules[building_type].mean()
                    summary[building_type][self.scoring_rules[building_type].name] = \
                        self.scoring_rules[building_type]
                
        if with_ppl and self.total_samples > 0:
            summary['ppl'] = self.get_ppl()
        if with_loss and self.total_samples > 0:
            summary['loss'] = self.accumulated_unnormalized_loss / self.total_samples
        return summary

    def reset(self, loss: bool = True) -> None:
        """Reset the metrics."""
        for building_type in [BuildingTypes.RESIDENTIAL, BuildingTypes.COMMERCIAL]:
            for metric in self.metrics[building_type]:
                metric.reset()
            if len(self.scoring_rules) > 0:
                self.scoring_rules[building_type].reset()
        if loss:
            self.accumulated_unnormalized_loss = 0
            self.total_samples = 0


    def __call__(self, 
                 y_true: torch.Tensor,
                 y_pred: torch.Tensor,
                 building_types_mask: torch.Tensor = None,
                 building_type: int = BuildingTypes.COMMERCIAL_INT,
                 **kwargs):
        """Compute metrics for a batch of predictions.
        
        Args:
            y_true (torch.Tensor): The true (unscaled) load values. (continuous)
                shape is [batch_size, pred_len, 1]
            y_pred (torch.Tensor): The predicted (unscaled) load values. (continuous)
                shape is [batch_size, pred_len, 1]
            building_types_mask (torch.Tensor): 
                A boolean mask indicating the building type of each building.
                True (1) if commercial, False (0). Shape is [batch_size].
            building_type (int): The building type of the batch. Can be provided 
                instead of building_types_mask if all buildings are of the same type.

        Keyword args:
            y_categories (torch.Tensor): The true load values. (quantized)
            y_distribution_params (torch.Tensor): logits, Gaussian params, etc.
            centroids (torch.Tensor): The bin values for the quantized load.
            loss (torch.Tensor): The loss for the batch.
        """
        if building_types_mask is None:
            building_types_mask = (building_type == BuildingTypes.COMMERCIAL_INT) * \
                                   torch.ones(y_true.shape[0], dtype=torch.bool, device=y_true.device)

        self._compute_all(y_true, y_pred, building_types_mask, **kwargs)
        if 'loss' in kwargs:
            batch_size, pred_len, _ = y_true.shape
            self._update_loss(kwargs['loss'], batch_size * pred_len)


class DatasetMetricsManager:
    """
    A class that manages a MetricsManager for each building
    in one or more benchmark datasets. 
    One DatasetMetricsManager can be used to keep track of all metrics
    when evaluating a model on all of the benchmark's datasets.

    This class wil create a Pandas Dataframe summary containing the metrics for each building.

    Default metrics are NRMSE (CVRMSE), NMAE, NMBE.
    """
    default_metrics = metrics_factory('cvrmse',
                      types=[MetricType.SCALAR, MetricType.HOUR_OF_DAY]) \
                      + metrics_factory('nmbe', types=[MetricType.SCALAR, MetricType.HOUR_OF_DAY]) \
                      + metrics_factory('nmae', types=[MetricType.SCALAR, MetricType.HOUR_OF_DAY])

    def __init__(self, 
                 metrics: List[Metric] = default_metrics,
                 scoring_rule: ScoringRule = None):
        """
        Args:
            metrics (List[Metric]): A list of metrics to compute for each
                building type.
            scoring_rule (ScoringRule): A scoring rule to compute for each
                building type.
        """
        self.metrics_list = metrics
        self.scoring_rule = scoring_rule
        self._metrics = {}

    def get_building_from_dataset(self, dataset_name: str, building_id: str) -> Optional[MetricsManager]:
        """If the dataset and building exist, return the MetricsManager for the building.

        Args:
            dataset_name (str): The name of the dataset.
            building_id (str): The unique building identifier.
        
        Returns:
            A MetricsManager if the dataset and building exist, otherwise None.
        """
        # Check if dataset exists
        if dataset_name not in self._metrics:
            return None
        # Check if building exists
        if building_id not in self._metrics[dataset_name]:
            return None
        return self._metrics[dataset_name][building_id]
    
    def add_building_to_dataset_if_missing(self, dataset_name: str, building_id: str) -> None:
        """If the building does not exist, add a new MetricsManager for the building.

        Args:
            dataset_name (str): The name of the dataset.
            building_id (str): The unique building identifier.
        """
        # Check if dataset exists
        if dataset_name not in self._metrics:
            self._metrics[dataset_name] = {}
        # Check if building already exists
        if building_id not in self._metrics[dataset_name]:
            # Use deepcopy to pass new Metric and Scoring Rule objects
            self._metrics[dataset_name][building_id] = \
                MetricsManager(deepcopy(self.metrics_list), deepcopy(self.scoring_rule))


    def summary(self, dataset_name: str = None) -> pd.DataFrame:
        """Return a summary of the metrics for the dataset.
        
        Args:
            dataset_name (str): The name of the dataset to summarize. If None,
                summarize all datasets.
        Returns:
            A Pandas dataframe with the following columns:

                - dataset: The name of the dataset.
                - building_id: The unique ID of the building.
                - building_type: The type of the building.
                - metric: The name of the metric.
                - metric_type: The type of the metric. (scalar or hour_of_day)
                - value: The value of the metric.
        """
        summary = {}
        if dataset_name is None: # summarize all datasets
            for dataset_name in self._metrics.keys():
                summary[dataset_name] = {}
                for building_id in self._metrics[dataset_name].keys():
                    summary[dataset_name][building_id] = \
                        self._metrics[dataset_name][building_id].summary()
        else:
            summary[dataset_name] = {}
            for building_id in self._metrics[dataset_name].keys():
                summary[dataset_name][building_id] = \
                    self._metrics[dataset_name][building_id].summary()
       
        # to Pandas dataframe
        columns = ['dataset', 'building_id', 'building_type', 'metric', 'metric_type', 'value']
        rows = []
        # for each dataset
        for dataset_name in summary.keys():
            # for each building
            for building_id in summary[dataset_name].keys():
                # for the building type 
                for building_type in summary[dataset_name][building_id].keys():
                    # for each metric
                    for metric_name in summary[dataset_name][building_id][building_type].keys():
                        # if scoring rule, skip
                        if self.scoring_rule and metric_name == self.scoring_rule.name:
                            continue
                        # if the metric is a scalar
                        if summary[dataset_name][building_id][building_type][metric_name].type == MetricType.SCALAR:
                            rows.append([dataset_name, building_id, building_type,
                                         metric_name.split('-')[0], MetricType.SCALAR,
                                         summary[dataset_name][building_id][building_type][metric_name].value.item()])
                        # if the metric is a list of scalars
                        elif summary[dataset_name][building_id][building_type][metric_name].type == MetricType.HOUR_OF_DAY:
                            multi_hour_value = summary[dataset_name][building_id][building_type][metric_name].value
                            for hour in range(multi_hour_value.shape[0]):
                                rows.append([dataset_name, building_id,
                                             building_type, metric_name.split('-')[0] + '_' + str(hour), 
                                             MetricType.HOUR_OF_DAY, multi_hour_value[hour].item()])
        
        metric_df = pd.DataFrame(rows, columns=columns)

        if self.scoring_rule:
            columns = ['dataset', 'building_id', 'building_type', 'scoring_rule', 'value']
            rows = []
            for dataset_name in summary.keys():
                for building_id in summary[dataset_name].keys():
                    for building_type in summary[dataset_name][building_id].keys():
                        if self.scoring_rule.name in summary[dataset_name][building_id][building_type]:
                            score = summary[dataset_name][building_id][building_type][self.scoring_rule.name].value
                            for hour in range(score.shape[0]):
                                rows.append([dataset_name, building_id, building_type,
                                            self.scoring_rule.name + '_' + str(hour), score[hour].item()])
            scoring_rule_df = pd.DataFrame(rows, columns=columns)
            return metric_df, scoring_rule_df
        else:
            return metric_df


    def __call__(self, 
                 dataset_name: str,
                 building_id: str,
                 y_true: torch.Tensor, 
                 y_pred: torch.Tensor,
                 building_types_mask: torch.Tensor = None,
                 building_type: int = BuildingTypes.COMMERCIAL_INT,
                 **kwargs) -> None:
        """Compute metrics for a batch of predictions for a single building in a dataset.

        Args:
            dataset_name (str): The name of the dataset.
            building_id (str): The unique building identifier.
            y_true (torch.Tensor): The true (unscaled) load values. (continuous)
                shape is [batch_size, pred_len, 1]
            y_pred (torch.Tensor): The predicted (unscaled) load values. (continuous)
                shape is [batch_size, pred_len, 1]
            building_types_mask (torch.Tensor): 
                A boolean mask indicating the building type of each building.
                True (1) if commercial, False (0). Shape is [batch_size]. Default is None.
            building_type (int): The building type of the batch. Can be provided 
                instead of building_types_mask if all buildings are of the same type.

        Keyword args:
            y_categories (torch.Tensor): The true load values. (quantized)
            y_distribution_params (torch.Tensor): logits, Gaussian params, etc.
            centroids (torch.Tensor): The bin values for the quantized load.
            loss (torch.Tensor): The loss for the batch.        
        """
        self.add_building_to_dataset_if_missing(dataset_name, building_id)
        
        if building_types_mask is None:
            building_types_mask = (building_type == BuildingTypes.COMMERCIAL_INT) * \
                                   torch.ones(y_true.shape[0], dtype=torch.bool, device=y_true.device)
        self._metrics[dataset_name][building_id](y_true, y_pred, building_types_mask, **kwargs)

    
