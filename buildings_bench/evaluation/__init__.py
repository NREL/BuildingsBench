from buildings_bench.evaluation.metrics import Metric, MetricType
from buildings_bench.evaluation.metrics import absolute_error, squared_error, bias_error
from buildings_bench.evaluation.scoring_rules import ScoringRule
from buildings_bench.evaluation.scoring_rules import RankedProbabilityScore
from buildings_bench.evaluation.scoring_rules import ContinuousRankedProbabilityScore
from typing import List


metrics_registry = [
    'rmse',
    'mae',
    'nrmse',
    'nmae',
    'mbe',
    'nmbe',
    'cvrmse'
]

scoring_rule_registry = [
    'rps',
    'crps'
]


def metrics_factory(name: str,
                    types: List[MetricType] = [MetricType.SCALAR]) -> List[Metric]:
    """
    Create a metric from a name.
    By default, will return a scalar metric.

    Args:
        name: The name of the metric.
        types: The types of the metric. List[MetricTypes]
    Returns:
        A list of metrics. List[Metric]
    """
    assert name.lower() in metrics_registry, f'Invalid metric name: {name}'

    if name.lower() == 'rmse':
        return [ Metric(f'{name.lower()}-{type}', type, squared_error, sqrt=True) for type in types ]
    elif name.lower() == 'mae':
        return [ Metric(f'{name.lower()}-{type}', type, absolute_error) for type in types ]
    elif name.lower() == 'nrmse':
        return [ Metric(f'{name.lower()}-{type}', type, squared_error, normalize=True, sqrt=True) for type in types ]
    elif name.lower() == 'nmae':
        return [ Metric(f'{name.lower()}-{type}', type, absolute_error, normalize=True) for type in types ]
    elif name.lower() == 'mbe':
        return [ Metric(f'{name.lower()}-{type}', type, bias_error) for type in types ]
    elif name.lower() == 'nmbe':
        return [ Metric(f'{name.lower()}-{type}', type, bias_error, normalize=True) for type in types ]
    elif name.lower() == 'cvrmse':
        return [ Metric(f'{name.lower()}-{type}', type, squared_error, normalize=True, sqrt=True) for type in types ]


def scoring_rule_factory(name: str) -> ScoringRule:
    """Create a scoring rule from a name.

    Args:
        name: The name of the scoring rule.
    Returns:
        A scoring rule.
    """
    assert name.lower() in scoring_rule_registry, f'Invalid scoring rule name: {name}'
    
    if name.lower() == 'crps':
        return ContinuousRankedProbabilityScore()
    elif name.lower() == 'rps':
        return RankedProbabilityScore()


def all_metrics_list() -> List[Metric]:
    """Returns all registered metrics.
    
    Returns:
        A list of metrics. List[Metric]
    """
    metrics_list = []
    for metric in metrics_registry:
        metrics_list += metrics_factory(metric, types=[MetricType.SCALAR, MetricType.HOUR_OF_DAY])
    return metrics_list
