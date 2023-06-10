from typing import Callable
import torch 
import abc

class MetricType:
    """Enum class for metric types.

    Attributes:
        SCALAR (str): A scalar metric.
        HOUR_OF_DAY (str): A metric that is calculated for each hour of the day.
    """
    SCALAR = 'scalar'
    HOUR_OF_DAY = 'hour_of_day'


class BuildingsBenchMetric(metaclass=abc.ABCMeta):
    """An abstract class for all metrics.

    The basic idea is to acculumate the errors etc. in a list and then
    calculate the mean of the errors etc. at the end of the evaluation.

    Calling the metric will add the error to the list of errors. Calling `.mean()`
    will calculate the mean of the errors, populating the `.value` attribute.


    Attributes:
        name (str): The name of the metric.
        type (MetricType): The type of the metric.
        value (float): The value of the metric.
    """
    def __init__(self, name: str, type: MetricType):
        self.name = name
        self.type = type
        self.value = None

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def mean(self):
        raise NotImplementedError()
 

class Metric(BuildingsBenchMetric):
    """A class that represents an error metric.  

    Example:
    
    ```python
    rmse = Metric('rmse', MetricType.SCALAR, squared_error, sqrt=True)
    mae = Metric('mae', MetricType.SCALAR, absolute_error)
    nmae = Metric('nmae', MetricType.SCALAR, absolute_error, normalize=True)
    cvrmse = Metric('cvrmse', MetricType.SCALAR, squared_error, normalize=True, sqrt=True)
    nmbe = Metric('nmbe', MetricType.SCALAR, bias_error, normalize=True)
    ```
    """
    def __init__(self, name: str, type: MetricType, function: Callable, **kwargs):
        super().__init__(name, type)
        self.function = function
        self.kwargs = kwargs
        self.global_values = []
        self.errors = []
        self.UNUSED_FLAG = True

    def __call__(self, y_true, y_pred) -> None:
        """
        y_true (torch.Tensor): shape [batch_size, pred_len]
        y_pred (torch.Tensor): shape [batch_size, pred_len]
        """
        self.UNUSED_FLAG = False
        self.errors += [self.function(y_true, y_pred)]
        self.global_values += [y_true]

    def reset(self) -> None:
        """Reset the metric."""
        self.global_values = []
        self.errors = []
        self.value = None
        self.UNUSED_FLAG = True

    def mean(self) -> None:
        """Calculate the mean of the error metric."""
        if self.UNUSED_FLAG:
            # Returning a number >= 0 is undefined,
            # because this metric is unused. -1
            # is a flag to indicate this.
            return
        
        # When we concatenate errors and global values
        # we want shape errors to be shape [batches, pred_len]
        # and global values to be 1D
        if self.errors[0].dim() == 1:
            self.errors = [e.unsqueeze(0) for e in self.errors]
        if self.global_values[0].dim() == 0:
            self.global_values = [g.unsqueeze(0) for g in self.global_values]
    
        all_errors = torch.concatenate(self.errors,0)
        if self.type == MetricType.SCALAR:
            mean = torch.mean(all_errors)
        elif self.type == MetricType.HOUR_OF_DAY:
            mean = torch.mean(all_errors, dim=0)
        # for root mean error
        if self.kwargs.get('sqrt', False):
            mean = torch.sqrt(mean)
        # normalize
        if self.kwargs.get('normalize', False):
            mean = mean / torch.mean(torch.concatenate(self.global_values,0))
        self.value = mean
    
    
################## METRICS ##################

def absolute_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """A PyTorch method that calculates the absolute error (AE) metric.

    Args:
        y_true (torch.Tensor): [batch, pred_len]
        y_pred (torch.Tensor): [batch, pred_len]
    
    Returns:
        error (torch.Tensor): [batch, pred_len]
    """
    return torch.abs(y_true - y_pred)


def squared_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """A PyTorch method that calculates the squared error (SE) metric.

    Args:
        y_true (torch.Tensor): [batch, pred_len]
        y_pred (torch.Tensor): [batch, pred_len]
    
    Returns:
        error (torch.Tensor): [batch, pred_len]
    """
    return torch.square(y_true - y_pred)

 
def bias_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """A PyTorch method that calculates the bias error (BE) metric.

    Args:
        y_true (torch.Tensor): [batch, pred_len]
        y_pred (torch.Tensor): [batch, pred_len]
    
    Returns:
        error (torch.Tensor): [batch, pred_len]    
    """
    return y_true - y_pred