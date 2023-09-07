# buildings_bench.models
import torch
from typing import Callable, Tuple, Dict

# Import models here
from buildings_bench.models.dlinear_regression import DLinearRegression
from buildings_bench.models.linear_regression import LinearRegression
from buildings_bench.models.transformers import LoadForecastingTransformer, LoadForecastingTransformerWithWeather
from buildings_bench.models.persistence import *



model_registry = {
    'TransformerWithTokenizer-L': LoadForecastingTransformer,
    'TransformerWithTokenizer-M': LoadForecastingTransformer,
    'TransformerWithTokenizer-S': LoadForecastingTransformer,
    'TransformerWithTokenizer-L-ignore-spatial': LoadForecastingTransformer,
    'TransformerWithTokenizer-L-8192': LoadForecastingTransformer,
    'TransformerWithTokenizer-L-344': LoadForecastingTransformer,
    'TransformerWithMSE': LoadForecastingTransformer,
    'TransformerWithGaussian-L': LoadForecastingTransformer,
    'TransformerWithGaussian-M': LoadForecastingTransformer,
    'TransformerWithGaussian-S': LoadForecastingTransformer,
    'TransformerWithGaussian-weather-S': LoadForecastingTransformerWithWeather,
    'TransformerWithGaussian-weather-M': LoadForecastingTransformerWithWeather,
    'TransformerWithGaussian-weather-L': LoadForecastingTransformerWithWeather,
    'AveragePersistence': AveragePersistence,
    'CopyLastDayPersistence': CopyLastDayPersistence,
    'CopyLastWeekPersistence': CopyLastWeekPersistence,
    'LinearRegression': LinearRegression,
    'DLinearRegression': DLinearRegression,

    # Register your model here
}


def model_factory(model_name: str, model_args: Dict) -> Tuple[torch.nn.Module, Callable, Callable]:
    """Instantiate and returns a model for the benchmark.

    Returns the model itself,
    the loss function to use, and the predict function.

    The predict function should return a tuple of two tensors: 
    (point predictions, prediction distribution parameters) where
    the distribution parameters may be, e.g., logits, or mean and variance.

    Args:
        model_name (str): Name of the model.
        model_args (Dict): The keyword arguments for the model.
    Returns:
        model (torch.nn.Module): the instantiated model  
        loss (Callable): loss function
        predict (Callable): predict function
    """
    assert model_name in model_registry.keys(), \
        f"Model {model_name} not in registry: {model_registry.keys()}"
    
    model = model_registry[model_name](**model_args)
    loss = model.loss
    predict = model.predict
    return model, loss, predict

