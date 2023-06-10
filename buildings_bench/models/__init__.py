# buildings_bench.models
import torch
from typing import Callable, Tuple, Dict

# Import models here
from buildings_bench.models.dlinear_regression import DLinearRegression
from buildings_bench.models.linear_regression import LinearRegression
from buildings_bench.models.transformers import LoadForecastingTransformer
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

    # if model_name == 'TransformerWithTokenizer':
    #     # TODO: directly pass kwargs to model
    #     model = LoadForecastingTransformer(context_len=args.context_len,
    #                                 pred_len=args.pred_len,
    #                                 vocab_size=args.vocab_size,
    #                                 num_encoder_layers=args.num_encoder_layers,
    #                                 num_decoder_layers=args.num_decoder_layers,
    #                                 dim_feedforward=args.dim_feedforward,
    #                                 d_model=args.d_model,
    #                                 dropout=args.dropout,
    #                                 activation=args.activation,
    #                                 nhead=args.nhead,
    #                                 continuous_loads=False,
    #                                 ignore_spatial=args.ignore_spatial)
    #     #loss = lambda x, y: torch.nn.functional.cross_entropy(x.reshape(-1, args.vocab_size),
    #     #                                                        y.long().reshape(-1))
    #     loss = model.loss
    #     predict = model.predict

    # elif model_name == 'TransformerWithMSE':
    #     model = LoadForecastingTransformer(context_len=args.context_len,
    #                                 pred_len=args.pred_len,
    #                                 num_encoder_layers=args.num_encoder_layers,
    #                                 num_decoder_layers=args.num_decoder_layers,
    #                                 dim_feedforward=args.dim_feedforward,
    #                                 d_model=args.d_model,
    #                                 dropout=args.dropout,
    #                                 activation=args.activation,
    #                                 nhead=args.nhead,
    #                                 continuous_loads=True,
    #                                 continuous_head='mse',
    #                                 ignore_spatial=args.ignore_spatial)
    #     loss = torch.nn.MSELoss()
    #     predict = model.predict

    # elif model_name == 'TransformerWithGaussian':
    #     model = LoadForecastingTransformer(context_len=args.context_len,
    #                                 pred_len=args.pred_len,
    #                                 vocab_size=args.vocab_size,
    #                                 num_encoder_layers=args.num_encoder_layers,
    #                                 num_decoder_layers=args.num_decoder_layers,
    #                                 dim_feedforward=args.dim_feedforward,
    #                                 d_model=args.d_model,
    #                                 dropout=args.dropout,
    #                                 activation=args.activation,
    #                                 nhead=args.nhead,
    #                                 continuous_loads=True,
    #                                 continuous_head='gaussian_nll',
    #                                 ignore_spatial=args.ignore_spatial)
    #     loss = lambda x, y: torch.nn.functional.gaussian_nll_loss(x[:, :, 0].unsqueeze(2), y, F.softplus(x[:, :, 1].unsqueeze(2)) **2)
    #     #predict = lambda model, x: model.generate_sample(x, greedy=True)
    #     predict = model.predict

    # elif model_name == 'AveragePersistence':
    #     model = AveragePersistence()
    #     loss = model.loss
    #     predict = AveragePersistence.predict

    # elif model_name == 'CopyLastDayPersistence':
    #     model = CopyLastDayPersistence()
    #     loss = model.loss
    #     predict = CopyLastDayPersistence.predict

    # elif model_name == 'CopyLastWeekPersistence':
    #     model = CopyLastWeekPersistence()
    #     loss = model.loss
    #     predict = CopyLastWeekPersistence.predict
        
    # # Register your model here
    # elif model_name == 'LinearRegression':
    #     model = LinearRegression(**model_args)
    #     loss = model.loss
    #     predict = LinearRegression.predict
    # # Register your model here
    # elif model_name == 'DLinearRegression':
    #     model = DLinearRegression(**model_args)
    #     loss = model.loss
    #     predict = DLinearRegression.predict
    # Register your model here        
    # else:
    #     raise ValueError(f'Unknown model name: {model_name}')
    # return model, loss, predict
    
