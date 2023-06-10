import abc
from typing import Tuple, Dict, Union
from pathlib import Path
import torch
import torch.nn as nn


class BaseModel(nn.Module, metaclass=abc.ABCMeta):
    """Base class for all models."""
    def __init__(self, context_len, pred_len, continuous_loads):
        super().__init__()
        self.context_len = context_len
        self.pred_len = pred_len
        self.continuous_loads = continuous_loads

    @abc.abstractmethod
    def forward(self, x: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. 
        
        Expected keys in x:

            - 'load': torch.Tensor of shape [batch_size, seq_len, 1]
            - 'building_type': torch.LongTensor of shape (batch_size, 1)
            - 'day_of_year': torch.FloatTensor of shape (batch_size, 1)
            - 'hour_of_day': torch.FloatTensor of shape (batch_size, 1)
            - 'day_of_week': torch.FloatTensor of shape (batch_size, 1)
            - 'latitude': torch.FloatTensor of shape (batch_size, 1)
            - 'longitude': torch.FloatTensor of shape (batch_size, 1)

        Args:
            x (Dict): dictionary of input tensors
        Returns:
            predictions, distribution parameters (Tuple[torch.Tensor, torch.Tensor]): outputs
        """
        raise NotImplementedError()
    

    @abc.abstractmethod
    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """A function for computing the loss."""
        raise NotImplementedError()

    
    @staticmethod
    def predict(model: nn.Module, x: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """A function for making a forecast on x with the model.

        Args:
            model (nn.Module): model
            x (Dict): dictionary of input tensors
        Returns:
            predictions (torch.Tensor): of shape (batch_size, pred_len, 1)
            distribution parameters (torch.Tensor]): of shape (batch_size, pred_len, -1)
        """
        raise NotImplementedError()


    @abc.abstractmethod
    def unfreeze_and_get_parameters_for_finetuning(self):
        """For transfer learning. 
        
        - Set requires_grad=True for parameters being fine-tuned (if necessary)
        - Return the parameters that should be fine-tuned.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def load_from_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Describes how to load the model from checkpoint_path."""
        raise NotImplementedError()
