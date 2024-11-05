import abc
from typing import Tuple, Dict, Union
from pathlib import Path
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

class BaseModel(nn.Module, PyTorchModelHubMixin, metaclass=abc.ABCMeta):
    """Base class for all models."""
    def __init__(self, context_len, pred_len, continuous_loads):
        """Init method for BaseModel.

        Args:
            context_len (int): length of context window
            pred_len (int): length of prediction window
            continuous_loads (bool): whether to use continuous load values
        """
        super().__init__()
        self.context_len = context_len
        self.pred_len = pred_len
        self.continuous_loads = continuous_loads

    @abc.abstractmethod
    def forward(self, x: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. 
        
        Expected keys in x:

            - 'load': torch.Tensor of shape (batch_size, seq_len, 1)
            - 'building_type': torch.LongTensor of shape (batch_size, seq_len, 1)
            - 'day_of_year': torch.FloatTensor of shape (batch_size, seq_len, 1)
            - 'hour_of_day': torch.FloatTensor of shape (batch_size, seq_len, 1)
            - 'day_of_week': torch.FloatTensor of shape (batch_size, seq_len, 1)
            - 'latitude': torch.FloatTensor of shape (batch_size, seq_len, 1)
            - 'longitude': torch.FloatTensor of shape (batch_size, seq_len, 1)

        Args:
            x (Dict): dictionary of input tensors
        Returns:
            predictions, distribution parameters (Tuple[torch.Tensor, torch.Tensor]): outputs
        """
        raise NotImplementedError()
    

    @abc.abstractmethod
    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """A function for computing the loss.
        
        Args:
            x (torch.Tensor): preds of shape (batch_size, seq_len, 1)
            y (torch.Tensor): targets of shape (batch_size, seq_len, 1)
        Returns:
            loss (torch.Tensor): scalar loss
        """
        raise NotImplementedError()

    @abc.abstractmethod 
    def predict(self, x: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """A function for making a forecast on x with the model.

        Args:
            x (Dict): dictionary of input tensors
        Returns:
            predictions (torch.Tensor): of shape (batch_size, pred_len, 1)
            distribution_parameters (torch.Tensor): of shape (batch_size, pred_len, -1)
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
