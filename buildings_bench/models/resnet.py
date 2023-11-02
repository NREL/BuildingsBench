import torch
from buildings_bench.models.base_model import BaseSurrogateModel
from buildings_bench.models.transformers import TimeSeriesSinusoidalPeriodicEmbedding
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict

'''
Basic Residual Block
'''
class ResBlock(nn.Module):

    def __init__(self, in_dim, out_dim, downsample=None):
        super(ResBlock, self).__init__()
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.activation(out)

        return out

'''
ResNet 
'''
class ResNet(BaseSurrogateModel):

    def __init__(self, hidden_dims,
                pred_len=1,
                continuous_head='mse',
                continuous_loads=True):
        
        super(ResNet,self).__init__(pred_len, continuous_loads)
        self.continuous_head = continuous_head
        out_dim = 1 if self.continuous_head == 'mse' else 2
        self.logits = nn.Linear(hidden_dims[-1], out_dim)
        
        self.building_embedding = nn.Embedding(2, 32)
        self.lat_embedding = nn.Linear(1, 32)
        self.lon_embedding = nn.Linear(1, 32)
        self.day_of_year_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32) 
        self.day_of_week_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32)
        self.hour_of_day_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32)
        
        blocks = []
        for in_dim, out_dim in zip(hidden_dims, hidden_dims[1:]):
            downsample = None
            if in_dim != out_dim:
                downsample = nn.Linear(in_dim, out_dim)
            block = ResBlock(in_dim=in_dim, out_dim=out_dim, downsample=downsample)
            blocks.append(block)

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        r"""Forward pass of the deep autoregressive model. 

        Args:
            x (Dict): dictionary of input tensors.
        Returns:
            logits (torch.Tensor): [batch_size, pred_len, vocab_size] if not continuous_loads,
                                   [batch_size, pred_len, 1] if continuous_loads and continuous_head == 'mse', 
                                   [batch_size, pred_len, 2] if continuous_loads and continuous_head == 'gaussian_nll'.
        """

        # [batch_size, seq_len, 256]

        time_series_embed = torch.cat([
            self.lat_embedding(x['latitude']),
            self.lon_embedding(x['longitude']),
            self.building_embedding(x['building_type']).squeeze(2),
            self.day_of_year_encoding(x['day_of_year']),
            self.day_of_week_encoding(x['day_of_week']),
            self.hour_of_day_encoding(x['hour_of_day']),
            x["building_char"],
            x["temperature"],
            x["humidity"],
            x["wind_speed"],
            x["wind_direction"],
            x["global_horizontal_radiation"],
            x["direct_normal_radiation"],
            x["diffuse_horizontal_radiation"]
        ], dim=2)

        # [batch_size, context_len, d]
        src_series_inputs = time_series_embed
        outs = self.blocks(src_series_inputs)
        return self.logits(outs)

    def predict(self, x: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(x)
        return out, None
    
    def loss(self, x, y):
        if self.continuous_head == 'mse':
            return F.mse_loss(x, y)
        elif self.continuous_head == 'gaussian_nll':
            return F.gaussian_nll_loss(x[:, :, 0].unsqueeze(2), y,
                                       F.softplus(x[:, :, 1].unsqueeze(2)) **2)

    def unfreeze_and_get_parameters_for_finetuning(self):
        for p in self.parameters():
            p.requires_grad = True    
        return self.parameters()

    def load_from_checkpoint(self, checkpoint_path):
        return None
