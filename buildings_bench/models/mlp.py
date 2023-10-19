import torch
from buildings_bench.models.base_model import BaseModel
from buildings_bench.models.transformers import TimeSeriesSinusoidalPeriodicEmbedding
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict

'''
MLP 
'''
class MLP(BaseModel):

    def __init__(self, hidden_dim=256,
                context_len=1,
                pred_len=1,
                continuous_head='mse',
                continuous_loads=True):
        super(MLP,self).__init__(context_len, pred_len, continuous_loads)
        self.continuous_head = continuous_head
        out_dim = 1 if self.continuous_head == 'mse' else 2
        self.logits = nn.Linear(hidden_dim, out_dim)
        
        self.building_embedding = nn.Embedding(2, 32)
        self.lat_embedding = nn.Linear(1, 32)
        self.lon_embedding = nn.Linear(1, 32)
        self.day_of_year_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32) 
        self.day_of_week_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32)
        self.hour_of_day_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32)
        self.mlp = nn.Sequential(
            nn.Linear(32*6 + 806 + 7, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim),
            nn.ReLU()
        )

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
        src_series_inputs = time_series_embed[:, :self.context_len, :]
        outs = self.mlp(src_series_inputs)
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

class MLPText(BaseModel):

    def __init__(self, hidden_dim=256,
                context_len=1,
                pred_len=1,
                continuous_head='mse',
                continuous_loads=True):
        super(MLPText,self).__init__(context_len, pred_len, continuous_loads)
        self.continuous_head = continuous_head
        out_dim = 1 if self.continuous_head == 'mse' else 2
        self.logits = nn.Linear(hidden_dim, out_dim)
        
        self.building_embedding = nn.Embedding(2, 32)
        self.lat_embedding = nn.Linear(1, 32)
        self.lon_embedding = nn.Linear(1, 32)
        self.day_of_year_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32) 
        self.day_of_week_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32)
        self.hour_of_day_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32)
        self.mlp = nn.Sequential(
            nn.Linear(32*6 + 768 + 7, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim),
            nn.ReLU()
        )
        self.train_mode = True

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
        src_series_inputs = time_series_embed[:, :self.context_len, :]
        outs = self.mlp(src_series_inputs)
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