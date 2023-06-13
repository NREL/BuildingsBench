import torch
from buildings_bench.models.base_model import BaseModel
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinearRegression(BaseModel):
    """
    Decomposition-Linear
    """
    def __init__(self,context_len=168, pred_len=24, continuous_loads=True):
        super(DLinearRegression, self).__init__(context_len, pred_len, continuous_loads)

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        # self.individual = True
        # self.channels = 1
       
        self.Linear_Seasonal = nn.Linear(context_len, self.pred_len)
        self.Linear_Trend = nn.Linear(context_len, self.pred_len)
        
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        src_series = x['load'][:,:self.context_len,:]
        seasonal_init, trend_init = self.decompsition(src_series)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
    
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel]
    
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(x,y)
    
    def predict(self, x):
        return self.forward(x), None

    def unfreeze_and_get_parameters_for_finetuning(self):
        for p in self.parameters():
            p.requires_grad = True
        return self.parameters()
    
    def load_from_checkpoint(self, checkpoint_path):
        return None
