import torch
from buildings_bench.models.base_model import BaseModel
from buildings_bench.models.transformers import TimeSeriesSinusoidalPeriodicEmbedding
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict


class DeepAutoregressiveRNN(BaseModel):

    def __init__(self, hidden_size = 40, lstm_layers = 3, 
                 context_len = 168, pred_len=24,
                 for_pretraining=False, 
                 continuous_head='gaussian_nll',
                 continuous_loads=True):
        super(DeepAutoregressiveRNN,self).__init__(context_len, pred_len, continuous_loads)
        self.continuous_head = continuous_head
        out_dim = 1 if self.continuous_head == 'mse' else 2
        self.logits = nn.Linear(hidden_size, out_dim)
        
        self.for_pretraining = for_pretraining
        if self.for_pretraining:

            self.power_embedding = nn.Linear(1, 64)
            self.building_embedding = nn.Embedding(2, 32)
            self.lat_embedding = nn.Linear(1, 32)
            self.lon_embedding = nn.Linear(1, 32)
            self.day_of_year_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32) 
            self.day_of_week_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32)
            self.hour_of_day_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32)

            self.encoder = nn.LSTM(256, hidden_size, num_layers=lstm_layers, batch_first=True)
            self.decoder = nn.LSTM(256, hidden_size, num_layers=lstm_layers, batch_first=True)
        else:
            self.encoder = nn.LSTM(1, hidden_size, num_layers=lstm_layers, batch_first=True)
            self.decoder = nn.LSTM(1, hidden_size, num_layers=lstm_layers, batch_first=True)

    def forward(self, x):
        r"""Forward pass of the deep autoregressive model. 

        Args:
            x (Dict): dictionary of input tensors.
        Returns:
            logits (torch.Tensor): [batch_size, pred_len, vocab_size] if not continuous_loads,
                                   [batch_size, pred_len, 1] if continuous_loads and continuous_head == 'mse', 
                                   [batch_size, pred_len, 2] if continuous_loads and continuous_head == 'gaussian_nll'.
        """
        if self.for_pretraining:
            # [batch_size, seq_len, 256]
            time_series_embed = torch.cat([
                self.lat_embedding(x['latitude']),
                self.lon_embedding(x['longitude']),
                self.building_embedding(x['building_type']).squeeze(2),
                self.day_of_year_encoding(x['day_of_year']),
                self.day_of_week_encoding(x['day_of_week']),
                self.hour_of_day_encoding(x['hour_of_day']),
                self.power_embedding(x['load']).squeeze(2),
            ], dim=2)
        else:
            time_series_embed = x['load']

        # [batch_size, context_len, d]
        src_series_inputs = time_series_embed[:, :self.context_len, :]
        # [batch_size, pred_len, d]
        # The last element of the target sequence is not used as input
        # The last element of the source sequence is used as the initial decoder input
        tgt_series_inputs = time_series_embed[:, self.context_len-1 : -1, :]
        _, (h_n, c_n) = self.encoder(src_series_inputs)
        outs, _ = self.decoder(tgt_series_inputs, (h_n, c_n))
        return self.logits(outs)

    def predict(self, x: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.generate_sample(x, greedy=True)
    
    def loss(self, x, y):
        if self.continuous_head == 'mse':
            return F.mse_loss(x, y)
        elif self.continuous_head == 'gaussian_nll':
            return F.gaussian_nll_loss(x[:, :, 0].unsqueeze(2), y,
                                       F.softplus(x[:, :, 1].unsqueeze(2)) **2)

    def unfreeze_and_get_parameters_for_finetuning(self):
        return self.parameters()

    def load_from_checkpoint(self, checkpoint_path):
        stored_ckpt = torch.load(checkpoint_path)
        model_state_dict = stored_ckpt['model']
        new_state_dict = {}
        for k,v in model_state_dict.items():
            # remove string 'module.' from the key
            if 'module.' in k:
                new_state_dict[k.replace('module.', '')] = v
            else:
                new_state_dict[k] = v
        self.load_state_dict(new_state_dict)    
        print(f"Loaded model checkpoint from {checkpoint_path}...")


    @torch.no_grad()
    def generate_sample(self, 
                 x,
                 temperature=1.0,
                 greedy=False,
                 num_samples=1):
        """Sample from the conditional distribution.

        Use output of decoder at each prediction step as input to the next decoder step.
        Implements greedy decoding and random temperature-controlled sampling.
        
        Args:
            x (Dict): dictionary of input tensors
            temperature (float): temperature for sampling
            greedy (bool): whether to use greedy decoding
            num_samples (int): number of samples to generate
        
        Returns:
            predictions (torch.Tensor): of shape [batch_size, pred_len, 1] or shape [batch_size, num_samples, pred_len] if num_samples > 1.
            distribution_parameters (torch.Tensor): of shape [batch_size, pred_len, 1]. Not returned if sampling.
        """
        if self.for_pretraining:
            time_series_embed = torch.cat([
                self.lat_embedding(x['latitude']),
                self.lon_embedding(x['longitude']),
                self.building_embedding(x['building_type']).squeeze(2),
                self.day_of_year_encoding(x['day_of_year']),
                self.day_of_week_encoding(x['day_of_week']),
                self.hour_of_day_encoding(x['hour_of_day']),
                self.power_embedding(x['load']).squeeze(2),
            ], dim=2)
        else:
            time_series_embed = x['load']
        # [batch_size, context_len, d_model]
        src_series_inputs = time_series_embed[:, :self.context_len, :]
        tgt_series_inputs = time_series_embed[:, self.context_len-1 : -1, :]

        _, (h_n, c_n) = self.encoder(src_series_inputs)
        decoder_input = tgt_series_inputs[:, 0, :].unsqueeze(1)
        if num_samples > 1 and not greedy:
            # [batch_size, 1, emb_size] --> [batch_size * num_sampes, 1, emb_size]
            decoder_input = decoder_input.repeat_interleave(num_samples, dim=0)
            # [num_layers, batch_size, hidden_size] -> [num_layers, batch_size * num_samples, hidden_size]
            h_n = h_n.repeat_interleave(num_samples, dim=1)
            c_n = c_n.repeat_interleave(num_samples, dim=1)
        all_preds, all_logits = [], []
        for k in range(1, self.pred_len+1):
            decoder_output, (h_n, c_n) = self.decoder(decoder_input, (h_n, c_n))
            # [batch_size, 1] if continuous (2 if head is gaussian_nll) or [batch_size, vocab_size] if not continuous_loads
            outputs = self.logits(decoder_output[:, -1, :])
            all_logits += [outputs.unsqueeze(1)]

            if self.continuous_head == 'mse':
                all_preds += [outputs] 
            elif self.continuous_head == 'gaussian_nll':
                if greedy:
                    all_preds += [outputs[:, 0].unsqueeze(1)] # mean only
                    outputs = all_preds[-1] # [batch_size, 1, 1]
                else:
                    mean = outputs[:,0]
                    std= torch.nn.functional.softplus(outputs[:,1])
                    outputs = torch.distributions.normal.Normal(mean, std).sample().unsqueeze(1)
                    all_preds += [outputs]    
                
            # [batch_size, d_model]
            if k < self.pred_len:
                # [batch_size, d_model]
                next_decoder_input = tgt_series_inputs[:, k]
                if num_samples > 1 and not greedy:
                    # [batch_size, d_model] --> [batch_size * num_samples, d_model]
                    next_decoder_input = next_decoder_input.repeat_interleave(num_samples, dim=0)
                if self.for_pretraining:
                    # Use the embedding predicted load instead of the ground truth load
                    embedded_pred = self.power_embedding(outputs)  
                    next_decoder_input = torch.cat([ next_decoder_input[:, :-embedded_pred.shape[-1]], embedded_pred ], dim=1)
                else:
                    next_decoder_input = torch.cat([ next_decoder_input[:, :-1], outputs ], dim=1)
                # Append the next decoder input to the decoder input
                decoder_input = torch.cat([decoder_input, next_decoder_input.unsqueeze(1)], dim=1)
        if num_samples == 1 or greedy:
            if self.continuous_head == 'gaussian_nll':
                # [batch_size, pred_len, 2]
                gaussian_params = torch.stack(all_logits,1)[:,:,0,:]
                means = gaussian_params[:,:,0]
                sigma = torch.nn.functional.softplus(gaussian_params[:,:,1])
                return torch.stack(all_preds,1), torch.cat([means.unsqueeze(2), sigma.unsqueeze(2)],2)
            else:
                return torch.stack(all_preds,1), torch.stack(all_logits,1)[:,:,0,:]
        else:
            # [batch_size, num_samples, pred_len]
            return torch.stack(all_preds,1).reshape(-1, num_samples, self.pred_len)        