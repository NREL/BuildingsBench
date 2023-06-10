import torch
from buildings_bench.models.base_model import BaseModel


class AveragePersistence(BaseModel):
    """Predict each hour as the average over each previous day.
    """
    def __init__(self, context_len=168, pred_len=24, continuous_loads=True):
        super().__init__(context_len, pred_len, continuous_loads)

    def forward(self, x):
        # [bsz x seqlen x 1]
        src_series = x['load'][:,:self.context_len,:]

        preds = []
        stds = []
        for i in range(self.pred_len):
            preds += [ torch.mean(src_series[:,i::24,:], dim=1) ]
            stds += [ torch.clamp(
                torch.std(src_series[:,i::24,:], dim=1),
                min=1e-3)]
        return torch.cat([torch.stack(preds, 1), torch.stack(stds, 1)],2)
    
    def loss(self, x, y):
        return x, y
    
    def predict(persistence_model, x):
        out = persistence_model(x)
        return out[:,:,0].unsqueeze(-1), out

    def unfreeze_and_get_parameters_for_finetuning(self):
        return None
    
    def load_from_checkpoint(self, checkpoint_path):
        return None
    

class CopyLastDayPersistence(BaseModel):
    """Predict each hour as the same hour from the previous day.
    """
    def __init__(self, context_len=168, pred_len=24, continuous_loads=True):
        super().__init__(context_len, pred_len, continuous_loads)
        assert self.context_len >= 24
        assert self.pred_len >= 24

    def forward(self, x):
        # [bsz x seqlen x 1]
        src_series = x['load'][:,:self.context_len,:]
        return src_series[:, self.context_len-24:]
    
    def loss(self, x, y):
        return x, y
    
    def predict(persistence_model, x):
        return persistence_model(x), None

    def unfreeze_and_get_parameters_for_finetuning(self):
        return None
    
    def load_from_checkpoint(self, checkpoint_path):
        return None


class CopyLastWeekPersistence(BaseModel):
    """Predict each hour as the same hour from the previous week.
    """
    def __init__(self, context_len=168, pred_len=24, continuous_loads=True):
        super().__init__(context_len, pred_len, continuous_loads)
        assert self.context_len >= 168
        assert self.pred_len >= 24

    def forward(self, x):
        # [bsz x seqlen x 1]
        src_series = x['load'][:,:self.context_len,:]
        return src_series[:, self.context_len-168 : self.context_len - 168 +24]

    def loss(self, x, y):
        return x, y
    
    def predict(persistence_model, x):
        return persistence_model(x), None
    
    def unfreeze_and_get_parameters_for_finetuning(self):
        return None
    
    def load_from_checkpoint(self, checkpoint_path):
        return None
