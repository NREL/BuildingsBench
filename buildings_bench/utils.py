import numpy as np
import random 
import torch 
import os 
import datetime 


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    #print(f"Random seed set as {seed}")


def save_model_checkpoint(model, optimizer, scheduler, step, path):
    """Save model checkpoint.
    """
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': step
    }
    torch.save(checkpoint, path)
    #print(f"Saved model checkpoint to {path}...")


def load_model_checkpoint(path, model, optimizer, scheduler, local_rank):
    """Load model checkpoint.
    """
    checkpoint = torch.load(path, map_location=f'cuda:{local_rank}')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    step = checkpoint['step']
    #print(f"Loaded model checkpoint from {path}")
    return model, optimizer, scheduler, step


def worker_init_fn_eulp(worker_id):
    """Set random seed for each worker and init file pointer
    for Buildings-900K dataset workers.

    Args:
        worker_id (int): worker id
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    worker_info = torch.utils.data.get_worker_info()
    worker_info.dataset.init_fp()
   

def time_features_to_datetime(time_features: np.ndarray,
                              year: int) -> np.array:
    """
    Convert time features to datetime objects.

    Args:
        time_features (np.ndarray): Array of time features.
            [:,0] is day of year,
            [:,1] is day of week,
            [:,2] is hour of day.
        year (int): Year to use for datetime objects.

    Returns:
        np.array: Array of datetime objects.
    """
    day_of_year = time_features[:,0]
    hour_of_day = time_features[:,2]
    return np.array([datetime.datetime(year, 1, 1, 0, 0, 0) +   # January 1st
                    datetime.timedelta(days=int(doy-1), hours=int(hod), minutes=0, seconds=0)
                    for doy, hod in zip(day_of_year, hour_of_day)])   
