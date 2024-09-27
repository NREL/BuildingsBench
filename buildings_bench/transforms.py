import numpy as np
from pathlib import Path
import pandas as pd
import torch
import pickle as pkl
import os
import sklearn.preprocessing as preprocessing
from typing import Union

       
class BoxCoxTransform:
    """A class that computes and applies the Box-Cox transform to data.
    """
    def __init__(self, max_datapoints: int = 1000000):
        """
        Args:
            max_datapoints (int): If the number of datapoints is greater than this, subsample.
        """
        self.boxcox = None
        self.max_datapoints = max_datapoints

    def train(self, data: np.array) -> None:
        """Train the Box-Cox transform on the data with sklearn.preprocessing.PowerTransformer.
        
        Args:
            data (np.array): of shape (n, 1) or (b,n,1)
        """       
        self.boxcox = preprocessing.PowerTransformer(method='box-cox', standardize=True)
        data = data.flatten().reshape(-1,1)
        if data.shape[0] > self.max_datapoints:
            #print(f'Box-Cox: subsampling {self.max_datapoints} datapoints')
            data = data[np.random.choice(data.shape[0], self.max_datapoints, replace=False)]
        self.boxcox.fit_transform(1e-6 + data)


    def save(self, output_path: Path) -> None:
        """Save the Box-Cox transform"""
        with open(output_path / "boxcox.pkl", 'wb') as f:
            pkl.dump(self.boxcox, f)
    
    def load(self, saved_path: Path) -> None:
        """Load the Box-Cox transform"""
        with open(saved_path / "boxcox.pkl", 'rb') as f:
            self.boxcox = pkl.load(f)

    def transform(self, sample: np.ndarray) -> np.ndarray:
        """Transform a sample via Box-Cox.
        Not ran on the GPU, so input/output are numpy arrays.

        Args:
            sample (np.ndarray): of shape (n, 1) or (b,n,1) 
        
        Returns:
            transformed_sample (np.ndarray): of shape (n, 1) or (b,n,1)
        """
        init_shape = sample.shape
        return self.boxcox.transform(1e-6 + sample.flatten().reshape(-1,1)).reshape(init_shape)


    def undo_transform(self, sample: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]: 
        """Undo the transformation of a sample via Box-Cox
        
        Args:
            sample (Union[np.ndarray, torch.LongTensor]): of shape (n, 1) or (b,n,1). 
                numpy if device is cpu or torch Tensor if device is cuda.

        Returns:
            unscaled_sample (Union[np.ndarray, torch.LongTensor]): of shape (n, 1) or (b,n,1).
        """
        is_tensor = isinstance(sample, torch.Tensor)
        # if torch.Tensor, convert to numpy first
        if is_tensor:
            device = sample.device
            sample = sample.cpu().numpy()
        init_shape = sample.shape       
        sample = self.boxcox.inverse_transform(sample.flatten().reshape(-1,1)).reshape(init_shape)
        # convert back to torch
        if is_tensor:
            sample = torch.from_numpy(sample).to(device)
        return sample
    

class StandardScalerTransform:
    """ A class that standardizes data by removing the mean and scaling to unit variance.
    """
    def __init__(self, max_datapoints=1000000, device='cpu'):
        """
        Args:
            max_datapoints (int): If the number of datapoints is greater than this, subsample.
            device (str): 'cpu' or 'cuda'
        """
        self.mean_ = None
        self.std_ = None
        self.max_datapoints = max_datapoints
        self.device=device

    def train(self, data: np.array) -> None:
        """Train the StandardScaler transform on the data.
        
        Args:
            data (np.array): of shape (n, 1) or (b,n,1)
        """       
        data = data.flatten().reshape(-1,1)
        if data.shape[0] > self.max_datapoints:
            #print(f'Subsampling {self.max_datapoints} datapoints to fit the StandardScalerTransform')
            data = data[np.random.choice(data.shape[0], self.max_datapoints, replace=False)]
        self.mean_ = torch.from_numpy(np.array([np.mean(data)])).float().to(self.device)
        self.std_ = torch.from_numpy(np.array([np.std(data)])).float().to(self.device)


    def save(self, output_path: Path) -> None:
        """Save the StandardScaler transform"""
        mean_ = self.mean_.cpu().numpy().reshape(-1)
        std_ = self.std_.cpu().numpy().reshape(-1)
        np.save(output_path / "standard_scaler.npy", np.array([mean_, std_]))
    
    
    def load(self, saved_path: Path) -> None:
        """Load the StandardScaler transform"""
        x = np.load(saved_path / "standard_scaler.npy")
        self.mean_ = torch.from_numpy(np.array([x[0]])).float().to(self.device)
        self.std_ = torch.from_numpy(np.array([x[1]])).float().to(self.device)


    def transform(self, sample: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Transform a sample via StandardScaler
        
        Args:
            sample (Union[np.ndarray, torch.LongTensor]): shape (n, 1) or (b,n,1) 
        Returns:
            transformed_samples (torch.Tensor): shape (n, 1) or (b,n,1)
        """
        if isinstance(sample, np.ndarray):
            sample = torch.from_numpy(sample).float().to(self.device)        
        return (sample - self.mean_) / self.std_


    def undo_transform(self, sample: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Undo the transformation of a sample via StandardScaler
        
        Args:
            sample (np.ndarray): of shape (n, 1) or (b,n,1) or torch.Tensor of shape (n, 1) or (b,n,1)

        Returns:
            unscaled_sample (torch.Tensor): of shape (n, 1) or (b,n,1)
        """
        if isinstance(sample, np.ndarray):
            sample = torch.from_numpy(sample).float().to(self.device)
        return self.std_ * sample + self.mean_        


    def undo_transform_std(self, scaled_std: torch.Tensor) -> torch.Tensor:
        """Undo transform for standard deviation.
        
        Args:
            scaled_std (torch.Tensor): of shape (n, 1) or (b,n,1)
        
        Returns:
            unscaled_std (torch.Tensor): of shape (n, 1) or (b,n,1)
        """
        return self.std_ * scaled_std
    
        
class LatLonTransform:
    """Pre-processing lat,lon data with standard normalization by Buildings-900K training set.
    """
    def __init__(self):
        metadata_path = Path(os.environ.get('BUILDINGS_BENCH','')) / 'metadata'
        # Load withheld pumas
        with open(metadata_path / 'withheld_pumas.tsv', 'r') as f:
            # tab separated file
            line = f.readlines()[0]
            self.withheld_pumas = line.split('\t')

        census_regions = ['northeast', 'midwest', 'south', 'west']
        self.puma_to_centroid = {}
        lat_means, lat_stds = 0, 0
        lon_means, lon_stds = 0, 0
        for idx,cr in enumerate(census_regions):
            cr_df = pd.read_csv(metadata_path / f'map_of_pumas_in_census_region_{idx+1}_{cr}.csv', header=0)
            # keys are cr_df['GISJOIN'], values are (cr_df['latitude'], cr_df['longitude'])
            self.puma_to_centroid.update(dict(zip(cr_df['GISJOIN'], zip(cr_df['latitude'], cr_df['longitude']))))
            # Filter out withheld pumas
            non_withheld_df = cr_df[~cr_df['GISJOIN'].isin(self.withheld_pumas)]
            # Compute mean and std 
            lat_means += non_withheld_df['latitude'].mean()
            lat_stds += non_withheld_df['latitude'].std()
            lon_means += non_withheld_df['longitude'].mean()
            lon_stds += non_withheld_df['longitude'].std()

        self.lat_means = lat_means / 4
        self.lon_means = lon_means / 4
        self.lat_stds = lat_stds / 4
        self.lon_stds = lon_stds / 4

        # convert self.puma_to_centroid values to np.ndarray of shape (2,)
        for k,v in self.puma_to_centroid.items():
            # Normalize
            v = (np.array(v, dtype=np.float32) - np.array([self.lat_means, self.lon_means])) / np.array([self.lat_stds, self.lon_stds])
            self.puma_to_centroid[k] = v.astype(np.float32)


    def transform_latlon(self, latlon: np.ndarray) -> np.ndarray:
        """Transform a raw Lat/Lon sample into a normalized Lat/Lon sample

        Args:
            latlon (np.ndarray): of shape (2,).

        Returns:
            transformed_latlon (np.ndarray): of shape (2,).
        """
        return (latlon - np.array([self.lat_means, self.lon_means])) / np.array([self.lat_stds, self.lon_stds])

    def undo_transform(self, normalized_latlon: np.ndarray) -> np.ndarray:
        """Undo the transformation of a sample

        Args:
            normalized_latlon (np.ndarray): of shape (n, 2) or (b,n,2).

        Returns:
            unnormalized_latlon (np.ndarray): of shape (n, 2) or (b,n,2).
        """
        init_shape = normalized_latlon.shape
        normalized_latlon = normalized_latlon.reshape(-1,2)
        lat = normalized_latlon[:,0] * self.lat_stds + self.lat_means
        lon = normalized_latlon[:,1] * self.lon_stds + self.lon_means
        return np.stack([lat, lon], axis=1).reshape(init_shape)

    def transform(self, puma_id: str) -> np.ndarray:
        """Look up a PUMA ID's normalized Lat/Lon centroid.

        This is used in the Buildings-900K Dataset to look up a lat/lon
        for each building's PUMA.

        Args:
            puma_id (str): PUMA ID
        
        Returns:
            centroid (np.ndarray): of shape (1,2)
        """
        return self.puma_to_centroid[puma_id].reshape(1,2)


class TimestampTransform:
    """Extract timestamp features from a Pandas timestamp Series.
    """
    def __init__(self, is_leap_year: bool = False) -> None:
        """
        Args:
            is_leap_year (bool): Whether the year of the building data is a leap year or not.
        """
        self.day_year_normalization = 365 if is_leap_year else 364
        self.hour_of_day_normalization = 23
        self.day_of_week_normalization = 6

    def transform(self, timestamp_series: pd.DataFrame) -> np.ndarray:
        """Extract timestamp features from a Pandas timestamp Series.


        - Day of week (0-6)
        - Day of year (0-364)
        - Hour of day (0-23)

        Args:
            timestamp_series (pd.DataFrame): of shape (n,) or (b,n)
        
        Returns:
            time_features (np.ndarray): of shape (n,3) or (b,n,3)
        """
        # If the input is a DatetimeIndex
        if isinstance(timestamp_series, pd.DatetimeIndex):
            timestamp_series = timestamp_series.to_series()
        # Convert to datetime
        timestamp_series = pd.to_datetime(timestamp_series)
        # Extract features
        day_of_week = timestamp_series.dt.dayofweek
        day_of_year = timestamp_series.dt.dayofyear
        hour_of_day = timestamp_series.dt.hour
        time_features = np.stack([day_of_year / self.day_year_normalization,
                         day_of_week / self.day_of_week_normalization,
                         hour_of_day / self.hour_of_day_normalization], axis=1).astype(np.float32)
        return time_features * 2 - 1


    def undo_transform(self, time_features: np.ndarray) -> np.ndarray:
        """Convert normalized time features back to original time features

        Args:
            time_features (np.ndarray): of shape (n, 3) or (b,n,3)
        
        Returns:
            unnormalized_time_features (np.ndarray): of shape (n, 3) or (b,n,3)
        """
        init_shape = time_features.shape
        time_features = time_features.reshape(-1,3)
        time_features = (time_features + 1) * 0.5
        day_of_year = np.round(time_features[:,0] * self.day_year_normalization)
        day_of_week = np.round(time_features[:,1] * self.day_of_week_normalization)
        hour_of_day = np.round(time_features[:,2] * self.hour_of_day_normalization)
        return np.stack([day_of_year, day_of_week, hour_of_day], axis=1).astype(np.int32).reshape(init_shape)
    