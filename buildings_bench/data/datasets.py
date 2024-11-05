from pathlib import Path
import torch
import pandas as pd
import numpy as np
from typing import List, Union, Iterator, Tuple
import buildings_bench.transforms as transforms
from buildings_bench.transforms import BoxCoxTransform, StandardScalerTransform
from buildings_bench import BuildingTypes
import pyarrow.parquet as pq
from buildings_bench.utils import get_puma_county_lookup_table
import datetime


class TorchBuildingDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for a single building's energy timeseries (a Pandas Dataframe)
      with a timestamp index and a `power` column.

    Used to iterate over mini-batches of 192-hour timeseries
    (168 hours of context, 24 hours prediction horizon).
    """
    def __init__(self, 
                dataframe: pd.DataFrame,
                building_latlon: List[float],
                building_type: BuildingTypes,
                context_len: int = 168,
                pred_len: int = 24,
                sliding_window: int = 24,
                apply_scaler_transform: str = '',
                scaler_transform_path: Path = None,
                is_leap_year = False,
                weather_dataframe: pd.DataFrame = None,
                weather_transform_path: Path = None):
        """
        Args:
            dataframe (pd.DataFrame): Pandas DataFrame with a timestamp index and a 'power' column.
            building_latlon (List[float]): Latitude and longitude of the building.
            building_type (BuildingTypes): Building type for the dataset.
            context_len (int, optional): Length of context. Defaults to 168.
            pred_len (int, optional): Length of prediction. Defaults to 24.
            sliding_window (int, optional): Stride for sliding window to split timeseries into test samples. Defaults to 24.
            apply_scaler_transform (str, optional): Apply scaler transform {boxcox,standard} to the load. Defaults to ''.
            scaler_transform_path (Path, optional): Path to the pickled data for BoxCox transform. Defaults to None.
            is_leap_year (bool, optional): Is the year a leap year? Defaults to False.
            weather_dataframe (pd.DataFrame, optional): Weather timeseries data. Defaults to None.
            weather_transform_path (Path, optional): Path to the pickled data for weather transform. Defaults to None. 
        """
        self.df = dataframe        
        self.building_type = building_type
        self.context_len = context_len
        self.pred_len = pred_len
        self.sliding_window = sliding_window
        self.apply_scaler_transform = apply_scaler_transform
        self.weather_df = weather_dataframe
        self.weather_transform_path = weather_transform_path
        self.normalized_latlon = transforms.LatLonTransform().transform_latlon(building_latlon)
        self.time_transform = transforms.TimestampTransform(is_leap_year=is_leap_year)
        if self.apply_scaler_transform == 'boxcox':
            self.load_transform = BoxCoxTransform()
            self.load_transform.load(scaler_transform_path)
        elif self.apply_scaler_transform == 'standard':
            self.load_transform = StandardScalerTransform()
            self.load_transform.load(scaler_transform_path)
        
    def __len__(self):
        return (len(self.df) - self.context_len - self.pred_len) // self.sliding_window

    def __getitem__(self, idx):
        seq_ptr = self.context_len + self.sliding_window * idx

        load_features = self.df['power'].iloc[seq_ptr-self.context_len : seq_ptr+self.pred_len].values.astype(np.float32)
        if self.apply_scaler_transform != '':
            load_features = self.load_transform.transform(load_features)
        time_features = self.time_transform.transform(self.df.index[seq_ptr-self.context_len : seq_ptr+self.pred_len ])
        latlon_features = self.normalized_latlon.reshape(1,2).repeat(self.context_len + self.pred_len, axis=0).astype(np.float32) 
        if self.building_type == BuildingTypes.RESIDENTIAL:
            building_features = BuildingTypes.RESIDENTIAL_INT * np.ones((self.context_len + self.pred_len,1), dtype=np.int32)
        elif self.building_type == BuildingTypes.COMMERCIAL:
            building_features = BuildingTypes.COMMERCIAL_INT * np.ones((self.context_len + self.pred_len,1), dtype=np.int32)

        sample = {
            'latitude': latlon_features[:, 0][...,None],
            'longitude': latlon_features[:, 1][...,None],
            'day_of_year': time_features[:, 0][...,None],
            'day_of_week': time_features[:, 1][...,None],
            'hour_of_day': time_features[:, 2][...,None],
            'building_type': building_features,
            'load': load_features[...,None]
        }

        if self.weather_df is None:
            return sample
        else:

            weather_df = self.weather_df.iloc[seq_ptr-self.context_len: seq_ptr+self.pred_len]
            
            # transform
            weather_transform = StandardScalerTransform()
            for col in weather_df.columns[1:]:
                weather_transform.load(self.weather_transform_path / col)
                sample.update({col : weather_transform.transform(weather_df[col].to_numpy())[0][...,None]})
            return sample
        

class TorchBuildingDatasetFromParquet:
    """Generate PyTorch Datasets out of EULP parquet files.
    
    Each file has multiple buildings (with same Lat/Lon and building type) and
    each building is a column. All time series are for the same year.     

    Attributes:
        building_datasets (dict): Maps unique building ids to a TorchBuildingDataset.   
    """
    def __init__(self,
                data_path: Path,
                parquet_datasets: List[str],
                building_latlons: List[List[float]],
                building_types: List[BuildingTypes],
                weather_inputs: List[str] = None,
                context_len: int = 168,
                pred_len: int = 24,
                sliding_window: int = 24,
                apply_scaler_transform: str = '',
                scaler_transform_path: Path = None,
                leap_years: List[int] = None):
        """
        Args:
            data_path (Path): Path to the dataset
            parquet_datasets (List[str]): List of paths to a parquet file, each has a timestamp index and multiple columns, one per building.
            building_latlons (List[List[float]]): List of latlons for each parquet file.
            building_types (List[BuildingTypes]): List of building types for each parquet file.
            weather_inputs (List[str]): list of weather feature names to use as additional inputs. Default: None.
            context_len (int, optional): Length of context. Defaults to 168.
            pred_len (int, optional): Length of prediction. Defaults to 24.
            sliding_window (int, optional): Stride for sliding window to split timeseries into test samples. Defaults to 24.
            apply_scaler_transform (str, optional): Apply scaler transform {boxcox,standard} to the load. Defaults to ''.
            scaler_transform_path (Path, optional): Path to the pickled data for BoxCox transform. Defaults to None.
            leap_years (List[int], optional): List of leap years. Defaults to None.
        """
        self.building_datasets = {}
    
        weather_df = None
        weather_transform_path = None
        if weather_inputs: # build a puma-county lookup table
            metadata_path = data_path / 'metadata'
            weather_transform_path = metadata_path / 'transforms' / 'weather'
            lookup_df = get_puma_county_lookup_table(metadata_path)

        for parquet_data, building_latlon, building_type in zip(parquet_datasets, building_latlons, building_types):
            
            if weather_inputs: # load weather data
                puma_id = parquet_data.split('puma=')[1]
                county = lookup_df.loc[puma_id]['nhgis_2010_county_gisjoin']

                # some processing to get the corresponding path for weather data
                weather_datapath = Path(parquet_data).parents[3] / 'weather'
                # load corresponding weather files
                weather_df = pd.read_csv(weather_datapath / f'{county}.csv')

                # This is assuming that the file always starts from January 1st (ignoring the year)
                assert datetime.datetime.strptime(weather_df['date_time'].iloc[0], '%Y-%m-%d %H:%M:%S').strftime('%m-%d') == '01-01',\
                    "The weather file does not start from Jan 1st"
                
                weather_df.columns = ['timestamp'] + weather_inputs 
                weather_df = weather_df[['timestamp'] + weather_inputs]

            df = pq.read_table(parquet_data)

            # Order by timestamp
            df = df.to_pandas().sort_values(by='timestamp')
            # Set timestamp as the index
            df.set_index('timestamp', inplace=True)
            df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')

            # TODO: Patrick.
            # remove the first row so that it aligns with weather files (which start from 01:00:00)
            if weather_inputs: 
                df = df[1:] 

            # split into multiple dataframes by column, keeping the index
            dfs = np.split(df, df.shape[1], axis=1)

            # for each column in the multi_building_dataset, create a BuildingYearDatasetFromCSV
            for building_dataframe in dfs:
                building_name = building_dataframe.columns[0]
                year = building_dataframe.index[0].year
                is_leap_year = True if year in leap_years else False

                # remove the year for aggregating over all years, later
                b_file = f'{building_type}_{Path(parquet_data).stem}/{building_name}'
                #by_file = f'{building_type}_{Path(parquet_data).stem}/{building_name}_year={year}'

                # rename column 1 to power
                building_dataframe.rename(columns={building_dataframe.columns[0]: 'power'}, inplace=True)
                self.building_datasets[b_file] = TorchBuildingDataset(building_dataframe, 
                                                                    building_latlon, 
                                                                    building_type, 
                                                                    context_len, 
                                                                    pred_len, 
                                                                    sliding_window,
                                                                    apply_scaler_transform,
                                                                    scaler_transform_path,
                                                                    is_leap_year,
                                                                    weather_dataframe=weather_df,
                                                                    weather_transform_path=weather_transform_path)


    def __len__(self):
        return len(self.building_datasets)

    def __iter__(self) -> Iterator[Tuple[str, TorchBuildingDataset]]:
        """Generator to iterate over the building datasets.

        Yields:
            A pair of building id, TorchBuildingDataset objects. 
        """
        for building_name, building_dataset in self.building_datasets.items():
            yield (building_name, building_dataset)


class TorchBuildingDatasetsFromCSV:
    """TorchBuildingDatasetsFromCSV
    
    Generate PyTorch Datasets from a list of CSV files.

    Attributes:
        building_datasets (dict): Maps unique building ids to a list of tuples (year, TorchBuildingDataset). 
    """
    def __init__(self,
                data_path: Path,
                building_year_files: List[str],
                building_latlon: List[float],
                building_type: BuildingTypes,
                weather_inputs: List[str] = None,
                context_len: int = 168,
                pred_len: int = 24,
                sliding_window: int = 24,
                apply_scaler_transform: str = '',
                scaler_transform_path: Path = None,
                leap_years: List[int] = None):
        """
        Args:
            data_path (Path): Path to the dataset
            building_year_files (List[str]): List of paths to a csv file, each has a timestamp index and multiple columns, one per building.
            building_type (BuildingTypes): Building type for the dataset.
            weather_inputs (List[str], optional): list of weather feature names to use as additional inputs. Defaults to None.
            context_len (int, optional): Length of context. Defaults to 168.
            pred_len (int, optional): Length of prediction sequence for the forecasting model. Defaults to 24.
            sliding_window (int, optional): Stride for sliding window to split timeseries into test samples. Defaults to 24.
            apply_scaler_transform (str, optional): Apply scaler transform {boxcox,standard} to the load. Defaults to ''.
            scaler_transform_path (Path, optional): Path to the pickled data for BoxCox transform. Defaults to None.
            leap_years (List[int], optional): List of leap years. Defaults to None.
        """
        self.building_datasets = {}
        self.building_type = building_type

        weather_df = None
        weather_transform_path = None
        if weather_inputs:
            metadata_path = data_path / 'metadata'
            weather_transform_path = metadata_path / 'transforms' / 'weather'
            ds_name = building_year_files[0].split('/')[0]

            if ds_name != 'SMART' and ds_name != 'BDG-2':
                 #TODO: finalize the weather source
                weather_df = pd.read_csv(data_path / f'{ds_name}' / 'weather_isd.csv',
                                          index_col=0, header=0, parse_dates=True)

        for building_year_file in building_year_files:
            name = building_year_file.split('_')[0].split('/')[1]
            year = int(building_year_file.split('=')[1])
            is_leap_year = True if year in leap_years else False
        
            df = pd.read_csv(data_path / (building_year_file + '.csv'),
                                                index_col=0, header=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')
            df = df.sort_index()
            if len(df.columns) > 1:
                bldg_names = df.columns
                # split into multiple dataframes by column, keeping the index
                dfs = np.split(df, df.shape[1], axis=1)
            else:
                bldg_names = [name]
                dfs = [df]

            if weather_inputs and (ds_name == 'SMART' or ds_name == 'BDG-2'):
                weather_df = pd.read_csv(data_path / f'{ds_name}' / f'weather_{name}.csv',
                                          index_col=0, header=0, parse_dates=True)

            # for each bldg, create a TorchBuildingDatasetFromCSV
            for bldg_name, bldg_df in zip(bldg_names, dfs):
                bldg_df.rename(columns={bldg_df.columns[0]: 'power'}, inplace=True)

                if not bldg_name in self.building_datasets:
                    self.building_datasets[bldg_name] = []

                self.building_datasets[bldg_name] += [(year, TorchBuildingDataset(bldg_df, 
                                                                                building_latlon, 
                                                                                building_type, 
                                                                                context_len, 
                                                                                pred_len, 
                                                                                sliding_window,
                                                                                apply_scaler_transform,
                                                                                scaler_transform_path,
                                                                                is_leap_year,
                                                                                weather_dataframe=weather_df,
                                                                                weather_transform_path=weather_transform_path))]
    
    def __len__(self):
        return len(self.building_datasets)                        
        
    def __iter__(self) -> Iterator[Tuple[str, torch.utils.data.ConcatDataset]]:
        """A Generator for TorchBuildingDataset objects.

        Yields:
            A tuple of the building id and a ConcatDataset of the TorchBuildingDataset objects for all years.    
        """
        for building_name, building_year_datasets in self.building_datasets.items():
            building_year_datasets = sorted(building_year_datasets, key=lambda x: x[0])
            building_dataset = torch.utils.data.ConcatDataset([
                byd[1] for byd in building_year_datasets])
            yield (building_name, building_dataset)


class PandasBuildingDatasetsFromCSV:
    """Generate Pandas Dataframes from a list of CSV files.
    
    Can be used with sklearn models or tree-based models that require Pandas Dataframes.
    In this case, use 'features' = 'engineered' to generate a dataframe with engineered features.

    Create a dictionary of building datasets from a list of csv files.
    Used as a generator to iterate over Pandas Dataframes for each building.
    The Pandas Dataframe contain all of the years of data for the building.

    Attributes:
        building_datasets (dict): Maps unique building ids to a list of tuples (year, Dataframe).    
    """
    def __init__(self, 
                data_path: Path,
                building_year_files: List[str],
                building_latlon: List[float],
                building_type: BuildingTypes,
                weather_inputs: List[str] = None,
                features: str = 'transformer',
                apply_scaler_transform: str = '',
                scaler_transform_path: Path = None,
                leap_years: List[int] = []):
        """        
        Args:
            data_path (Path): Path to the dataset
            building_year_files (List[str]): List of paths to a csv file, each has a timestamp index and multiple columns, one per building.
            building_type (BuildingTypes): Building type for the dataset.
            weather_inputs (List[str], optional): list of weather feature names to use as additional inputs. Defaults to None.
            features (str, optional): Type of features to use. Defaults to 'transformer'. {'transformer','engineered'}
                'transformer' features: load, latitude, longitude, hour of day, day of week, day of year, building type
                'engineered' features are an expansive list of mainly calendar-based features, useful for traditional ML models.
            apply_scaler_transform (str, optional): Apply scaler transform {boxcox,standard} to the load. Defaults to ''.
            scaler_transform_path (Path, optional): Path to the pickled data for BoxCox transform. Defaults to None.
            leap_years (List[int], optional): List of leap years. Defaults to None.
        """
        self.building_type = building_type
        self.features = features 
        self.apply_scaler_transform = apply_scaler_transform
        self.leap_years = leap_years
        
        if self.features == 'transformer':
            self.normalized_latlon = transforms.LatLonTransform().transform_latlon(building_latlon)
            
            if self.apply_scaler_transform == 'boxcox':
                self.load_transform = BoxCoxTransform()
                self.load_transform.load(scaler_transform_path)
            elif self.apply_scaler_transform == 'standard':
                self.load_transform = StandardScalerTransform()
                self.load_transform.load(scaler_transform_path)

        self.building_datasets = {}
        
        weather_df = None
        if weather_inputs:
            assert weather_inputs == ['temperature'], \
                "Only temperature is supported by PandasBuildingDatasetsFromCSV for now."

            self.weather_transform_path = data_path / 'metadata' / 'transforms' / 'weather'
            ds_name = building_year_files[0].split('/')[0]
            if ds_name != 'SMART' and ds_name != 'BDG-2':
                #TODO: Truc. finalize the weather source
                weather_df = pd.read_csv(data_path / f'{ds_name}' / 'weather_isd.csv',
                                          index_col=0, header=0, parse_dates=True) 
        
        for building_year_file in building_year_files:
            #fullname = building_year_file.split('_')[0]
            name = building_year_file.split('_')[0].split('/')[1]
            year = int(building_year_file.split('=')[1])

            # load the csv file
            df = pd.read_csv(data_path / (building_year_file + '.csv'),
                             index_col=0, header=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')
            df = df.asfreq('h')
            df = df.sort_index()
            
            bldg_dfs =[]
            # is multi-building file? 
            if len(df.columns) > 1:
                bldg_names = df.columns
                # split into multiple dataframes by column, keeping the index
                bldg_dfs = np.split(df, df.shape[1], axis=1)
            else:
                bldg_names = [name]
                bldg_dfs = [df]

            if weather_inputs and (ds_name == 'SMART' or ds_name == 'BDG-2'):
                weather_df = pd.read_csv(data_path / f'{ds_name}' / f'weather_{name}.csv',
                                          index_col=0, header=0, parse_dates=True)
            
            for bldg_name,df in zip(bldg_names, bldg_dfs):
                if self.features == 'engineered':
                    self._prepare_data_with_engineered_features(bldg_name, df, year, weather_df)
                elif self.features == 'transformer':
                    self._prepare_data_transformer(bldg_name, df, year, weather_df)


    def _prepare_data_with_engineered_features(self, bldg_name, df, year, weather_df):
        """ Utility function for applying feature engineering to the dataframe `df`.
        """
        # rename column 1 to power
        df.rename(columns={df.columns[0]: 'power'}, inplace=True)

        # Create hour_of_day,.., etc columns
        df["hour_x"] = np.sin(np.radians((360/24) * df.index.hour))
        df["hour_y"] = np.cos(np.radians((360/24) * df.index.hour))
        df["month_x"] = np.sin(np.radians((360/12) * df.index.month))
        df["month_y"] = np.cos(np.radians((360/12) * df.index.month))
    

        # add calendar-based variables as categorical data
        # see https://colab.research.google.com/drive/1ZWpJY03xLIsUrlOzgTNHemKyLatMgKrp?usp=sharing#scrollTo=NJABd7ow5EHC
        df["day_of_week"] = df.index.weekday
        df["hour_of_day"] = df.index.hour
        df["month_of_year"] = df.index.month
        df["weekend"] = df.index.weekday.isin([5,6])
        df= pd.get_dummies(df, columns=["day_of_week", "hour_of_day", "month_of_year", "weekend"], dtype=np.int32)

        if weather_df is not None:
            weather_df = weather_df.loc[:, ['temperature']]
            df = df.join(weather_df)

        if bldg_name in self.building_datasets:
            self.building_datasets[bldg_name] += [(year,df)]
        else:
            self.building_datasets[bldg_name] = [(year,df)]


    def _prepare_data_transformer(self, bldg_name, df, year, weather_df):
        """ Utility function for applying feature transformations expected by Transformer-based models to the dataframe `df`."""
        is_leap_year = True if year in self.leap_years else False            
        time_transform = transforms.TimestampTransform(is_leap_year)
        
        df.rename(columns={df.columns[0]: 'load'}, inplace=True)
        if self.apply_scaler_transform != '':
            df['load'] = self.load_transform.transform(df['load'].values)

        # create a column called "latitude" and "longitude" with the normalized lat/lon
        # of the same shape as 'load'
        df["latitude"] = self.normalized_latlon[0] * np.ones(df.shape[0])
        df["longitude"] = self.normalized_latlon[1] * np.ones(df.shape[0])
        

        if self.building_type == BuildingTypes.RESIDENTIAL:
            df["building_type"] = BuildingTypes.RESIDENTIAL_INT * np.ones(df.shape[0])
        elif self.building_type == BuildingTypes.COMMERCIAL:
            df["building_type"] = BuildingTypes.COMMERCIAL_INT * np.ones(df.shape[0])


        time_features = time_transform.transform(df.index)
        df["day_of_week"] = time_features[:,1] * np.ones(df.shape[0])
        df["hour_of_day"] = time_features[:,2] * np.ones(df.shape[0])
        df["day_of_year"] = time_features[:, 0] * np.ones(df.shape[0])

        if weather_df is not None:
            # TODO: Test

            #weather_df = weather_df.loc[:, ['temperature']]
            #weather_transform = StandardScalerTransform()
            #weather_transform.load(self.weather_transform_path / 'temperature')
            #weather_df['temperature'] = weather_transform.transform(weather_df['temperature'].to_numpy())[0]

            weather_transform = StandardScalerTransform()
            for col in weather_df.columns[1:]:
                weather_transform.load(self.weather_transform_path / col)
                weather_df.update({col : weather_transform.transform(weather_df[col].to_numpy())[0][...,None]})

            weather_df = weather_df.loc[:, ['temperature']]

            df = df.join(weather_df)

        if bldg_name in self.building_datasets:
            self.building_datasets[bldg_name] += [(year,df)]
        else:
            self.building_datasets[bldg_name] = [(year,df)]

    def __len__(self):
        return len(self.building_datasets)  

    def __iter__(self) -> Iterator[Tuple[str, pd.DataFrame]]:
        """Generator for iterating over the dataset.

        Yields:
            A pair of building id and Pandas dataframe. 
                The dataframe has all years concatenated.    
        """
        for building_id, building_dataset in self.building_datasets.items():
            building_dataset = sorted(building_dataset, key=lambda x: x[0])
            df = pd.concat([df[1] for df in building_dataset])
            # fill missing values with 0
            df = df.fillna(0) 
            yield (building_id, df)


class PandasTransformerDataset(torch.utils.data.Dataset):
    """Create a Torch Dataset from a Pandas DataFrame.

    Used to iterate over mini-batches of e.g, 192-hour (168 hours context + 24 hour pred horizon) timeseries.
    """
    def __init__(self, 
                 df: pd.DataFrame,
                 context_len: int = 168,
                 pred_len: int = 24,
                 sliding_window: int = 24,
                 weather_inputs: List[str] = None):
        """
        Args:
            df (pd.DataFrame): Pandas DataFrame with columns: load, latitude, longitude, hour of day, day of week, day of year, building type
            context_len (int, optional): Length of context.. Defaults to 168.
            pred_len (int, optional): Length of prediction sequence for the forecasting model. Defaults to 24.
            sliding_window (int, optional): Stride for sliding window to split timeseries into test samples. Defaults to 24.
            weather_inputs (List[str], optional): list of weather feature names to use as additional inputs. Defaults to None.
                The df is assumed to already have the weather inputs in the list as columns.
        """
        self.df = df
        self.context_len = context_len
        self.pred_len = pred_len
        self.sliding_window = sliding_window
        self.weather_inputs = weather_inputs
        if self.weather_inputs:
            assert self.weather_inputs == ['temperature'], \
                "Only temperature is supported by PandasTransformerDataset for now."
            # check that the weather inputs are in the dataframe
            assert all([col in self.df.columns for col in self.weather_inputs]), \
                f"Some weather_inputs {weather_inputs} are not in the dataframe `df`"

    def __len__(self):
        return (len(self.df) - self.context_len - self.pred_len) // self.sliding_window

    def __getitem__(self, idx):
        seq_ptr = self.context_len + self.sliding_window * idx
        load_features = self.df['load'].iloc[seq_ptr-self.context_len : seq_ptr+self.pred_len].values.astype(np.float32)
        building_features = self.df['building_type'].iloc[seq_ptr-self.context_len : seq_ptr+self.pred_len].values.astype(np.int32)
        latlon_features = self.df[['latitude', 'longitude']].iloc[seq_ptr-self.context_len : seq_ptr+self.pred_len].values.astype(np.float32)
        time_features = self.df[['day_of_year', 'day_of_week', 'hour_of_day']].iloc[seq_ptr-self.context_len : seq_ptr+self.pred_len].values.astype(np.float32)
        sample = {
            'latitude': latlon_features[:, 0][...,None],
            'longitude': latlon_features[:, 1][...,None],
            'day_of_year': time_features[:, 0][...,None],
            'day_of_week': time_features[:, 1][...,None],
            'hour_of_day': time_features[:, 2][...,None],
            'building_type': building_features[...,None],
            'load': load_features[...,None]
        }
        if self.weather_inputs:
            for col in self.weather_inputs:
                sample.update({col : self.df[col].iloc[seq_ptr-self.context_len: seq_ptr+self.pred_len].values.astype(np.float32)[...,None]}) 
        
        return sample


def keep_buildings(dataset_generator: Union[TorchBuildingDatasetsFromCSV, TorchBuildingDatasetFromParquet],
                     building_ids: List[str]) -> Union[TorchBuildingDatasetsFromCSV, TorchBuildingDatasetFromParquet]:
    """Remove all buildings *not* listed in building_ids from the building_datasets dictionary from the generator class.
    
    Args:
        dataset_generator (Union[TorchBuildingDatasetsFromCSV, TorchBuildingDatasetFromParquet]): Dataset generator class.
        building_ids (List[str]): List of building ids to keep.
    
    Returns:
        dataset_generator (Union[TorchBuildingDatasetsFromCSV, TorchBuildingDatasetFromParquet]): Dataset generator 
            class with only the buildings listed in building_ids.
    """
    for building_id in list(dataset_generator.building_datasets.keys()):
        if building_id not in building_ids:
            del dataset_generator.building_datasets[building_id]
    return dataset_generator
