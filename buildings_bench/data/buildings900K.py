import torch
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq
import buildings_bench.transforms as transforms
from buildings_bench.transforms import BoxCoxTransform, StandardScalerTransform
import pandas as pd
from typing import List
from sklearn.preprocessing import OneHotEncoder
from buildings_bench.data.buildings_utils import *

class Buildings900K(torch.utils.data.Dataset):
    r"""This is an indexed dataset for the Buildings-900K dataset.
    It uses an index file to quickly load a sub-sequence from a time series in a multi-building
    Parquet file. The index file is a tab separated file with the following columns:

    0. Building-type-and-year (e.g., comstock_tmy3_release_1)
    1. Census region (e.g., by_puma_midwest)
    2. PUMA ID
    3. Building ID
    4. Hour of year pointer (e.g., 0070)

    The sequence pointer is used to extract the slice
    [pointer - context length : pointer + pred length] for a given building ID.
    
    The time series are not stored chronologically and must be sorted by timestamp after loading.

    Each dataloader worker has its own file pointer to the index file. This is to avoid
    weird multiprocessing errors from sharing a file pointer. We 'seek' to the correct
    line in the index file for random access.
    """
    def __init__(self, 
                dataset_path: Path,
                index_file: str,
                context_len: int = 168,
                pred_len: int = 24,
                apply_scaler_transform: str = '',
                scaler_transform_path: Path = None,
                weather: List[str] = None,
                use_buildings_chars : bool = False,
                use_text_embedding: bool = False,
                surrogate_mode: bool = False):
        """
        Args:
            dataset_path (Path): Path to the pretraining dataset.
            index_file (str): Name of the index file
            context_len (int, optional): Length of the context. Defaults to 168. 
                The index file has to be generated with the same context length.
            pred_len (int, optional): Length of the prediction horizon. Defaults to 24.
                The index file has to be generated with the same pred length.
            apply_scaler_transform (str, optional): Apply a scaler transform to the load. Defaults to ''.
            scaler_transform_path (Path, optional): Path to the scaler transform. Defaults to None.
            weather (List[str]): list of weather features to use. Default: None.
            use_buildings_chars (bool): whether include building characteristics.
            use_text_embedding (bool): whether encode building characteristics with text embeddings. Default: False.
            surrogate_mode (bool): whether enable surrogate mode, which expects index_file to have one building per row. Default: False.
        """
        self.dataset_path = dataset_path / 'Buildings-900K' / 'end-use-load-profiles-for-us-building-stock' / '2021'
        self.metadata_path = dataset_path / 'metadata_dev'
        self.context_len = context_len
        self.pred_len = pred_len
        self.building_type_and_year = ['comstock_tmy3_release_1',
                                       'resstock_tmy3_release_1',
                                       'comstock_amy2018_release_1',
                                       'resstock_amy2018_release_1']
        self.census_regions = ['by_puma_midwest', 'by_puma_south', 'by_puma_northeast', 'by_puma_west']

        self.surrogate_mode = surrogate_mode
        self.index_file = self.metadata_path / index_file
        self.index_fp = None
        self.__read_index_file(self.index_file)
        self.weather = weather
        self.time_transform = transforms.TimestampTransform()
        self.spatial_transform = transforms.LatLonTransform()
        self.apply_scaler_transform = apply_scaler_transform
        self.use_buildings_chars = use_buildings_chars
        self.use_text_embedding = use_text_embedding

        # calculate total hours, only used for surrogate mode
        if surrogate_mode:
            assert self.context_len == 0
            self.total_hours = int((pd.Timestamp('2018-12-31') - pd.Timestamp('2018-01-01')).total_seconds() / 3600)
            np.random.seed(0)

        if self.apply_scaler_transform == 'boxcox':
            self.load_transform = BoxCoxTransform()
            self.load_transform.load(scaler_transform_path)
        elif self.apply_scaler_transform == 'standard':
            self.load_transform = StandardScalerTransform()
            self.load_transform.load(scaler_transform_path)

        if self.use_buildings_chars:
            # read categorical meta data characteristics of commercial buildings
            df1 = pd.read_parquet(self.metadata_path / "comstock_amy2018.parquet", engine="pyarrow")
            df2 = pd.read_parquet(self.metadata_path / "comstock_tmy3.parquet", engine="pyarrow")
            df = pd.concat([df1, df2])
            self.com_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
            self.com_encoder.fit(df[com_chars].values)
            self.com_num = len(com_chars)
            self.com_dim = self.com_encoder.transform([[None] * self.com_num]).shape

            # read categorical meta data characteristics of residential buildings
            df1 = pd.read_parquet(self.metadata_path / "resstock_amy2018.parquet", engine="pyarrow")
            df2 = pd.read_parquet(self.metadata_path / "resstock_tmy3.parquet", engine="pyarrow")
            df = pd.concat([df1, df2])
            self.res_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
            self.res_encoder.fit(df[res_chars].values)
            self.res_num = len(res_chars)
            self.res_dim = self.res_encoder.transform([[None] * self.res_num]).shape

            self.meta_dfs = []
            self.meta_dataset_names = ["comstock_tmy3", "resstock_tmy3", "comstock_amy2018", "resstock_amy2018"]
            for name in self.meta_dataset_names:
                df = pd.read_parquet(self.metadata_path / f"{name}.parquet", engine="pyarrow")
                self.meta_dfs.append(df)


        if weather: # build a puma-county lookup table
            # lookup_df = pd.read_csv(self.metadata_path / 'puma_county_lookup_weather_only.csv', index_col=0)
            lookup_df = pd.read_csv(self.metadata_path / 'spatial_tract_lookup_table.csv')

            # select rows that have weather
            df_has_weather = lookup_df[(lookup_df.weather_file_2012 != 'No weather file') 
                                       & (lookup_df.weather_file_2015 != 'No weather file') 
                                       & (lookup_df.weather_file_2016 != 'No weather file') 
                                       & (lookup_df.weather_file_2017 != 'No weather file') 
                                       & (lookup_df.weather_file_2018 != 'No weather file') 
                                       & (lookup_df.weather_file_2019 != 'No weather file')]

            df_has_weather = df_has_weather[['nhgis_2010_county_gisjoin', 'nhgis_2010_puma_gisjoin']]
            df_has_weather = df_has_weather.set_index('nhgis_2010_puma_gisjoin')
            self.lookup_df = df_has_weather[~df_has_weather.index.duplicated()] # remove duplicated indices


    def init_fp(self):
        """Each worker needs to open its own file pointer to avoid 
        weird multiprocessing errors from sharing a file pointer.

        This is not called in the main process.
        This is called in the DataLoader worker_init_fn.
        The file is opened in binary mode which lets us disable buffering.
        """
        self.index_fp = open(self.index_file, 'rb', buffering=0)
        self.index_fp.seek(0)

    def __read_index_file(self, index_file: Path) -> None:
        """Extract metadata from index file.
        """
        # Fast solution to get the number of time series in index file
        # https://pynative.com/python-count-number-of-lines-in-file/
        
        def _count_generator(reader):
            b = reader(1024 * 1024)
            while b:
                yield b
                b = reader(1024 * 1024)

        with open(index_file, 'rb') as fp:
            c_generator = _count_generator(fp.raw.read)
            # count each \n
            self.num_time_series = sum(buffer.count(b'\n') for buffer in c_generator)
        
        # Count the number of chars per line
        with open(index_file, 'rb', buffering=0) as fp:
            first_line = fp.readline()
            self.chunk_size = len(first_line)
        
        #print(f'Counted {self.num_time_series} indices in index file.')

    def __del__(self):
        if self.index_fp:
            self.index_fp.close()

    def __len__(self):
        return self.num_time_series
    
    def get_sample(self, df, bldg_id, seq_ptr, dataset_id, puma):
        # Slice each column from seq_ptr-context_len : seq_ptr + pred_len
        time_features = self.time_transform.transform(df['timestamp'].iloc[seq_ptr-self.context_len : seq_ptr+self.pred_len ])
        # Running faiss on CPU is  slower than on GPU, so we quantize loads later after data loading.
        load_features = df[bldg_id].iloc[seq_ptr-self.context_len : seq_ptr+self.pred_len ].values.astype(np.float32)  # (context_len+pred_len,)
        # For BoxCox transform
        if self.apply_scaler_transform != '':
            load_features = self.load_transform.transform(load_features)
        latlon_features = self.spatial_transform.transform(puma).repeat(self.context_len + self.pred_len, axis=0) 
        
        # residential = 0 and commercial = 1
        building_features = np.ones((self.context_len + self.pred_len,1), dtype=np.int32) * int(dataset_id % 2 == 0)

        sample = {
            'latitude': latlon_features[:, 0][...,None],
            'longitude': latlon_features[:, 1][...,None],
            'day_of_year': time_features[:, 0][...,None],
            'day_of_week': time_features[:, 1][...,None],
            'hour_of_day': time_features[:, 2][...,None],
            'building_type': building_features,
            'load': load_features[...,None]
        }

        if self.use_buildings_chars:
            # one-hot encode meta data characteristics
            # text embedding
            df = self.meta_dfs[dataset_id]
            # if commercial
            if dataset_id % 2 == 0:
                # df = self.meta_dfs[int(ts_idx[0])]
                ch = df[df.index == int(bldg_id)][com_chars].values
                ft = np.hstack([self.com_encoder.transform(ch), np.zeros(self.res_dim)])
                bd_subtype = df[df.index == int(bldg_id)]["in.building_type"].values[0]
                bd_subtype = list(self.com_encoder.categories_[1]).index(bd_subtype)
            else:
                # df = self.meta_dfs[int(ts_idx[0])]
                ch = df[df.index == int(bldg_id)][res_chars].values
                ft = np.hstack([np.zeros(self.com_dim), self.res_encoder.transform(ch)])
                bd_subtype = -1

            # overwrite ft if use text embedding
            if self.use_text_embedding:
                ft = np.load(self.metadata_path / "simcap" / self.meta_dataset_names[dataset_id] / f"{bldg_id}_emb.npy")
                ft = np.expand_dims(ft, axis=0)

            sample['building_char']    = np.repeat(ft, self.context_len + self.pred_len, axis=0).astype(np.float32)
            sample['building_id']      = int(bldg_id)
            sample['dataset_id']       = int(dataset_id)
            sample["building_subtype"] = bd_subtype

        if self.weather is None:
            return sample
        
        ## Append weather features

        # get county ID
        county = self.lookup_df.loc[puma]['nhgis_2010_county_gisjoin']

        # load corresponding weather files
        weather_df = pd.read_csv(str(self.dataset_path / self.building_type_and_year[dataset_id] / 'weather' / f'{county}.csv'))

        # This is assuming that the file always starts from January 1st (ignoring the year)
        import datetime
        assert datetime.datetime.strptime(weather_df['date_time'].iloc[0], '%Y-%m-%d %H:%M:%S').strftime('%m-%d') == '01-01',\
            "The weather file does not start from Jan 1st"
        
        weather_df.columns = ['timestamp', 'temperature', 'humidity', 'wind_speed', 'wind_direction', 'global_horizontal_radiation', 
                        'direct_normal_radiation', 'diffuse_horizontal_radiation']
        weather_df = weather_df[['timestamp'] + self.weather]

        weather_df = weather_df.iloc[seq_ptr-self.context_len-1: seq_ptr+self.pred_len-1] # add -1 because the file starts from 01:00:00
        
        # convert temperature to fahrenheit (note: keep celsius for now)
        # weather_df['temperature'] = weather_df['temperature'].apply(lambda x: x * 1.8 + 32) 

        # transform
        weather_transform = StandardScalerTransform()
        for col in weather_df.columns[1:]:
            weather_transform.load(self.metadata_path / 'transforms/weather-900K/' / col)
            sample.update({col : weather_transform.transform(weather_df[col].to_numpy())[0][...,None]})

        return sample

    def __getitem__(self, idx):
        # Open file pointer if not already open
        if not self.index_fp:
           self.index_fp = open(self.index_file, 'rb', buffering=0)
           self.index_fp.seek(0)

        # Get the index of the time series
        self.index_fp.seek(idx * self.chunk_size, 0)
        ts_idx = self.index_fp.read(self.chunk_size).decode('utf-8')

        # Parse the index
        ts_idx = ts_idx.strip('\n').split('\t')

        # strip loading zeros
        if self.surrogate_mode:
            # starting from 1 to skip the first hour of the first day
            seq_ptr = np.random.randint(1, self.total_hours - self.pred_len)
        else:
            seq_ptr = ts_idx[-1]
            seq_ptr = int(seq_ptr.lstrip('0')) if seq_ptr != '0000' else 0

        # Building ID
        bldg_id = ts_idx[3].lstrip('0')

        # Select timestamp and building column
        df = pq.read_table(str(self.dataset_path / self.building_type_and_year[int(ts_idx[0])]
                        / 'timeseries_individual_buildings' / self.census_regions[int(ts_idx[1])]
                        / 'upgrade=0' / f'puma={ts_idx[2]}'), columns=['timestamp', bldg_id])

        # Order by timestamp
        df = df.to_pandas().sort_values(by='timestamp')
        
        dataset_id = int(ts_idx[0])
        puma       = ts_idx[2]
        sample = self.get_sample(df, bldg_id, seq_ptr, dataset_id, puma)

        return sample
