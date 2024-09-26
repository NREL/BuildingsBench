import argparse
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import random
import glob 
import pandas as pd
import os
from tqdm import tqdm
import pickle


def main():
    metadata_dir = Path(os.environ.get('BUILDINGS_BENCH', ''), 'metadata')

    lookup_df = pd.read_csv(metadata_dir / 'spatial_tract_lookup_table.csv')
    out_lookup_file = metadata_dir / 'puma_county_lookup_weather_only.csv'

    # select rows that have weather
    df_has_weather = lookup_df[(lookup_df.weather_file_2012 != 'No weather file') & (lookup_df.weather_file_2015 != 'No weather file') & (lookup_df.weather_file_2016 != 'No weather file') & (lookup_df.weather_file_2017 != 'No weather file') & (lookup_df.weather_file_2018 != 'No weather file') & (lookup_df.weather_file_2019 != 'No weather file')]

    df_has_weather = df_has_weather[['nhgis_2010_county_gisjoin', 'nhgis_2010_puma_gisjoin']]
    df_has_weather = df_has_weather.set_index('nhgis_2010_puma_gisjoin')
    df_has_weather = df_has_weather[~df_has_weather.index.duplicated()] # remove duplicated indices
    df_has_weather.to_csv(out_lookup_file)

if __name__ == '__main__':
    main()
