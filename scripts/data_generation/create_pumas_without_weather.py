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
    out_filter_file = metadata_dir / 'pumas_without_weather.pkl'

    # select rows that don't have weather
    df_no_weather = lookup_df[(lookup_df.weather_file_2012 == 'No weather file') | (lookup_df.weather_file_2015 == 'No weather file') | (lookup_df.weather_file_2016 == 'No weather file') | (lookup_df.weather_file_2017 == 'No weather file') | (lookup_df.weather_file_2018 == 'No weather file') | (lookup_df.weather_file_2019 == 'No weather file')]

    # select rows that have weather
    df_has_weather = lookup_df[(lookup_df.weather_file_2012 != 'No weather file') & (lookup_df.weather_file_2015 != 'No weather file') & (lookup_df.weather_file_2016 != 'No weather file') & (lookup_df.weather_file_2017 != 'No weather file') & (lookup_df.weather_file_2018 != 'No weather file') & (lookup_df.weather_file_2019 != 'No weather file')]

    has_weather = set(df_has_weather['nhgis_2010_puma_gisjoin'])
    no_weather = set(df_no_weather['nhgis_2010_puma_gisjoin'])

    withheld_puma = sorted(no_weather - has_weather)
    for i in range(len(withheld_puma)):
        withheld_puma[i] = f'puma={withheld_puma[i]}'

    with open(out_filter_file, 'wb') as fp:
        pickle.dump(withheld_puma, fp) 

if __name__ == '__main__':
    main()
