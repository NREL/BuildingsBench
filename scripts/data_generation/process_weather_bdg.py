import numpy as np
from pathlib import Path
import requests
import pandas as pd
import os
from tqdm import tqdm
from datetime import timezone, timedelta
from datetime import datetime as dt
import argparse


def calc_relative_humidity(temp, dewpoint):
    # https://earthscience.stackexchange.com/questions/16570/how-to-calculate-relative-humidity-from-temperature-dew-point-and-pressure
    import math
    b = 17.625
    c = 243.04
    return 100 * math.exp((c * b * (dewpoint - temp)) / ((c + temp) * (c + dewpoint)))

def main(args):
    sites = args.sites
    dataset_path = Path(os.environ.get('BUILDINGS_BENCH', ''))
    
    weather_df = pd.read_csv(args.weather_csv, index_col=0)
    weather_df.index = pd.to_datetime(weather_df.index, format='%Y-%m-%d %H:%M:%S')

    for site in tqdm(sites):
        weather_site = weather_df[weather_df['site_id'] == site]
        weather_site = weather_site[['airTemperature', 'dewTemperature']]

        assert weather_site[weather_site.index.duplicated()].empty

        weather_site.interpolate('linear', inplace=True)
        weather_site['humidity'] = weather_site.apply(lambda x: calc_relative_humidity(x['airTemperature'], x['dewTemperature']), axis=1)
        weather_site.drop(columns=['dewTemperature'], inplace=True)
        weather_site.rename(columns={'airTemperature' : 'temperature'}, inplace=True)

        weather_site.to_csv(dataset_path / f'BDG-2/weather_{site}.csv')
        weather_site.to_csv(dataset_path / f'remove_outliers/BDG-2/weather_{site}.csv')

   
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--weather-csv', type=str, required=True,
                        help='Path to BDG\'s weather file (.csv)')
    args.add_argument('-s', '--sites', nargs='+', type=str, default=['Bear', 'Fox', 'Panther', 'Rat'],
                        help="List of BDG sites to process. Default: Bear Fox Pather Rat")
    args = args.parse_args()
    main(args)
