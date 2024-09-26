import numpy as np
from pathlib import Path
import requests
import pandas as pd
import os
from tqdm import tqdm
from datetime import timezone, timedelta
from datetime import datetime as dt

def get_weather_pandas(path_to_csv: str) -> pd.DataFrame:
    weather_data = pd.read_csv(path_to_csv)
    
    # Unix timestamp to UTC-5
    weather_data['time'] = weather_data['time'].apply(lambda x: (dt.fromtimestamp(x, timezone.utc) - timedelta(hours=5)).strftime('%Y-%m-%d %H:%M:%S'))
    
    weather_data.rename(columns={'time' : 'timestamp'}, inplace=True)
    weather_data = weather_data.set_index('timestamp')
    weather_data.index = pd.to_datetime(weather_data.index, format='%Y-%m-%d %H:%M:%S')
    
    assert weather_data[weather_data.index.duplicated()].empty # making sure there's no duplicates

    weather_data = weather_data[['temperature', 'humidity']]
    weather_data['temperature'] = weather_data['temperature'].apply(lambda x: (x - 32) / 1.8) # 900K uses celcius
    weather_data['humidity'] = weather_data['humidity'].apply(lambda x: x * 100)
    
    return weather_data

def url_retrieve(url: str, outfile: Path):
    R = requests.get(url, allow_redirects=True)
    assert R.status_code == 200, f'Cannot download {url}'

    outfile.write_bytes(R.content)

def main():
    dataset_path = Path(os.environ.get('BUILDINGS_BENCH', ''))
    home = [('HomeB', [2014, 2015, 2016]), ('HomeC', [2014, 2015, 2016]), ('HomeD', [2015, 2016]), ('HomeF', [2014, 2015, 2016]), ('HomeG', [2015, 2016])]
    url_prefix = 'https://lass.cs.umass.edu/smarttraces/2017/'

    for h in tqdm(home):

        # Download files
        file_name = f'{h[0]}-weather.tar.gz'
        url_retrieve(f'{url_prefix}{file_name}', Path(file_name))
        os.system(f'tar -xzf {file_name}')
        os.system(f'rm {file_name}')

        # Process weather files
        weather_df = pd.DataFrame()
        for y in h[1]:
            df = get_weather_pandas(f'{h[0]}/{h[0][0].lower() + h[0][1:]}{y}.csv')
            weather_df = pd.concat([weather_df, df])

        # save in BuildingsBench
        weather_df.to_csv(dataset_path / f'SMART/weather_{h[0]}.csv')
        weather_df.to_csv(dataset_path / f'remove_outliers/SMART/weather_{h[0]}.csv')

        # clean up
        os.system(f'rm -r {h[0]}')
   
if __name__ == '__main__':
    main()
