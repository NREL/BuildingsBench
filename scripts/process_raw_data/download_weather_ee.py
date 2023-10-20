import numpy as np
from pathlib import Path
import requests
import pandas as pd
import os
from tqdm import tqdm
import math
from eeweather import ISDStation
from datetime import datetime as dt
import pytz

def main():
    real_building_prefix = Path(os.environ.get('BUILDINGS_BENCH', ''))

    real_building_ds = [('LCL', '037700', '2011-01-01', '2014-12-31', 0), ('IDEAL', '031660', '2017-01-01', '2018-12-31', 0), 
                      ('Sceaux', '071560', '2006-12-31', '2011-01-01', 1), ('Borealis', '713680', '2010-12-31', '2013-01-01', -5),
                      ('SMART', '744910', '2013-12-31', '2017-01-01', -5)] 
    
    for ds in real_building_ds:
        print('Downloading', ds[0])

        output_dir = real_building_prefix / ds[0]

        station = ISDStation(ds[1])
        start_date = dt.strptime(ds[2], '%Y-%m-%d').replace(tzinfo=pytz.UTC)
        end_date = dt.strptime(ds[3], '%Y-%m-%d').replace(tzinfo=pytz.UTC, hour=23)

        weather, warnings = station.load_isd_hourly_temp_data(start_date, end_date)
        print(warnings)

        weather = weather.to_frame()
        weather.columns = ['temperature']
        weather.index.rename('timestamp', inplace=True)
        weather.interpolate('linear', inplace=True)
        weather.index = weather.index.tz_localize(None)
        weather.index = weather.index + pd.to_timedelta(f'{ds[4]}h')

        weather.to_csv(output_dir / 'weather_isd.csv')
        # weather.to_csv(real_building_prefix / 'remove_outliers' / ds[0] / f'weather_isd.csv')


   
if __name__ == '__main__':
    main()
