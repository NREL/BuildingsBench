import numpy as np
from pathlib import Path
import requests
import pandas as pd
import os
from tqdm import tqdm
from scripts.process_raw_data.era5_cds import get_hourly_weather_pandas, AREA_LONLAT
import math

def calc_relative_humidity(temp, dewpoint):
    # https://earthscience.stackexchange.com/questions/16570/how-to-calculate-relative-humidity-from-temperature-dew-point-and-pressure
    import math
    b = 17.625
    c = 243.04
    return 100 * math.exp((c * b * (dewpoint - temp)) / ((c + temp) * (c + dewpoint)))

def main():
    real_building_prefix = Path(os.environ.get('BUILDINGS_BENCH', ''))
    real_building_ds = [('LCL', 'london', '2011-01-01', '2014-12-31', 0), ('IDEAL', 'edinburg', '2016-01-01', '2018-12-31', 0), 
                      ('Sceaux', 'sceaux', '2006-12-31', '2011-01-01', 1), ('Borealis', 'waterloo', '2010-12-31', '2013-01-01', -5),
                      ('Electricity', 'lisbon', '2011-01-01', '2014-12-31', 0), ('SMART', 'massachusetts', '2013-12-31', '2017-01-01', -5),
                      ('Panther', 'ucf', '2015-12-31', '2018-01-01', -5), ('Fox', 'asu', '2015-12-31', '2018-01-01', -7), 
                      ('Bear', 'uc-b', '2015-12-31', '2018-01-01', -8), ('Rat', 'dc', '2015-12-31', '2018-01-01', -5)] 
    
    BDG = {'Panther', 'Fox', 'Bear', 'Rat'}
 
    for ds in real_building_ds:
        print('Downloading', ds[0])

        if ds[0] in BDG:
            output_dir = real_building_prefix / 'BDG-2'
        else:
            output_dir = real_building_prefix / ds[0]

        weather = get_hourly_weather_pandas(ds[2], ds[3], AREA_LONLAT[ds[1]], variable=['2m_temperature','2m_dewpoint_temperature'])

        weather['2m_temperature'] = weather['2m_temperature'].apply(lambda x: x-273.15)
        weather['2m_dewpoint_temperature'] = weather['2m_dewpoint_temperature'].apply(lambda x: x-273.15)

        weather['humidity'] = weather.apply(lambda x: calc_relative_humidity(x['2m_temperature'], x['2m_dewpoint_temperature']), axis=1)
        weather.drop(columns=['2m_dewpoint_temperature'], inplace=True)
        weather.rename(columns={'2m_temperature' : 'temperature'}, inplace=True)

        weather.index.rename('timestamp', inplace=True)
        weather.index = weather.index + pd.to_timedelta(f'{ds[4]}h')

        if ds[0] in BDG:
            weather.to_csv(output_dir / f'weather_{ds[0]}_era5.csv')
        else:
            weather.to_csv(output_dir / 'weather_era5.csv')


   
if __name__ == '__main__':
    main()
