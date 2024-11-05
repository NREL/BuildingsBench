import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import random
import glob 
import os

from buildings_bench.transforms import StandardScalerTransform


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(os.environ.get('BUILDINGS_BENCH', ''), 'metadata', 'transforms', 'weather')
    # training set dir
    time_series_dir = Path(os.environ.get('BUILDINGS_BENCH', ''), 'Buildings-900K', 'end-use-load-profiles-for-us-building-stock', '2021')
    building_years = ['comstock_tmy3_release_1', 'resstock_tmy3_release_1', 'comstock_amy2018_release_1', 'resstock_amy2018_release_1'] 

    weather_columns = ['timestamp', 'temperature', 'humidity', 'wind_speed', 'wind_direction', 'global_horizontal_radiation', 
                              'direct_normal_radiation', 'diffuse_horizontal_radiation']

    weather_df = pd.DataFrame(columns=weather_columns)

    for by in building_years:
        weather_path = time_series_dir / by / 'weather'
        # subsample county for faster quantization
        county_files = glob.glob(str(weather_path / '*.csv'))
        random.shuffle(county_files)
        # limit to 10 random counties
        county_files = county_files[:10]
        for cf in county_files:
            # load the weather file
            df = pd.read_csv(cf)
            df.columns = weather_columns
            # append to weather_df
            weather_df = pd.concat([weather_df, df], ignore_index=True)


    for col in weather_columns[1:]:
        print('Fitting StandardScaler...', col)
        ss = StandardScalerTransform()
        ss.train(weather_df[col].to_numpy())
        ss.save(output_dir / col)
        print('StandardScaler: ', ss.mean_, ss.std_)

    # print('Fitting BoxCox...')
    # bc = BoxCoxTransform()
    # bc.train(np.vstack(all_buildings))
    # bc.save(output_dir)
    # print('BoxCox: ', bc.boxcox.lambdas_)

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument('--seed', type=int, default=1, required=False,
                        help='Random seed shuffling. Default: 1')


    args = args.parse_args()

    main(args)