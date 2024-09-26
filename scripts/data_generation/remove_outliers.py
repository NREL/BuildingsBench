"""
Process the unfiltered BuildingsBench evaluation data by removing outliers
and store the filtered data in a subfolder called "remove_outliers"
"""
from pathlib import Path 
import os 
import numpy as np
import glob
import pandas as pd


def distance_filter(series, window_size):
    # make a new series in which we overwrite the outliers
    new_series = series.copy()

    ignore = np.zeros((window_size,)) # ignore the center point
    ignore[window_size//2] = 100000

    # For each point in the series, compute the 1-nearest-neighbor
    # distance in the window
    diff =series.rolling(window=window_size, center=True).apply(
        lambda x: (ignore+np.abs(x - x[window_size//2])).min()
    )
    # select threshold 
    average_gap = series.rolling(window=24, center=True).max().mean() - series.rolling(window=24, center=True).min().mean()

    print('average_gap', average_gap)
    mask = diff > average_gap
    
    # calculate the median of the data in the neighborhood
    rolling_median = series.rolling(window=window_size, center=True).median()
    new_series[mask] = rolling_median[mask]

    print('Share of outliers in the series:', mask.mean())

    return new_series, mask


if __name__ == '__main__':
    """
    Creating a copy of BuildingsBench metered datasets with outliers removed
    """
    benchmarks = [
        'BDG-2',
        'Borealis',
        'Electricity',
        'IDEAL',
        'LCL',
        'Sceaux',
        'SMART'
    ]

    dataset_dir = Path(os.environ.get('BUILDINGS_BENCH', ''))
    output_dir = Path(os.environ.get('BUILDINGS_BENCH', '')) / 'remove_outliers'
    output_dir.mkdir(exist_ok=True)

    for benchmark in benchmarks:
        output_dir_ = output_dir / benchmark
        output_dir_.mkdir(exist_ok=True)
        # glob all files
        files = glob.glob(str(dataset_dir / benchmark / '*.csv'))
        
        if benchmark == 'BDG-2' or benchmark == 'Electricity': # multi-building
            for file in files:
                multi_building_df = pd.read_csv(file, index_col=0, parse_dates=True)
                for building in multi_building_df.columns:
                    df = multi_building_df[building]
                    print(file)
                    filtered_load, mask = distance_filter(df, 25)
                    multi_building_df[building] = filtered_load
                multi_building_df.to_csv(output_dir_ / Path(file).name)        
        else:
            
            for file in files:
                df = pd.read_csv(file, index_col=0, parse_dates=True)
                print(file)
                # get column name
                col_name = df.columns[0]
                filtered_load, mask = distance_filter(df[col_name], 25)
                df[col_name] = filtered_load
                df.to_csv(output_dir_ / Path(file).name)