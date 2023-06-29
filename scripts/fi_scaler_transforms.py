import argparse
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import random
import glob 
import os

from buildings_bench.transforms import StandardScalerTransform, BoxCoxTransform 


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(os.environ.get('BUILDINGS_BENCH', ''), 'metadata')
    # training set dir
    time_series_dir = Path(os.environ.get('BUILDINGS_BENCH', ''), 'Buildings-900K', 'end-use-load-profiles-for-us-building-stock', '2021')
    building_years = ['comstock_tmy3_release_1', 'resstock_tmy3_release_1', 'comstock_amy2018_release_1', 'resstock_amy2018_release_1'] 
    pumas = ['by_puma_midwest', 'by_puma_south', 'by_puma_northeast', 'by_puma_west']

    all_buildings = []

    for by in building_years:
        by_path = time_series_dir / by / 'timeseries_individual_buildings'
        for pum in pumas:
            pum_path = by_path / pum / 'upgrade=0'
            # subsample pumas for faster quantization
            pum_files = glob.glob(str(pum_path / 'puma=*'))
            random.shuffle(pum_files)
            # limit to 10 random pumas per
            pum_files = pum_files[:10]
            for pum_file in pum_files:
                # load the parquet file and convert each column to a numpy array
                #df = spark.read.parquet(pum_file)
                df = pq.read_table(pum_file).to_pandas()
                #df = df.toPandas()
                # convert each column to a numpy array and stack vertically
                all_buildings += [np.vstack([df[col].to_numpy() for col in df.columns if col != 'timestamp'])]



    print('Fitting StandardScaler...')
    ss = StandardScalerTransform()
    ss.train(np.vstack(all_buildings))
    ss.save(output_dir)
    print('StandardScaler: ', ss.mean_, ss.std_)

    print('Fitting BoxCox...')
    bc = BoxCoxTransform()
    bc.train(np.vstack(all_buildings))
    bc.save(output_dir)
    print('BoxCox: ', bc.lambdas_)
 

        
if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument('--seed', type=int, default=1, required=False,
                        help='Random seed shuffling. Default: 1')


    args = args.parse_args()

    main(args)
