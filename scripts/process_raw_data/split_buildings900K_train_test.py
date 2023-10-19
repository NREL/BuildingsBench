import argparse
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import random
import glob 
import pandas as pd
import os
from tqdm import tqdm
import shutil

def main():

    output_dir = Path(os.environ.get('BUILDINGS_BENCH', ''), 'metadata')
    time_series_dir = Path(os.environ.get('BUILDINGS_BENCH', ''), 'Buildings-900K', 'end-use-load-profiles-for-us-building-stock', '2021')
    out_dir = Path(os.environ.get('BUILDINGS_BENCH', ''), 'Buildings-900K-test', '2021')
    building_years = ['comstock_tmy3_release_1', 'resstock_tmy3_release_1', 'comstock_amy2018_release_1', 'resstock_amy2018_release_1'] 
    pumas = ['by_puma_midwest', 'by_puma_south', 'by_puma_northeast', 'by_puma_west']
    

    # withhold 1 puma from each census region (all res and com buildingss) for test only
    # midwest, south, northeast, west
    # read withheld pumas from file
    with open(Path(os.environ.get('BUILDINGS_BENCH','')) / 'metadata' / 'withheld_pumas.tsv', 'r') as f:
        # tab separated file
        line = f.readlines()[0]
        withheld_pumas = line.strip('\n').split('\t')
    print(f'Withheld pumas: {withheld_pumas}')


    for building_type_and_year, by in enumerate(building_years): 
        # for each census region, check if any withheld puma is in the census region
        by_path = time_series_dir / by / 'timeseries_individual_buildings'
        benchmark_by_path = out_dir / by / 'timeseries_individual_buildings'
        for census_region_idx, pum in enumerate(pumas):
            pum_path = by_path / pum / 'upgrade=0'
            benchmark_by_pum_path = benchmark_by_path / pum / 'upgrade=0'
            pum_files = [os.path.basename(x) for x in glob.glob(str(pum_path / 'puma=*'))]
            for withheld_puma in withheld_pumas:
                if withheld_puma in pum_files:
                    benchmark_by_pum_path.mkdir(exist_ok=True, parents=True)
                    # Move directory pum_path / withheld_puma to benchmark_by_pum_path                       
                    shutil.copytree(pum_path / withheld_puma, benchmark_by_pum_path / withheld_puma)
                    shutil.rmtree(pum_path / withheld_puma)
                    print(f'Moved {pum_path / withheld_puma} to {benchmark_by_pum_path / withheld_puma}')
   

if __name__ == '__main__':
    main()
