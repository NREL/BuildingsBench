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


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(os.environ.get('BUILDINGS_BENCH', ''), 'metadata_dev')
    time_series_dir = Path(os.environ.get('BUILDINGS_BENCH', ''), 'Buildings-900K', 'end-use-load-profiles-for-us-building-stock', '2021')
    building_years = ['comstock_tmy3_release_1', 'resstock_tmy3_release_1', 'comstock_amy2018_release_1', 'resstock_amy2018_release_1'] 
    pumas = ['by_puma_midwest', 'by_puma_south', 'by_puma_northeast', 'by_puma_west']
    
    # Each line in the index file indicates a building and n 
    #   <building_type_and_year> <census_region> <puma_id> <building_id> <seq ptr>
    #  e.g. <0-4> <0-4> G17031 23023 65
    train_idx_file = open(output_dir / f'train_simcap_300k.idx', 'w')
    val_idx_file  = open(output_dir / f'val_simcap_300k.idx', 'w')
    test_idx_file = open(output_dir / f'test_simcap_300k.idx', 'w')

    # withhold 1 puma from each census region (all res and com buildingss) for test only
    # midwest, south, northeast, west
    # read withheld pumas from file
    with open(Path(os.environ.get('BUILDINGS_BENCH','')) / 'metadata_dev' / 'withheld_pumas.tsv', 'r') as f:
        # tab separated file
        line = f.readlines()[0]
        withheld_pumas = line.strip('\n').split('\t')

    # withhold pumas without weather
    with open(Path(os.environ.get('BUILDINGS_BENCH','')) / 'metadata_dev' / 'pumas_without_weather.pkl', 'rb') as f:
        without_weather_pumas = pickle.load(f)
    print(f'Withheld pumas: {withheld_pumas}')
    print(f'Without weather puma: {without_weather_pumas}')

    # 2 weeks heldout for val
    train_tmy_timerange = (pd.Timestamp('2018-01-01'), pd.Timestamp('2018-12-31'))
    # train_amy2018_timerange = (pd.Timestamp('2018-01-01'), pd.Timestamp('2018-12-17'))
    # val_timerange = (pd.Timestamp('2018-12-17'), pd.Timestamp('2018-12-31'))
    

    for building_type_and_year, by in enumerate(building_years):
        by_path = time_series_dir / by / 'timeseries_individual_buildings'

        # building ids contained in the simcap df
        bd_ids = pd.read_csv(output_dir / "simcap" / f"{by[:-10]}_simcap_300000.csv")["bldg_id"].values
        bd_ids = np.array(bd_ids)

        # randomly split train (60%) and val (20%) test (20%) sets
        n = len(bd_ids)
        idx = np.random.permutation(n)
        train_idx, val_idx, _ = np.split(idx, [int(n * 0.8), n])
        train_idx = bd_ids[train_idx]
        val_idx   = bd_ids[val_idx]

        train_hours = int((train_tmy_timerange[1] - train_tmy_timerange[0]).total_seconds() / 3600)

        for census_region_idx, pum in enumerate(pumas): # census regions
            pum_path = by_path / pum / 'upgrade=0'
            pum_files = glob.glob(str(pum_path / 'puma=*'))

            for pum_file in tqdm(pum_files):
                try:
                    bldg_ids = pq.read_table(pum_file).to_pandas().columns[1:]
                except:
                    print(f'Failed to read {pum_file}')
                    import pdb; pdb.set_trace()
                    continue
                
                # skip pumas without weather data
                if os.path.basename(pum_file) in without_weather_pumas:
                    continue

                for bldg_id in bldg_ids:
                    if int(bldg_id) in train_idx or int(bldg_id) in val_idx:
                        # train or test
                        bldg_id = bldg_id.zfill(6)
                        # Sample the starting index between (0,24)
                        s_start = 1 # skip the 00:00:00 timestamp in the beginning of the file
                        s_idx = s_start                           
                        seq_ptr = str(args.context_len + s_idx).zfill(4)  # largest seq ptr is < 10000
                        # NB: We don't *need* \n at the end of each line, but it makes it easier to count # of lines for dataloading
                        linestr = f'{building_type_and_year}\t{census_region_idx}\t{os.path.basename(pum_file).split("=")[1]}\t{bldg_id}\t{seq_ptr}\n'
                        assert len(linestr) == 26, f'linestr: {linestr}'
                        if not os.path.basename(pum_file) in withheld_pumas:
                            if int(bldg_id) in train_idx:
                                train_idx_file.write(linestr)
                            else:
                                val_idx_file.write(linestr)
                        else:
                            test_idx_file.write(linestr)

    # Close files
    train_idx_file.close()
    val_idx_file.close()
    test_idx_file.close()
   
if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument('--seed', type=int, default=1, required=False,
                        help='Random seed for KMeans and shuffling. Default: 1')
    args.add_argument('--sliding_window_stride', type=int, default=24, required=False,
                        help='Stride for sliding window to split timeseries into training examples. Default: 24 hours')
    args.add_argument('--context_len', type=int, default=168, required=False,
                                help='Length of context sequence. For handling year beginning and year end. Default: 168 hours')
    args.add_argument('--pred_len', type=int, default=24, required=False,
                                help='Length of prediction sequence. For handling year beginning and year end. Default: 24 hours')

    args = args.parse_args()

    main(args)
