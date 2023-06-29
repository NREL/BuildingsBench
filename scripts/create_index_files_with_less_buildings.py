import argparse
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import random
import glob 
import pandas as pd
import os
from tqdm import tqdm


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_path = Path(os.environ.get('BUILDINGS_BENCH', ''))
            
    # Each line in the index file indicates a building and n 
    #   <building_type_and_year> <census_region> <puma_id> <building_id> <seq ptr>
    #  e.g. <0-4> <0-4> G17031 23023 65
    train_idx_file = open(dataset_path / 'metadata' / f'train_weekly_{args.num_buildings}.idx', 'w')
    val_idx_file = open(dataset_path / 'metadata' / f'val_weekly_{args.num_buildings}.idx', 'w')
    building_years = ['comstock_tmy3_release_1', 'resstock_tmy3_release_1', 'comstock_amy2018_release_1', 'resstock_amy2018_release_1'] 
    pumas = ['by_puma_midwest', 'by_puma_south', 'by_puma_northeast', 'by_puma_west']

    # withhold 1 puma from each census region (all res and com buildingss) for test only
    # midwest, south, northeast, west
    # read withheld pumas from file
    with open(dataset_path / 'metadata' / 'withheld_pumas.tsv', 'r') as f:
        # tab separated file
        line = f.readlines()[0]
        withheld_pumas = line.strip('\n').split('\t')
    print(f'Withheld pumas: {withheld_pumas}')


    # 2 weeks heldout for val
    train_tmy_timerange = (pd.Timestamp('2018-01-01'), pd.Timestamp('2018-12-31'))
    train_amy2018_timerange = (pd.Timestamp('2018-01-01'), pd.Timestamp('2018-12-17'))
    val_timerange = (pd.Timestamp('2018-12-17'), pd.Timestamp('2018-12-31'))

    bldgs_per_bty = args.num_buildings // len(building_years)

    for building_type_and_year, by in enumerate(building_years):
        num_buildings = 0

        by_path = dataset_path / 'Buildings-900K' / 'end-use-load-profiles-for-us-building-stock/2021' \
              / by / 'timeseries_individual_buildings'

        if 'amy2018' in by:
            train_hours = int((train_amy2018_timerange[1] - train_amy2018_timerange[0]).total_seconds() / 3600)
            print(f'AMY2018 train hours: {train_hours}')

            val_hours = int((val_timerange[1] - val_timerange[0]).total_seconds() / 3600)
            print(f'AMY2018 Val hours: {val_hours}')
        else:
            train_hours = int((train_tmy_timerange[1] - train_tmy_timerange[0]).total_seconds() / 3600)
            print(f'TMY train hours: {train_hours}')

        # select buildings from census regions in random order,
        # until we have enough buildings. alternatively, we could
        # randomly sample the census region for each building.
        census_region_idxs = [0, 1, 2, 3]
        random.shuffle(census_region_idxs)

        for census_region_idx in census_region_idxs: # census regions
            pum_path = by_path / pumas[census_region_idx] / 'upgrade=0'
            pum_files = glob.glob(str(pum_path / 'puma=*'))

            # shuffle order of pum_files
            random.shuffle(pum_files)

            for pum_file in tqdm(pum_files):
                try:
                    bldg_ids = pq.read_table(pum_file).to_pandas().columns[1:]
                except:
                    print(f'Failed to read {pum_file}')
                    import pdb; pdb.set_trace()
                    continue
                
                if not os.path.basename(pum_file) in withheld_pumas:
                    # store building ids
                    for bldg_id in bldg_ids:
                        # train
                        if num_buildings >= bldgs_per_bty:
                            break
                        bldg_id = bldg_id.zfill(6)
                        # Sample the starting index between (0,24)
                        s_start = np.random.randint(0, 24)
                        for s_idx in range(s_start, train_hours - (args.context_len + args.pred_len), args.sliding_window_stride):                            
                            seq_ptr = str(args.context_len + s_idx).zfill(4)  # largest seq ptr is < 10000
                            # NB: We don't *need* \n at the end of each line, but it makes it easier to count # of lines for dataloading
                            linestr = f'{building_type_and_year}\t{census_region_idx}\t{os.path.basename(pum_file).split("=")[1]}\t{bldg_id}\t{seq_ptr}\n'
                            assert len(linestr) == 26, f'linestr: {linestr}'
                            train_idx_file.write(linestr)
                        # val
                        if 'amy2018' in by:
                            s_start += train_hours 
                            for s_idx in range(s_start, (train_hours + val_hours) - (args.context_len + args.pred_len), args.sliding_window_stride):
                                seq_ptr = str(args.context_len + s_idx).zfill(4)  # largest seq ptr is < 10000
                                assert len(linestr) == 26, f'linestr: {linestr}'
                                linestr = f'{building_type_and_year}\t{census_region_idx}\t{os.path.basename(pum_file).split("=")[1]}\t{bldg_id}\t{seq_ptr}\n'
                                val_idx_file.write(linestr)
                        num_buildings += 1
                # for each puma in census region
                if num_buildings >= bldgs_per_bty:
                    break
            # for each census region
            if num_buildings >= bldgs_per_bty:
                 break
    # Close files
    train_idx_file.close()
    val_idx_file.close()


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument('--num_buildings', type=int, default=1000, required=False,)
    args.add_argument('--seed', type=int, default=1, required=False,
                        help='Random seed for KMeans and shuffling. Default: 1')
    args.add_argument('--sliding_window_stride', type=int, default=24, required=False,
                        help='Stride for sliding window to split timeseries into training examples. Default: 24 hours')
    args.add_argument('--context_len', type=int, default=168, required=False,
                                help='Length of context sequence. For handling year beginning and year end. Default: 72 hours')
    args.add_argument('--pred_len', type=int, default=24, required=False,
                                help='Length of prediction sequence. For handling year beginning and year end. Default: 24 hours')


    args = args.parse_args()

    main(args)