import argparse
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import random
import glob 
import os
from buildings_bench.tokenizer import LoadQuantizer


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(os.environ.get('BUILDINGS_BENCH', ''), 'metadata')
    time_series_dir = Path(os.environ.get('BUILDINGS_BENCH', ''), 'Buildings-900K', 'end-use-load-profiles-for-us-building-stock', '2021')    
    building_years = ['comstock_tmy3_release_1', 'resstock_tmy3_release_1', 'comstock_amy2018_release_1', 'resstock_amy2018_release_1'] 
    pumas = ['by_puma_midwest', 'by_puma_south', 'by_puma_northeast', 'by_puma_west']

    all_buildings = []
    num_buildings = 0

    for by in building_years:
        by_path = time_series_dir / by / 'timeseries_individual_buildings'
        for pum in pumas:
            pum_path = by_path / pum / 'upgrade=0'
            pum_files = glob.glob(str(pum_path / 'puma=*'))
            # subsample pumas for faster quantization to 10 per census region
            random.shuffle(pum_files)
            pum_files = pum_files[:10]

            for pum_file in pum_files:
                # load the parquet file and convert each column to a numpy array
                df = pq.read_table(pum_file).to_pandas()
                # convert each column to a numpy array and stack vertically
                all_buildings += [np.vstack([df[col].to_numpy() for col in df.columns if col != 'timestamp'])]
                num_buildings += len(all_buildings[-1])

    print(f'Loaded {num_buildings} buildings for tokenization')

    lq = LoadQuantizer(args.seed, 
                       num_centroids=args.num_clusters,
                       with_merge=(not args.without_merge),
                       merge_threshold=args.merge_threshold,
                       device=args.device)
    lq.train(np.vstack(all_buildings))
    lq.save(output_dir)


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument('--num_clusters', type=int, default=512, required=False,
                        help='Number of clusters for KMeans. Default: 512')
    args.add_argument('--without_merge', action='store_true',
                        help='Do not merge clusters in KMeans. Default: False')
    args.add_argument('--merge_threshold', type=float, default=0.01, required=False,
                        help='Threshold for merging clusters during tokenization. Default: 0.01') 
    args.add_argument('--device', type=str, default='cuda:0', required=False,
                            help='Device to use. Default: cuda:0')
    args.add_argument('--seed', type=int, default=1, required=False,
                        help='Random seed for KMeans and shuffling. Default: 1')

    args = args.parse_args()

    main(args)
