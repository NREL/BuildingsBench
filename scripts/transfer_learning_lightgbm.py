from pathlib import Path
import os
import argparse 
from buildings_bench import utils
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg

from buildings_bench import BuildingTypes
from buildings_bench import load_pandas_dataset, benchmark_registry
from buildings_bench.data.datasets import keep_buildings
from buildings_bench import utils
from buildings_bench.evaluation.managers import DatasetMetricsManager


SCRIPT_PATH = Path(os.path.realpath(__file__)).parent


def transfer_learning(args, results_path: Path):
    global benchmark_registry
    lag = 168

    # remove synthetic
    benchmark_registry = [b for b in benchmark_registry if b != 'buildings-900k-test']
    if args.benchmark[0] == 'all':
        args.benchmark = benchmark_registry

    metrics_manager = DatasetMetricsManager()

    target_buildings = []
    if not args.dont_subsample_buildings:
        metadata_dir = Path(os.environ.get('BUILDINGS_BENCH', ''), 'metadata')
        with open(metadata_dir / 'transfer_learning_commercial_buildings.txt', 'r') as f:
            target_buildings += f.read().splitlines()
        with open(metadata_dir / 'transfer_learning_residential_buildings.txt', 'r') as f:
            target_buildings += f.read().splitlines()

    for dataset_name in args.benchmark:
        dataset_generator = load_pandas_dataset(dataset_name, 
                                                feature_set='engineered',
                                                include_outliers=args.include_outliers,
                                                weather_inputs=['temperature'] if args.use_temperature_input else None)
        # Filter to target buildings
        if len(target_buildings) > 0:
            dataset_generator = keep_buildings(dataset_generator, target_buildings)

        # For metrics management
        if dataset_generator.building_type == BuildingTypes.COMMERCIAL:
            building_types_mask = (BuildingTypes.COMMERCIAL_INT * torch.ones([1,24,1])).bool()
        else:
            building_types_mask = (BuildingTypes.RESIDENTIAL_INT * torch.ones([1,24,1])).bool()
                
        for building_name, bldg_df in dataset_generator:

            # if date range is less than 120 days, skip - 90 days training, 30+ days eval.
            if len(bldg_df) < (args.num_training_days+30)*24:
                print(f'{dataset_name} {building_name} has too few days {len(bldg_df)}')
                continue

            print(f'dataset {dataset_name} building {building_name}')
            
            metrics_manager.add_building_to_dataset_if_missing(
                 dataset_name, f'{building_name}',
            )

            # Split into fine-tuning and evaluation set by date
            # Get the first month of data from bldg_df by index
            start_timestamp = bldg_df.index[0]
            end_timestamp = start_timestamp + pd.Timedelta(days=args.num_training_days)
            historical_date_range = pd.date_range(start=start_timestamp, end=end_timestamp, freq='H')

            training_set = bldg_df.loc[historical_date_range]
            
            test_set = bldg_df.loc[~bldg_df.index.isin(historical_date_range)]
            test_start_timestamp = test_set.index[0]
            test_end_timestamp = test_start_timestamp + pd.Timedelta(days=180)
            #test_date_range = pd.date_range(start=test_start_timestamp, end=test_end_timestamp, freq='H')
            #test_set = test_set.loc[test_date_range]
            test_set = test_set[test_set.index <= test_end_timestamp]

            print(f'fine-tune set date range: {training_set.index[0]} {training_set.index[-1]}, '
                  f'test set date range: {test_set.index[0]} {test_set.index[-1]}')

            # train the model
            forecaster = ForecasterAutoreg(
                    regressor        = LGBMRegressor(max_depth=-1, n_estimators=100, n_jobs=24, verbose=-1),
                    lags             = lag
                )
            forecaster.fit(
                y               = training_set['power'],
                exog            = training_set[[key for key in training_set.keys() if key != 'power']]
            )
        
            pred_days = (len(test_set) - lag - 24) // 24
            for i in range(pred_days):
                
                seq_ptr =lag + 24 * i

                last_window  = test_set.iloc[seq_ptr - lag : seq_ptr]
                ground_truth = test_set.iloc[seq_ptr : seq_ptr + 24]

                predictions = forecaster.predict(
                    steps       = 24,
                    last_window = last_window['power'],
                    exog        = ground_truth[[key for key in test_set.keys() if key != 'power']]
                )

                metrics_manager(
                    dataset_name,
                    f'{building_name}',
                    torch.from_numpy(ground_truth['power'].values).float().view(1,24,1),
                    torch.from_numpy(predictions.values).float().view(1,24,1),
                    building_types_mask
                )

    print('Generating summaries...')
    variant_name = f':{args.variant_name}' if args.variant_name != '' else ''
    metrics_file = results_path / f'TL_metrics_lightgbm{variant_name}.csv'

    metrics_df = metrics_manager.summary()    
    if metrics_file.exists():    
        metrics_df.to_csv(metrics_file, mode='a', index=False, header=False)
    else:
        metrics_df.to_csv(metrics_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--results_path', type=str, default='results/')
    parser.add_argument('--benchmark', nargs='+', type=str, default=['all'],
                        help='Which datasets in the benchmark to run. Default is ["all."] '
                             'See the dataset registry in buildings_bench.data.__init__.py for options.')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--variant_name', type=str, default='',
                        help='Name of the variant. Optional. Used for results files.')
    parser.add_argument('--include_outliers', action='store_true')

    # Transfer learning - data
    parser.add_argument('--num_training_days', type=int, default=180,
                        help='Number of days for fine-tuning (last 30 used for early stopping)')
    parser.add_argument('--dont_subsample_buildings', action='store_true', default=False,
                        help='Evaluate on all instead of a subsample of 100 res/100 com buildings')      
    parser.add_argument('--use_temperature_input', action='store_true',
                        help='Include temperature as an additional feature in the model')

    args = parser.parse_args()
    utils.set_seed(args.seed)
    

    results_path = Path(args.results_path)
    if args.include_outliers:
        results_path = results_path / 'buildingsbench_with_outliers'
    results_path.mkdir(parents=True, exist_ok=True)

    transfer_learning(args, results_path)
