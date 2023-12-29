import torch 
from pathlib import Path
import argparse 
import os
import tomli
import pandas as pd
import calendar

from buildings_bench import load_torch_dataset, benchmark_registry
from buildings_bench import utils
from buildings_bench.tokenizer import LoadQuantizer
from buildings_bench.data import g_weather_features
from buildings_bench.evaluation.managers import DatasetMetricsManager
from buildings_bench.evaluation import aggregate
from buildings_bench.models import model_factory
from buildings_bench.evaluation import scoring_rule_factory
import buildings_bench.transforms as transforms

SCRIPT_PATH = Path(os.path.realpath(__file__)).parent


@torch.no_grad()
def zero_shot_learning(args, model_args, results_path: Path):
    device = args.device

    if args.weather: 
        model_args['weather_features'] = g_weather_features[:1]

    model, _, predict = model_factory(args.model, model_args)
    model = model.to(device)

    transform_path = Path(os.environ.get('BUILDINGS_BENCH', '')) \
        / 'metadata' / 'transforms'
    weather_transform_path = transform_path / 'weather-900K'

    if not model.continuous_loads:   
        load_transform = LoadQuantizer(
            with_merge=(not args.tokenizer_without_merge),
              num_centroids=model.vocab_size,
              device='cuda:0' if 'cuda' in device else 'cpu')
        load_transform.load(transform_path)

    # Load from ckpts
    if args.checkpoint != '':
        model.load_from_checkpoint(args.checkpoint)
    model.eval()

    if args.benchmark[0] == 'all':
        args.benchmark = benchmark_registry
    elif args.benchmark[0] == 'real':
        y = [x for x in benchmark_registry if x != 'buildings-900k-test']
        args.benchmark = y
        
    if args.ignore_scoring_rules:
        metrics_manager = DatasetMetricsManager()
    elif model.continuous_loads:
        metrics_manager = DatasetMetricsManager(scoring_rule = scoring_rule_factory('crps'))
    else:
        metrics_manager = DatasetMetricsManager(scoring_rule = scoring_rule_factory('rps'))

    print(f'Evaluating model on test datasets {args.benchmark}...')

    # Iterate over each dataset in the benchmark
    for dataset_name in args.benchmark:
        # Load the dataset generator
        buildings_datasets_generator = load_torch_dataset(dataset_name,
                                                          apply_scaler_transform=args.apply_scaler_transform,
                                                          scaler_transform_path=transform_path,
                                                          include_outliers=args.include_outliers,
                                                          weather=args.weather)
        
        num_of_buildings = len(buildings_datasets_generator)
        print(f'dataset {dataset_name}: {num_of_buildings} buildings')
        # For each building
        for count, (building_name, building_dataset) in enumerate(buildings_datasets_generator, start=1):
            print(f'dataset {dataset_name} {count}/{num_of_buildings} building-year {building_name} '
                    f'day-ahead forecasts {len(building_dataset)}')

            # Create a dataframe to store the results
            building_df = pd.DataFrame(columns=['timestamp', 'load_historical', 'temperature_historical', 'load_forecasted'])

            if not model.continuous_loads: # Quantized loads
                 transform = load_transform.transform
                 inverse_transform = load_transform.undo_transform
            elif args.apply_scaler_transform != '': # Scaling continuous values
                 transform = lambda x: x 

                 if isinstance(building_dataset, torch.utils.data.ConcatDataset):
                     load_transform = building_dataset.datasets[0].load_transform
                     inverse_transform = load_transform.undo_transform
                 else:
                     load_transform = building_dataset.load_transform
                     inverse_transform = load_transform.undo_transform
            else: # Continuous unscaled values
                 transform = lambda x: x
                 inverse_transform = lambda x: x

            weather_transform = transforms.StandardScalerTransform()
            weather_transform.load(weather_transform_path / 'temperature')

            # create a dataloader for the building
            building_dataloader = torch.utils.data.DataLoader(
                                    building_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False)
            for batch in building_dataloader:

                for k,v in batch.items():
                    batch[k] = v.to(device)

                year_arr = batch['year'].clone() # get year of each sample, knowing that it is the same for all samples in the batch
                del batch['year']

                continuous_load = batch['load'].clone()
                continuous_targets = continuous_load[:, model.context_len:]

                # Transform if needed
                batch['load'] = transform(batch['load'])
                # These could be tokens or continuous
                targets = batch['load'][:, model.context_len:]

                if args.device == 'cuda':
                    with torch.cuda.amp.autocast():
                        predictions, distribution_params = predict(batch)
                else:
                    predictions, distribution_params = predict(batch)
                    
                predictions = inverse_transform(predictions)

                if args.apply_scaler_transform != '':
                    continuous_targets = inverse_transform(continuous_targets)

                # Note: there can be multiple years in one batch 
                    
                for i in range(year_arr.shape[0]):
                    cur_year = year_arr[i].item()
                    time_transform = transforms.TimestampTransform(is_leap_year=calendar.isleap(cur_year))

                    # undo time transform
                    timestamp = time_transform.undo_transform(torch.cat((batch['day_of_year'][i, model.context_len:], 
                                                                            batch['day_of_week'][i, model.context_len:], 
                                                                            batch['hour_of_day'][i, model.context_len:]), dim=1).cpu())
                    timestamp = utils.time_features_to_datetime(timestamp, cur_year)

                    temperature = batch['temperature'][i, model.context_len:]

                    df = pd.DataFrame({
                        'timestamp': timestamp, 
                        'load_historical': continuous_targets[i].view(-1).cpu(), 
                        'temperature_historical': weather_transform.undo_transform(temperature.cpu()).view(-1),
                        'load_forecasted': predictions[i].view(-1).cpu()
                    })

                    building_df = pd.concat([building_df, df], ignore_index=True)
            
            building_df.to_csv(results_path / f'{dataset_name}_{building_name.replace("/", "-")}_forecasted_load.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--results_path', type=str, default='results/')
    parser.add_argument('--model', type=str, default='', required=True,
                        help='Name of your model. Should match the config'
                             ' filename without .toml extension.'
                             ' Example: "TransformerWithTokenizer-S"')
    parser.add_argument('--benchmark', nargs='+', type=str, default=['all'],
                        help='Which datasets in the benchmark to run. Default is ["all."] '
                             'See the dataset registry in buildings_bench.data.__init__.py for options.')
    parser.add_argument('--include_outliers', action='store_true',
                        help='Eval with a filtered variant with certain outliers removed')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=360)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--ignore_scoring_rules', action='store_true', 
                        help='Do not compute a scoring rule for this model.')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Path to a checkpoint to load. Optional. '
                        ' One can also load a checkpoint from Wandb by specifying the run_id.')
    parser.add_argument('--variant_name', type=str, default='',
                        help='Name of the variant. Optional. Used for results files.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use. Default is cuda if available else cpu.')
    parser.add_argument('--tokenizer_without_merge', action='store_true', default=False, 
                        help='Use the tokenizer without merge. Default is False.')
    parser.add_argument('--apply_scaler_transform', type=str, default='',
                        choices=['', 'standard', 'boxcox'], 
                        help='Apply a scaler transform to the load values.')
    parser.add_argument('--use-weather', dest='weather', action='store_true', 
                        help='Use weather data.')
    
    args = parser.parse_args()
    utils.set_seed(args.seed)
    
    config_path = SCRIPT_PATH  / '..' / 'buildings_bench' / 'configs'
    if (config_path / f'{args.model}.toml').exists():
        toml_args = tomli.load(( config_path / f'{args.model}.toml').open('rb'))
        model_args = toml_args['model']
        if 'zero_shot' in toml_args:
            for k,v in toml_args['zero_shot'].items():
                if k != 'weather':
                    setattr(args, k, v)
                elif v != 'False':
                    setattr(args, k, True)
        if not model_args['continuous_loads'] or 'apply_scaler_transform' not in args:
            setattr(args, 'apply_scaler_transform', '')
    else:
        raise ValueError(f'Config {args.model}.toml not found.')

    results_path = Path(args.results_path)
    if args.include_outliers:
        results_path = results_path / 'buildingsbench_with_outliers'

    results_path.mkdir(parents=True, exist_ok=True)

    zero_shot_learning(args, model_args, results_path)
