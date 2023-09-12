import torch 
from pathlib import Path
import argparse 
import os
import tomli

from buildings_bench import load_torch_dataset, benchmark_registry
from buildings_bench import utils
from buildings_bench.tokenizer import LoadQuantizer
from buildings_bench.evaluation.managers import DatasetMetricsManager
from buildings_bench.models import model_factory
from buildings_bench.evaluation import scoring_rule_factory

SCRIPT_PATH = Path(os.path.realpath(__file__)).parent


@torch.no_grad()
def zero_shot_learning(args, model_args, results_path: Path):
    device = args.device

    model, _, predict = model_factory(args.config, model_args)
    model = model.to(device)

    transform_path = Path(os.environ.get('BUILDINGS_BENCH', '')) \
        / 'metadata' / 'transforms'

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
                                                          remove_outliers=args.remove_outliers,
                                                          weather=args.weather)
        
        num_of_buildings = len(buildings_datasets_generator)
        print(f'dataset {dataset_name}: {num_of_buildings} buildings')
        # For each building
        for count, (building_name, building_dataset) in enumerate(buildings_datasets_generator, start=1):
            print(f'dataset {dataset_name} {count}/{num_of_buildings} building-year {building_name} '
                    f'day-ahead forecasts {len(building_dataset)}')

            metrics_manager.add_building_to_dataset_if_missing(
                dataset_name, building_name,
            )

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
        
            # create a dataloader for the building
            building_dataloader = torch.utils.data.DataLoader(
                                    building_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False)
            for batch in building_dataloader:
                building_types_mask = batch['building_type'][:,0,0] == 1

                for k,v in batch.items():
                    batch[k] = v.to(device)

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
                    # invert for crps
                    targets = inverse_transform(targets)
                    if args.apply_scaler_transform == 'standard':
                        mu = inverse_transform(distribution_params[:,:,0])
                        sigma = load_transform.undo_transform_std(distribution_params[:,:,1])
                        distribution_params = torch.cat([mu.unsqueeze(-1), sigma.unsqueeze(-1)],-1)
                    
                    elif args.apply_scaler_transform == 'boxcox':
                        ######## backproject approximate Gaussian in unscaled space ########
                        mu = inverse_transform(distribution_params[:,:,0])
                        muplussigma = inverse_transform(torch.sum(distribution_params,-1))
                        sigma = muplussigma - mu
                        muminussigma = inverse_transform(distribution_params[:,:,0] - distribution_params[:,:,1])
                        sigma = (sigma + (mu - muminussigma)) / 2
                        distribution_params = torch.cat([mu.unsqueeze(-1), sigma.unsqueeze(-1)],-1)

                if not model.continuous_loads:
                    centroids = load_transform.kmeans.centroids.squeeze() \
                        if args.tokenizer_without_merge else load_transform.merged_centroids
                else:
                    centroids = None
                
                metrics_manager(
                    dataset_name,
                    building_name,
                    continuous_targets,
                    predictions,
                    building_types_mask,
                    y_categories=targets,
                    y_distribution_params=distribution_params,
                    centroids=centroids
                )
    print('Generating summaries...')
    variant_name = f':{args.variant_name}' if args.variant_name != '' else ''
    metrics_file = results_path / f'metrics_{args.config}{variant_name}.csv'
    scoring_rule_file = results_path / f'scoring_rule_{args.config}{variant_name}.csv'

    if not args.ignore_scoring_rules:
        metrics_df, scoring_rule_df = metrics_manager.summary()    
        if metrics_file.exists():    
            metrics_df.to_csv(metrics_file, mode='a', index=False, header=False)
        else:
            metrics_df.to_csv(metrics_file, index=False)
        if scoring_rule_file.exists():
            scoring_rule_df.to_csv(scoring_rule_file, mode='a', index=False, header=False)
        else:
            scoring_rule_df.to_csv(scoring_rule_file, index=False)
    else:
        metrics_df = metrics_manager.summary()    
        if metrics_file.exists():    
            metrics_df.to_csv(metrics_file, mode='a', index=False, header=False)
        else:
            metrics_df.to_csv(metrics_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--results_path', type=str, default='results/')
    parser.add_argument('--config', type=str, default='', required=True)
    parser.add_argument('--benchmark', nargs='+', type=str, default=['all'],
                        help='Which datasets in the benchmark to run. Default is ["all."] '
                             'See the dataset registry in buildings_bench.data.__init__.py for options.')
    parser.add_argument('--remove_outliers', action='store_true',
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
    parser.add_argument('--weather', action='store_true', 
                        help='Use weather data')
    
    args = parser.parse_args()
    utils.set_seed(args.seed)
    
    config_path = SCRIPT_PATH  / '..' / 'buildings_bench' / 'configs'
    if (config_path / f'{args.config}.toml').exists():
        toml_args = tomli.load(( config_path / f'{args.config}.toml').open('rb'))
        model_args = toml_args['model']
        if 'zero_shot' in toml_args:
            for k,v in toml_args['zero_shot'].items():
                setattr(args, k, v)
        if not model_args['continuous_loads'] or 'apply_scaler_transform' not in args:
            setattr(args, 'apply_scaler_transform', '')
    else:
        raise ValueError(f'Config {args.config}.toml not found.')

    results_path = Path(args.results_path)
    if args.remove_outliers:
        results_path = results_path / 'remove_outliers'

    results_path.mkdir(parents=True, exist_ok=True)

    zero_shot_learning(args, model_args, results_path)
