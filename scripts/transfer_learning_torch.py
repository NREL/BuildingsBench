from pathlib import Path
import os
import argparse 
from buildings_bench import utils
import pandas as pd
import torch
import tomli 
import numpy as np
from copy import deepcopy

from buildings_bench import load_pandas_dataset, benchmark_registry
from buildings_bench.data.datasets import PandasTransformerDataset
from buildings_bench.data import g_weather_features
from buildings_bench.data.datasets import keep_buildings
from buildings_bench import utils
from buildings_bench.tokenizer import LoadQuantizer
from buildings_bench.evaluation.managers import DatasetMetricsManager
from buildings_bench.evaluation import aggregate
from buildings_bench.models import model_factory
from buildings_bench.evaluation import scoring_rule_factory


SCRIPT_PATH = Path(os.path.realpath(__file__)).parent


def train(df_tr, df_val, args, model, transform, loss, lr, device):
    torch_train_set = PandasTransformerDataset(df_tr, sliding_window=24, weather=args.weather)
    
    train_dataloader = torch.utils.data.DataLoader(
                                torch_train_set,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers, 
                                shuffle=True)
    
    torch_val_set = PandasTransformerDataset(df_val, sliding_window=24, weather=args.weather)
    val_dataloader = torch.utils.data.DataLoader(
                                torch_val_set,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                shuffle=False)
    # Unfreeze the layer being fine-tuned
    # and pass to the optimizer
    params = model.unfreeze_and_get_parameters_for_finetuning()
    if params is None:
        raise ValueError('No parameters provided for fine-tuning. Did you mean to run with --eval_zero_shot?')

    optimizer = torch.optim.AdamW(params, lr=lr) 

    model.train()
    best_val = 100000
    patience_counter = 0

    # Fine tune the loaded model with frozen weights
    for epoch in range(args.max_epochs):
        losses = []
        print(f'epoch {epoch}')
        for batch in train_dataloader:
            optimizer.zero_grad()

            for k,v in batch.items():
                batch[k] = v.to(device)

            # Apply transform to load if needed
            batch['load'] = transform(batch['load'])
                                
            with torch.cuda.amp.autocast():
                preds = model(batch)
                targets = batch['load'][:, model.context_len:]      
                batch_loss = loss(preds, targets)

            losses.append(batch_loss.item())
            batch_loss.backward()

            optimizer.step()

        # run validation
        losses_val = []
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                for k,v in batch.items():
                    batch[k] = v.to(device)

                # Apply transform to load if needed
                batch['load'] = transform(batch['load'])
                                    
                with torch.cuda.amp.autocast():
                    preds = model(batch)
                    targets = batch['load'][:, model.context_len:]
                    batch_loss = loss(preds, targets)

                losses_val.append(batch_loss.item())
        
        epoch_val_loss = np.mean(losses_val)
        
        model.train()
        print('epoch train loss = ', np.mean(losses))
        print('epoch val loss = ', np.mean(losses_val))
        
        if epoch_val_loss > best_val:
            patience_counter += 1
        else:
            patience_counter = 0
            best_val = epoch_val_loss
        
        if patience_counter > args.patience:
            print(f'early stopping after epoch {epoch} with patience={args.patience}, best val loss {best_val}')
            break
            
    return model


def transfer_learning(args, model_args, results_path: Path):
    global benchmark_registry
    device = args.device

    if args.weather:
        model_args['weather_features'] = g_weather_features[:1]

    # load and configure the model for transfer learning
    model, loss, _ = model_factory(args.model, model_args)
    model = model.to(args.device)
    transform_path = Path(os.environ.get('BUILDINGS_BENCH', '')) / 'metadata' / 'transforms'


    if not model.continuous_loads:
        load_transform = LoadQuantizer(
            with_merge=(not args.tokenizer_without_merge),
            num_centroids=model.vocab_size,
            device='cuda:0' if 'cuda' in device else 'cpu')
        load_transform.load(transform_path)


    if args.checkpoint != '':
        # By default, fine tune all layers
        model.load_from_checkpoint(args.checkpoint)
    model.train()

    # remove synthetic
    benchmark_registry = [b for b in benchmark_registry if b != 'buildings-900k-test']
    if args.benchmark[0] == 'all':
        args.benchmark = benchmark_registry

    if args.ignore_scoring_rules:
        metrics_manager = DatasetMetricsManager()
    elif model.continuous_loads:
        metrics_manager = DatasetMetricsManager(scoring_rule = scoring_rule_factory('crps'))
    else:
        metrics_manager = DatasetMetricsManager(scoring_rule = scoring_rule_factory('rps'))

    target_buildings = []
    if not args.dont_subsample_buildings:
        metadata_dir = Path(os.environ.get('BUILDINGS_BENCH', ''), 'metadata')
        if len(args.hyper_opt) > 0:
            with open(metadata_dir / 'transfer_learning_hyperparameter_tuning.txt', 'r') as f:
                target_buildings += f.read().splitlines()
        else:
            with open(metadata_dir / 'transfer_learning_commercial_buildings.txt', 'r') as f:
                target_buildings += f.read().splitlines()
            with open(metadata_dir / 'transfer_learning_residential_buildings.txt', 'r') as f:
                target_buildings += f.read().splitlines()

    for dataset in args.benchmark:
        dataset_generator = load_pandas_dataset(dataset,
                                                feature_set='transformer',
                                                apply_scaler_transform=args.apply_scaler_transform,
                                                scaler_transform_path=transform_path,
                                                include_outliers=args.include_outliers,
                                                weather=args.weather)
        # Filter to target buildings
        if len(target_buildings) > 0:
            dataset_generator = keep_buildings(dataset_generator, target_buildings)

        # Transforms
        if not model.continuous_loads: 
            transform = load_transform.transform
            inverse_transform = load_transform.undo_transform
        elif args.apply_scaler_transform != '':
            transform = lambda x: x
            load_transform = dataset_generator.load_transform
            inverse_transform = load_transform.undo_transform
        else: # Continuous unscaled values
            transform = lambda x: x
            inverse_transform = lambda x: x
        
        num_of_buildings = len(dataset_generator)
        print(f'dataset {dataset}: {num_of_buildings} buildings')
        for count, (building_name, bldg_df) in enumerate(dataset_generator, start=1):
            # if date range is less than 120 days, skip - 90 days training, 30+ days eval.
            if len(bldg_df) < (args.num_training_days+30)*24:
                print(f'{dataset} {building_name} has too few days {len(bldg_df)}')
                continue

            print(f'dataset {dataset} building {building_name} {count}/{num_of_buildings}')
            
            metrics_manager.add_building_to_dataset_if_missing(
                 dataset, building_name,
            )

            # Split into fine-tuning and evaluation set by date
            # Get the first month of data from bldg_df by index
            start_timestamp = bldg_df.index[0]
            end_timestamp = start_timestamp + pd.Timedelta(days=args.num_training_days)
            historical_date_range = pd.date_range(start=start_timestamp, end=end_timestamp, freq='H')

            training_set = bldg_df.loc[historical_date_range]
            training_start_timestamp = training_set.index[0]
            training_end_timestamp = training_start_timestamp + pd.Timedelta(days=args.num_training_days-30)

            train_date_range = pd.date_range(start=training_start_timestamp, end=training_end_timestamp, freq='H')
            training_set_ = training_set.loc[train_date_range]
            validation_set = training_set[~training_set.index.isin(train_date_range)]

            test_set = bldg_df.loc[~bldg_df.index.isin(historical_date_range)]
            test_start_timestamp = test_set.index[0]
            test_end_timestamp = test_start_timestamp + pd.Timedelta(days=180) # test on subsequent <= 180 days
            test_set = test_set[test_set.index <= test_end_timestamp]

            print(f'fine-tune set date range: {training_set_.index[0]} {training_set_.index[-1]}, '
                  f'test set date range: {test_set.index[0]} {test_set.index[-1]}')
 
            if not args.eval_zero_shot:
                # train the model
                # we use deepcopy to avoid modifying the original model
                tuned_model = train(training_set_, validation_set, args, deepcopy(model),
                                    transform, loss, args.lr, args.device)
            else:
                tuned_model = deepcopy(model)

            # do the evaluation on the building test data
            torch_test_set = PandasTransformerDataset(test_set,
                                                      sliding_window=24, weather=args.weather)
            test_dataloader = torch.utils.data.DataLoader(
                                        torch_test_set,
                                        batch_size=360,
                                        num_workers=args.num_workers,
                                        shuffle=False)

            tuned_model.eval()
            with torch.no_grad():
                for batch in test_dataloader:
                    building_types_mask = batch['building_type'][:,0,0] == 1

                    for k,v in batch.items():
                        batch[k] = v.to(args.device)

                    continuous_load = batch['load'].clone()
                    continuous_targets = continuous_load[:, tuned_model.context_len:]

                    # Transform if needed
                    batch['load'] = transform(batch['load'])
                    targets = batch['load'][:, tuned_model.context_len:]

                    if args.device == 'cuda':
                        with torch.cuda.amp.autocast():
                            predictions, distribution_params = tuned_model.predict(batch)
                    else:
                        predictions, distribution_params = tuned_model.predict(batch)

                    predictions = inverse_transform(predictions)

                    if args.apply_scaler_transform != '':
                        continuous_targets = inverse_transform(continuous_targets)
                        targets = inverse_transform(targets)
                        if not args.ignore_scoring_rules and args.apply_scaler_transform == 'standard':
                            mu = inverse_transform(distribution_params[:,:,0])
                            sigma = load_transform.undo_transform_std(distribution_params[:,:,1])
                            distribution_params = torch.cat([mu.unsqueeze(-1), sigma.unsqueeze(-1)],-1)
                        
                        elif not args.ignore_scoring_rules and args.apply_scaler_transform == 'boxcox':
                            ######## approximate Gaussian in unscaled space ########
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
                        dataset,
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
    metrics_file = results_path / f'TL_metrics_{args.model}{variant_name}.csv'
    scoring_rule_file = results_path / f'TL_scoring_rule_{args.model}{variant_name}.csv'

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

    if len(args.benchmark) == len(benchmark_registry):
        # Compute and display aggregate statistics
        oov_bldgs = []
        with open(Path(os.environ.get('BUILDINGS_BENCH', '')) / 'metadata' / 'oov.txt', 'r') as f:
            for line in f:
                oov_bldgs += [line.strip().split(' ')[1]]

            metric_names = [m.name for m in metrics_manager.metrics_list]
            if metrics_manager.scoring_rule:
                metric_names += [metrics_manager.scoring_rule.name]

            # Returns a dictionary with the median of the nrmse (cv-rmse)
            # and crps metrics for the model with boostrapped 95% confidence intervals
            print('BuildingsBench (real)')
            results_dict = aggregate.return_aggregate_median(
                                model_list = [f'{args.model}{variant_name}'],
                                results_dir = str(results_path),
                                experiment = 'transfer_learning',
                                metrics = metric_names,
                                oov_list = oov_bldgs
                            )
            aggregate.pretty_print_aggregates(results_dict)



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
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--ignore_scoring_rules', action='store_true', help='Do not compute a scoring rule')
    parser.add_argument('--variant_name', type=str, default='',
                        help='Name of the variant. Optional. Used for results files.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--tokenizer_without_merge', action='store_true', default=False, 
                        help='Use the tokenizer without merge. Default is False.')
    parser.add_argument('--apply_scaler_transform', type=str, default='',
                        choices=['', 'standard', 'boxcox'], 
                        help='Apply a scaler transform to the load values.')
    parser.add_argument('--include_outliers', action='store_true')
    parser.add_argument('--use-weather', dest='weather', action='store_true', 
                        help='Use weather data')
    parser.add_argument('--hyper_opt', nargs='*', default=[],
                        help='Tells this script to not override the argparse values for'
                             ' these hyperparams with values in the config file.'
                             ' Expects the hyperparameter value to be set via argparse '
                             ' from the CLI. Example: --hyper_opt batch_size lr')
    
    # Transfer Learning - model
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Path to a checkpoint to load.')
    parser.add_argument('--max_epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=1e-6) # 1e-4, 1e-5, 1e-6
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--patience', type=int, default=2)

    # Transfer learning - data
    parser.add_argument('--num_training_days', type=int, default=180,
                        help='Number of days for fine-tuning (last 30 used for early stopping)')
    parser.add_argument('--dont_subsample_buildings', action='store_true', default=False,
                        help='Evaluate on all instead of a subsample of 100 res/100 com buildings')    
    parser.add_argument('--eval_zero_shot', action='store_true', default=False,
                        help='Evaluate on the test data without fine-tuning, '
                             'useful for getting baseline perf')

    args = parser.parse_args()
    utils.set_seed(args.seed)
    
    config_path = SCRIPT_PATH  / '..' / 'buildings_bench' / 'configs'
    if (config_path / f'{args.model}.toml').exists():
            toml_args = tomli.load(( config_path / f'{args.model}.toml').open('rb'))
            model_args = toml_args['model']
            if 'transfer_learning' in toml_args:
                for k,v in toml_args['transfer_learning'].items():
                    if not k in args.hyper_opt:
                        if hasattr(args, k):
                            print(f'Overriding argparse default for {k} with {v}')
                        setattr(args, k, v)
            if not model_args['continuous_loads']:
                setattr(args, 'apply_scaler_transform', '')
    else:
        raise ValueError(f'Config {args.model}.toml not found.')

   
    results_path = Path(args.results_path)
    if args.include_outliers:
        results_path = results_path / 'buildingsbench_with_outliers'

    results_path.mkdir(parents=True, exist_ok=True)

    transfer_learning(args, model_args, results_path)
