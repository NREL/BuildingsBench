import pandas as pd
from pathlib import Path 
from rliable import library as rly
import numpy as np
from buildings_bench import BuildingTypes


def return_aggregate_median(model_list, 
                            results_dir,
                            experiment='zero_shot',
                            metrics=['cvrmse'], 
                            exclude_simulated = True,
                            only_simulated = False,
                            oov_list = [],
                            reps=50000):
    """Compute the aggregate median for a list of models and metrics over all buildings.
    Also returns the stratified 95% boostrap CIs for the aggregate median.

    Args:
        model_list (list): List of models to compute aggregate median for.
        results_dir (str): Path to directory containing results.
        experiment (str, optional): Experiment type. Defaults to 'zero_shot'.
            Options: 'zero_shot', 'transfer_learning'.
        metrics (list, optional): List of metrics to compute aggregate median for. Defaults to ['cvrmse'].
        exclude_simulated (bool, optional): Whether to exclude simulated data. Defaults to True.
        only_simulated (bool, optional): Whether to only include simulated data. Defaults to False.
        oov_list (list, optional): List of OOV buildings to exclude. Defaults to [].
        reps (int, optional): Number of bootstrap replicates to use. Defaults to 50000.

    Returns:
        result_dict (Dict): Dictionary containing aggregate median and CIs for each metric and building type.
    """

    result_dict = {}        
    aggregate_func = lambda x : np.array([
        np.median(x.reshape(-1))])
    for building_type in [BuildingTypes.RESIDENTIAL, BuildingTypes.COMMERCIAL]:
        result_dict[building_type] = {}
        for metric in metrics:
            result_dict[building_type][metric] = {}

            if experiment == 'zero_shot' and (metric == 'rps' or metric == 'crps'):
                prefix = 'scoring_rule'
            elif experiment == 'transfer_learning' and (metric == 'rps' or metric == 'crps'):
                prefix = 'TL_scoring_rule'
            elif experiment == 'zero_shot':
                prefix = 'metrics'
            elif experiment == 'transfer_learning':
                prefix = 'TL_metrics'
    
            for model in model_list:
                df = pd.read_csv(Path(results_dir) / f'{prefix}_{model}.csv')

                if len(oov_list) > 0:
                    # Remove OOV buildings
                    df = df[~df['building_id'].str.contains('|'.join(oov_list))]
                
                if exclude_simulated:
                    # Exclude synthetic data
                    df = df[~( (df['dataset'] == 'buildings-900k-test') | (df['dataset'] == 'buildings-1m-test') )]
                elif only_simulated:
                    df = df[ (df['dataset'] == 'buildings-900k-test') | (df['dataset'] == 'buildings-1m-test') ]
                
                # if any df values are inf or nan
                if df.isnull().values.any() or np.isinf(df.value).values.any():
                    print(f'Warning: {model} has inf/nan values')
                # REmove inf/nan values
                df = df.replace(np.inf, np.nan)
                df = df.dropna() 

                if metric != 'rps' and metric != 'crps':    
                    result_dict[building_type][metric][model] = \
                        df[(df['metric'] == metric) & (df['building_type'] == building_type)]['value'].values.reshape(-1,1)
                else:
                    result_dict[building_type][metric][model] = \
                        df[df['building_type'] == building_type]['value'].values.reshape(-1,1)

            aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
                result_dict[building_type][metric], aggregate_func, reps=reps)
            result_dict[building_type][metric] = (aggregate_scores, aggregate_score_cis)
    return result_dict


def pretty_print_aggregates(results_dict) -> None:
    """Pretty print the aggregate results. 
    
    `model_name building_type metric_name: value (CI)`

    Args:
        result_dict (Dict): Dictionary of aggregate metrics for residential and commercial buildings.
    """
    print('model name\t building type\t metric name\t value (95% C.I.)')
    print('==========================================================')

    for building_type, v in results_dict.items():
        for metric_name, vv in v.items():
            agg_scores, agg_cis = vv
            for model_name, metric_value in agg_scores.items():
                cis = agg_cis[model_name]
                if metric_name in ['nrmse', 'nmae', 'nmbe', 'cvrmse']:
                    metric_value = metric_value[0]*100
                else:
                    metric_value = metric_value[0]
                print(f'{model_name} {building_type} {metric_name}: {metric_value:.3f} ({cis[0][0]:.3f},{cis[1][0]:.3f})')


if __name__ == '__main__':

    import os

    oov = []
    with open(Path(os.environ.get('BUILDINGS_BENCH', '')) / 'metadata' / 'oov.txt', 'r') as f:
        for l in f:
            oov += [l.strip().split(' ')[1]]

    results_dir = '/data/local/projects/foundation/NREL/BuildingsBench/results/remove_outliers'

    pretty_print_aggregates(return_aggregate_median(['AveragePersistence', 'TransformerWithGaussian-L'],
                             results_dir,
                             experiment='zero_shot',
                             oov_list = oov))
