from pathlib import Path
import torch
import tomli
import os
from buildings_bench.data.buildings900K import Buildings900K
from buildings_bench.data.datasets import TorchBuildingDatasetsFromCSV
from buildings_bench.data.datasets import TorchBuildingDatasetFromParquet
from buildings_bench.data.datasets import PandasBuildingDatasetsFromCSV
from buildings_bench import BuildingTypes
from buildings_bench import transforms
from typing import List, Union


dataset_registry = [
    'buildings-900k-train',
    'buildings-900k-val',
    'buildings-900k-test',
    'sceaux',
    'borealis',
    'ideal',
    'bdg-2:panther',
    'bdg-2:fox',
    'bdg-2:rat',
    'bdg-2:bear',
    'electricity',
    'smart',
    'lcl'
]

benchmark_registry = [
    'buildings-900k-test',
    'sceaux',
    'borealis',
    'ideal',
    'bdg-2:panther',
    'bdg-2:fox',
    'bdg-2:rat',
    'bdg-2:bear',
    'electricity',
    'smart',
    'lcl'
]

g_weather_features = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'global_horizontal_radiation', 
                              'direct_normal_radiation', 'diffuse_horizontal_radiation']

def parse_building_years_metadata(datapath: Path, dataset_name: str):
    with open(datapath / 'metadata' / 'building_years.txt', 'r') as f:
        building_years = f.readlines()
    building_years = [building_year.strip() for building_year in building_years]
    building_years = filter(lambda building_year: dataset_name in building_year.lower(), building_years)
    
    return list(building_years)


def load_pretraining(
        name: str,
        num_buildings_ablation: int = -1,
        apply_scaler_transform: str = '',
        scaler_transform_path: Path = None,
        weather: List[str] = None,
        custom_idx_filename: str = '',
        context_len=168, # week
        pred_len=24,
        use_text_embedding=False) -> torch.utils.data.Dataset:
    r"""
    Pre-training datasets: buildings-900k-train, buildings-900k-val

    Args:
        name (str): Name of the dataset to load.
        num_buildings_ablation (int): Number of buildings to use for pre-training.
                                        If -1, use all buildings.
        apply_scaler_transform (str): If not using quantized load or unscaled loads,
                                 applies a {boxcox,standard} scaling transform to the load. Default: ''.
        scaler_transform_path (Path): Path to data for transform, e.g., pickled data for BoxCox transform.
        weather (List[str]): list of weather features to use. Default: None.
        custom_idx_filename (str): customized index filename. Default: ''
        context_len (int): Length of the context. Defaults to 168.
        pred_len (int): Length of the prediction horizon. Defaults to 24.
    
    Returns:
        torch.utils.data.Dataset: Dataset for pretraining.
    """
    dataset_path = Path(os.environ.get('BUILDINGS_BENCH', ''))
    if not dataset_path.exists():
        raise ValueError('BUILDINGS_BENCH environment variable not set')

    if num_buildings_ablation > 0:
        idx_file_suffix = f'_{num_buildings_ablation}'
    else:
        idx_file_suffix = ''
    if name.lower() == 'buildings-900k-train':
        idx_file = f'train_weekly{idx_file_suffix}.idx' if custom_idx_filename == '' else custom_idx_filename
        dataset = Buildings900K(dataset_path,
                               idx_file,
                               context_len=context_len,
                               pred_len=pred_len,
                               apply_scaler_transform=apply_scaler_transform,
                               scaler_transform_path = scaler_transform_path,
                               weather=weather)
    elif name.lower() == 'buildings-900k-val':
        idx_file = f'val_weekly{idx_file_suffix}.idx' if custom_idx_filename == '' else custom_idx_filename
        dataset = Buildings900K(dataset_path,
                               idx_file,
                               context_len=context_len,
                               pred_len=pred_len,
                               apply_scaler_transform=apply_scaler_transform,
                               scaler_transform_path = scaler_transform_path,
                               weather=weather)
    elif name.lower() == 'simcap-10k-train':
        idx_file = f'train_simcap.idx' if custom_idx_filename == '' else custom_idx_filename
        dataset = Buildings900K(dataset_path,
                               idx_file,
                               context_len=context_len,
                               pred_len=pred_len,
                               apply_scaler_transform=apply_scaler_transform,
                               scaler_transform_path = scaler_transform_path,
                               weather=weather,
                               use_text_embedding=use_text_embedding)
    elif name.lower() == 'simcap-10k-val':
        idx_file = f'val_simcap.idx' if custom_idx_filename == '' else custom_idx_filename
        dataset = Buildings900K(dataset_path,
                               idx_file,
                               context_len=context_len,
                               pred_len=pred_len,
                               apply_scaler_transform=apply_scaler_transform,
                               scaler_transform_path = scaler_transform_path,
                               weather=weather,
                               use_text_embedding=use_text_embedding)
    elif name.lower() == 'simcap-10k-test':
        idx_file = f'test_simcap.idx' if custom_idx_filename == '' else custom_idx_filename
        dataset = Buildings900K(dataset_path,
                               idx_file,
                               context_len=context_len,
                               pred_len=pred_len,
                               apply_scaler_transform=apply_scaler_transform,
                               scaler_transform_path = scaler_transform_path,
                               weather=weather,
                               use_text_embedding=use_text_embedding)
    return dataset
        
    
def load_torch_dataset(
        name: str,
        dataset_path: Path = None,
        apply_scaler_transform: str = '',
        scaler_transform_path: Path = None,
        weather: bool = False,
        remove_outliers: bool = False,
        context_len = 168,
        pred_len = 24
        ) -> Union[TorchBuildingDatasetsFromCSV, TorchBuildingDatasetFromParquet]:
    r"""Load datasets by name.

    Args:
        name (str): Name of the dataset to load.
        dataset_path (Path): Path to the benchmark data. Optional.
        apply_scaler_transform (str): If not using quantized load or unscaled loads,
                                 applies a {boxcox,standard} scaling transform to the load. Default: ''.
        scaler_transform_path (Path): Path to data for transform, e.g., pickled data for BoxCox transform.
        weather (bool): load weather data. Default: False
        remove_outliers (bool): Use version of BuildingsBench with some outliers removed.
        context_len (int): Length of the context. Defaults to 168.
        pred_len (int): Length of the prediction horizon. Defaults to 24.
    
    Returns:
        dataset (Union[TorchBuildingDatasetsFromCSV, TorchBuildingDatasetFromParquet]): Dataset for benchmarking.
    """
    if not dataset_path:
        dataset_path = Path(os.environ.get('BUILDINGS_BENCH', ''))
        if not dataset_path.exists():
            raise ValueError('BUILDINGS_BENCH environment variable not set')
        
    with open(dataset_path / 'metadata' / 'benchmark.toml', 'rb') as f:
        metadata = tomli.load(f)['buildings_bench']

    if name.lower() == 'buildings-900k-test':
        spatial_lookup = transforms.LatLonTransform()
        puma_files = list((dataset_path / 'Buildings-900K-test' / '2021').glob('*2018*/*/*/*/*/*.parquet'))
        if len(puma_files) == 0:
            raise ValueError(f'Could not find any Parquet files in '
                             f' {str(dataset_path / "Buildings-900K-test" / "2021")}')
        # to string
        puma_files = [str(Path(pf).parent) for pf in puma_files]
        puma_ids = [pf.split('puma=')[1] for pf in puma_files]
        building_types = []
        for pf in puma_files:
            if 'res' in pf:
                building_types += [BuildingTypes.RESIDENTIAL]
            elif 'com' in pf:
                building_types += [BuildingTypes.COMMERCIAL]
        dataset_generator = TorchBuildingDatasetFromParquet(
                                                         dataset_path,
                                                         puma_files,
                                                         [spatial_lookup.undo_transform( # pass unnormalized lat lon coords
                                                            spatial_lookup.transform(pid)) for pid in puma_ids],
                                                         building_types,
                                                         weather=weather,
                                                         context_len=context_len,
                                                         pred_len=pred_len,
                                                         apply_scaler_transform=apply_scaler_transform,
                                                         scaler_transform_path = scaler_transform_path,
                                                         leap_years=metadata['leap_years'])
    elif ':' in name.lower():
        name, subset = name.lower().split(':')
        dataset_metadata = metadata[name.lower()]
        all_by_files = parse_building_years_metadata(dataset_path, name.lower())
        all_by_files = filter(lambda by_file: subset in by_file.lower(), all_by_files)
        if remove_outliers:
            dataset_path = dataset_path / 'remove_outliers'
        print(f'loading data from {dataset_path}')
        dataset_generator = TorchBuildingDatasetsFromCSV(dataset_path,
                                                         all_by_files,
                                                         dataset_metadata[subset]['latlon'],
                                                         dataset_metadata[subset]['building_type'],
                                                         context_len=context_len,
                                                         pred_len=pred_len,
                                                         apply_scaler_transform=apply_scaler_transform,
                                                         scaler_transform_path = scaler_transform_path,
                                                         leap_years=metadata['leap_years']) 
    elif name.lower() in benchmark_registry:
        dataset_metadata = metadata[name.lower()]
        all_by_files = parse_building_years_metadata(dataset_path, name.lower())
        if remove_outliers:
            dataset_path = dataset_path / 'remove_outliers'
        print(f'loading data from {dataset_path}')
        dataset_generator = TorchBuildingDatasetsFromCSV(dataset_path,
                                                         all_by_files,
                                                         dataset_metadata['latlon'],
                                                         dataset_metadata['building_type'],
                                                         context_len=context_len,
                                                         pred_len=pred_len,
                                                         apply_scaler_transform=apply_scaler_transform,
                                                         scaler_transform_path = scaler_transform_path,
                                                         leap_years=metadata['leap_years']) 
    
    else:
        raise ValueError(f'Unknown dataset {name}')
    
    return dataset_generator


def load_pandas_dataset(
        name: str,
        dataset_path: Path = None,
        feature_set: str = 'engineered',
        weather: bool = False,
        apply_scaler_transform: str = '',
        scaler_transform_path: Path = None,
        remove_outliers: bool = False) -> PandasBuildingDatasetsFromCSV:
    """
    Load datasets by name.

    Args:
        name (str): Name of the dataset to load.
        dataset_path (Path): Path to the benchmark data. Optional.
        feature_set (str): Feature set to use. Default: 'engineered'.
        weather (bool): load weather data. Default: False
        apply_scaler_transform (str): If not using quantized load or unscaled loads,
                                    applies a {boxcox,standard} scaling transform to the load. Default: ''. 
        scaler_transform_path (Path): Path to data for transform, e.g., pickled data for BoxCox transform.
        remove_outliers (bool): Use version of BuildingsBench with some outliers removed.

    Returns:
        dataset (PandasBuildingDatasetsFromCSV): Generator of Pandas datasets for benchmarking.
    """
    if not dataset_path:
        dataset_path = Path(os.environ.get('BUILDINGS_BENCH', ''))
        if not dataset_path.exists():
            raise ValueError('BUILDINGS_BENCH environment variable not set')
        
    if name.lower() == 'buildings-900k-test':
        raise ValueError(f'{name.lower()} unavailable for now as pandas dataset')

    with open(dataset_path / 'metadata' / 'benchmark.toml', 'rb') as f:
        metadata = tomli.load(f)['buildings_bench']

    if ':' in name.lower():
        name, subset = name.lower().split(':')
        dataset_metadata = metadata[name.lower()]
        all_by_files = parse_building_years_metadata(dataset_path, name.lower())
        all_by_files = filter(lambda by_file: subset in by_file.lower(), all_by_files)
        building_type = dataset_metadata[subset]['building_type']
        building_latlon = dataset_metadata[subset]['latlon']
    else:
        dataset_metadata = metadata[name.lower()]
        all_by_files = parse_building_years_metadata(dataset_path, name.lower())
        building_type = dataset_metadata['building_type']
        building_latlon = dataset_metadata['latlon']
    if remove_outliers:
        dataset_path = dataset_path / 'remove_outliers'

    return PandasBuildingDatasetsFromCSV(
            dataset_path,
            all_by_files,
            building_latlon,
            building_type,
            weather=weather,
            features=feature_set,
            apply_scaler_transform = apply_scaler_transform,
            scaler_transform_path = scaler_transform_path,
            leap_years = metadata['leap_years'])

if __name__ == "__main__":
    dataset = load_torch_dataset("buildings-900k-test")
    for d in dataset:
        print(d[0])