
"""
Randomly select 100 residential and 100 commercial buildings 
from BuildingsBench for transfer learning.
"""
from buildings_bench import load_pandas_dataset, benchmark_registry
from pathlib import Path
import numpy as np
import random
import os

dataset_path = Path(os.environ.get('BUILDINGS_BENCH', ''))
datasets = [b for b in benchmark_registry if b != 'buildings-900k-test']

def consecutive(arr):
    if len(arr) == 1:
        return True
        
    diffs = arr[1:] - arr[:-1]
    if (diffs>1).any():
        return False
    return True


residential_buildings = set()
commercial_buildings = set()
    
for dataset in datasets:
    building_datasets_generator = load_pandas_dataset(dataset, dataset_path =dataset_path)
    for building_name, building_dataset in building_datasets_generator:
        years = building_dataset.index.year.unique()
        years = np.sort(years)
                
        # ensure the data we have across multiple years spans consecutive years,
        # else we might try to use data from Dec 2014 and Jan 2016 for example.
        # 
        # also ensure the building has at least ~210 days of data
        if consecutive(years) and len(building_dataset) > (180+30)*24:
                
            if building_datasets_generator.building_type == 'residential':
                residential_buildings.add(building_name)
            elif building_datasets_generator.building_type == 'commercial':
                commercial_buildings.add(building_name)

# sample 100 residential buildings
residential_buildings = list(residential_buildings)
commercial_buildings = list(commercial_buildings)

np.random.seed(100)
random.seed(100)

residential_buildings = random.sample(residential_buildings, 100)
commercial_buildings = random.sample(commercial_buildings, 100)

with open(dataset_path / 'metadata' / 'transfer_learning_residential_buildings.txt', 'w') as f:
    for b in residential_buildings:
        f.write(f'{b}\n')

with open(dataset_path / 'metadata' / 'transfer_learning_commercial_buildings.txt', 'w') as f:
    for b in commercial_buildings:
        f.write(f'{b}\n')
