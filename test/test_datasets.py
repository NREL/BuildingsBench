import unittest
from buildings_bench import load_torch_dataset, load_pandas_dataset, load_pretraining
from buildings_bench.data.datasets import PandasBuildingDatasetsFromCSV
from buildings_bench import BuildingTypes
from pathlib import Path
import torch
import os

class TestLoadDatasets(unittest.TestCase):

    def test_dataset_not_in_registry(self):
        with self.assertRaises(ValueError):
            datasets = load_torch_dataset('not-a-dataset')


    def test_load_buildings900k_test(self):
        building_dataset_generator = load_torch_dataset('buildings-900k-test')
        for building_name, building in building_dataset_generator:
                print(f'building {building_name} dataset length: {len(building)}')
                # create a dataloader for the building
                building_dataloader = torch.utils.data.DataLoader(building, batch_size=256, shuffle=False)
                for sample in building_dataloader:
                    x = sample['load']
                    break
                break

    def test_load_electricity(self):
        building_dataset_generator = load_torch_dataset('electricity')
        for building_name, building in building_dataset_generator:
            print(f'building {building_name} dataset length: {len(building)}')
            # create a dataloader for the building
            building_dataloader = torch.utils.data.DataLoader(building, batch_size=256, shuffle=False)
            for sample in building_dataloader:
                x = sample['load']
                break
            break

    def test_load_sceaux(self):
        building_dataset_generator = load_torch_dataset('sceaux')
        for building_name, building in building_dataset_generator:
            print(f'building {building_name} dataset length: {len(building)}')
            # create a dataloader for the building
            building_dataloader = torch.utils.data.DataLoader(building, batch_size=256, shuffle=False)
            for sample in building_dataloader:
                x = sample['load']
                break
            break

    def test_load_borealis(self):
        building_dataset_generator = load_torch_dataset('borealis')
        for building_name, building in building_dataset_generator:
            print(f'building {building_name} dataset length: {len(building)}')
            # create a dataloader for the building
            building_dataloader = torch.utils.data.DataLoader(building, batch_size=256, shuffle=False)
            for sample in building_dataloader:
                x = sample['load']
                break
            break

    def test_load_bdg2_panther(self):
        building_dataset_generator = load_torch_dataset('bdg-2:panther')
        for building_name, building in building_dataset_generator:
            print(f'building {building_name} dataset length: {len(building)}')
            # create a dataloader for the building
            building_dataloader = torch.utils.data.DataLoader(building, batch_size=256, shuffle=False)
            for sample in building_dataloader:
                x = sample['load']
                break
            break

    def test_load_bdg2_rat(self):
        building_dataset_generator = load_torch_dataset('bdg-2:rat')
        for building_name, building in building_dataset_generator:
            print(f'building {building_name} dataset length: {len(building)}')
            # create a dataloader for the building
            building_dataloader = torch.utils.data.DataLoader(building, batch_size=256, shuffle=False)
            for sample in building_dataloader:
                x = sample['load']
                break
            break

    def test_load_ideal(self):
        building_dataset_generator = load_torch_dataset('ideal')
        for building_name, building in building_dataset_generator:
            print(f'building {building_name} dataset length: {len(building)}')
            # create a dataloader for the building
            building_dataloader = torch.utils.data.DataLoader(building, batch_size=256, shuffle=False)
            for sample in building_dataloader:
                x = sample['load']
                break
            break
    

    def test_load_lcl(self):
        building_dataset_generator = load_torch_dataset('lcl')
        for building_name, building in building_dataset_generator:
            print(f'building {building_name} dataset length: {len(building)}')
            # create a dataloader for the building
            building_dataloader = torch.utils.data.DataLoader(building, batch_size=256, shuffle=False)
            for sample in building_dataloader:
                x = sample['load']
                break
            break

    def test_load_smart(self):
        building_dataset_generator = load_torch_dataset('smart')
        for building_name, building in building_dataset_generator:
            print(f'building {building_name} dataset length: {len(building)}')
            # create a dataloader for the building
            building_dataloader = torch.utils.data.DataLoader(building, batch_size=256, shuffle=False)
            for sample in building_dataloader:
                x = sample['load']
                break
            break

    
    def test_load_pandas_datasets_direct(self):
        dataset_path = Path(os.environ['BUILDINGS_BENCH_BENCHMARK'])
        
        building_files = ['BDG-2/Bear_clean=2016', 'BDG-2/Bear_clean=2017']

        datasets = PandasBuildingDatasetsFromCSV(
                    dataset_path,
                    building_files,
                    [23.123, -32.1234],
                    BuildingTypes.COMMERCIAL,
                    features='transformer')

        for building_name, building_dset in datasets:
            print(f'building {building_name}, shape {len(building_dset)}')
            break

        datasets = PandasBuildingDatasetsFromCSV(
                    dataset_path,
                    building_files,
                    [23.123, -32.1234],
                    BuildingTypes.COMMERCIAL,
                    features='engineered'
                    )

        for building_name, building_dset in datasets:
            print(f'building {building_name}, shape {len(building_dset)}')
            break

    def test_load_pandas_datasets(self):
        
        datasets = load_pandas_dataset('bdg-2:bear')

        for building_name, building_dset in datasets:
            print(f'building {building_name}, shape {len(building_dset)}')
            break


if __name__ == '__main__':
    unittest.main()
