# buildings_bench.data

Functions and class definitions for loading Torch and Pandas datasets.

Main entry points for loading PyTorch and Pandas datasets:

- `load_pretraining()` (used for pretraining)
- `load_torch_dataset()` (used for benchmark tasks)
- `load_pandas_dataset()` (used for benchmark tasks)

Available PyTorch Datasets:

- `Buildings900K` (used for pretraining)
- `TorchBuildingsDataset` (used for benchmark tasks)
- `PandasTransformerDataset` (used for benchmark tasks)

---

## load_pretraining

::: buildings_bench.data.load_pretraining
    options:
        show_source: false
        heading_level: 3
        show_root_heading: true

## load_torch_dataset

::: buildings_bench.data.load_torch_dataset
    options:
        show_source: false
        heading_level: 3
        show_root_heading: true


## load_pandas_dataset

::: buildings_bench.data.load_pandas_dataset
    options:
        show_source: false
        heading_level: 3
        show_root_heading: true

---

## The Buildings-900K PyTorch Dataset

::: buildings_bench.data.buildings900K.Buildings900K
    options:
        show_source: false
        heading_level: 3
        show_root_heading: false

---

## TorchBuildingDataset

::: buildings_bench.data.datasets.TorchBuildingDataset
    options:
        show_source: false
        heading_level: 3
        show_root_heading: false

## PandasTransformerDataset

::: buildings_bench.data.datasets.PandasTransformerDataset
    options:
        show_source: false
        heading_level: 3
        show_root_heading: false

## TorchBuildingDatasetsFromParquet

::: buildings_bench.data.datasets.TorchBuildingDatasetFromParquet
    options:
        show_source: false
        heading_level: 3
        show_root_heading: false

## TorchBuildingDatasetsFromCSV

::: buildings_bench.data.datasets.TorchBuildingDatasetsFromCSV
    options:
        show_source: false
        heading_level: 3
        show_root_heading: false

## PandasBuildingDatasetsFromCSV

::: buildings_bench.data.datasets.PandasBuildingDatasetsFromCSV
    options:
        show_source: false
        heading_level: 3
        show_root_heading: false