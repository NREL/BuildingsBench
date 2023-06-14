# buildings_bench.data

Main entry point for loading PyTorch and Pandas datasets:

- `load_pretraining()` (used for pretraining)
- `load_torch_dataset()` (used for benchmark tasks)
- `load_pandas_dataset()` (used for benchmark tasks)

Available PyTorch Datasets:

- `Buildings900K` (used for pretraining)
- `TorchBuildingsDataset` (used for benchmark tasks)
- `PandasTransformerDataset` (used for benchmark tasks)

---

## Loading benchmark datasets

Function definitions for loading Torch and Pandas datasets.

::: buildings_bench.data
    options:
        show_source: false
        heading_level: 2
        show_root_heading: true

---

## Buildings-900K PyTorch Dataset

::: buildings_bench.data.buildings900K
    options:
        show_source: false
        heading_level: 2
        show_root_heading: true

---

## Generators and Datasets

::: buildings_bench.data.datasets
    options:
        show_source: false
        heading_level: 2
        show_root_heading: true
