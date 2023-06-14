# Welcome to BuildingsBench

## Overview 

BuildingsBench is a platform for:

- Large-scale pretraining with the synthetic Buildings-900K dataset for short-term load forecasting (STLF).
- Benchmarking on two tasks evaluating generalization: zero-shot STLF and transfer learning for STLF.

We provide an index-based PyTorch [Dataset](https://nrel.github.io/BuildingsBench/API/data/buildings_bench-data/#buildings-900k-pytorch-dataset) for large-scale pretraining, easy data loading for multiple real building energy consumption datasets as [PyTorch Tensors](https://nrel.github.io/BuildingsBench/API/data/buildings_bench-data/#buildings_bench.data.datasets.TorchBuildingDatasetsFromCSV) or [Pandas DataFrames](https://nrel.github.io/BuildingsBench/API/data/buildings_bench-data/#buildings_bench.data.datasets.PandasBuildingDatasetsFromCSV), from simple ([persistence](https://nrel.github.io/BuildingsBench/API/models/buildings_bench-models/#buildings_bench.models.persistence)) to advanced ([transformers](https://nrel.github.io/BuildingsBench/API/models/buildings_bench-models/#buildings_bench.models.transformers)) baselines, [metrics management](https://nrel.github.io/BuildingsBench/API/utilities/buildings_bench-evaluation/), a [tokenizer](https://nrel.github.io/BuildingsBench/API/utilities/buildings_bench-tokenizer/) based on KMeans for load time series, and more.


### Load a benchmark dataset


```python
from buildings_bench import load_torch_dataset

# Load a dataset generator for a dataset of buildings
buildings_dataset_generator = load_torch_dataset('bdg-2:panther')

# Each building is a torch.utils.data.Dataset
for building_name, building in buildings_dataset_generator:
    building_dataloader = torch.utils.data.DataLoader(building,
                                                      batch_size=358,
                                                      num_workers=4,
                                                      shuffle=False)
    for sample in building_dataloader:
        x = sample['load']
        # context = x[:, :168], 1 week hourly of context
        # target = x[:, -24:], 24 hour target prediction
        # ...
```

## Installation

To just access the provided dataloaders, models, metrics, etc., install the package with:

```bash
pip install buildings_bench
```

To run the benchmark itself with provided Python scripts, clone this repository and install it in editable mode in a virtual environment or a conda environment.

First, create an environment with `python>=3.8`, for example: `conda create -n buildings_bench python=3.8`.

Then, install the package in editable mode with
```bash
git clone https://github.com/NREL/BuildingsBench.git
cd BuildingsBench
pip install -e ".[benchmark]"
```

### Installing faiss-gpu

Due to a PyPI limitation, we have to install `faiss-gpu` (for KMeans) by directly downloading the wheel from [https://github.com/kyamagu/faiss-wheels/releases/](https://github.com/kyamagu/faiss-wheels/releases/).
Download the wheel for the python version you are using, then install it in your environment.

For example:

```bash
wget https://github.com/kyamagu/faiss-wheels/releases/download/v1.7.3/faiss_gpu-1.7.3-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

pip install faiss_gpu-1.7.3-cp38-cp38-manylinux2014_x86_64.whl
```

### [Optional] LightGBM

If running the LightGBM baseline, you will need to install LightGBM.
Follow instructions [here](https://pypi.org/project/lightgbm/) for your OS. 
Then, `pip install skforecast`.

### Environment variables

Set the environment variable `BUILDINGS_BENCH` to the path where your stored the datasets `BuildingsBench`.

```bash
export BUILDINGS_BENCH=/path/to/BuildingsBench`
```

If using `wandb`, set the following:

- `WANDB_ENTITY`: your wandb username
- `WANDB_PROJECT`: the name of your wandb project for this benchmark


### Download the datasets

Download the tar files to disk and untar, which will create a directory called `BuildingsBench` with the data.

The files will be available for download [here](https://data.openei.org/submissions/5859).


### Run tests

Verify your installation by running unit tests:

```bash
python -m unittest
```
