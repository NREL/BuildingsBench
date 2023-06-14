 [![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

# Welcome to BuildingsBench!

![A .gif of a load forecast for CONUS](./assets/commercial_forecast.gif)

## Overview 

BuildingsBench is a platform for
- Large-scale pretraining with the synthetic Buildings-900K dataset for short-term load forecasting (STLF). Buildings-900K is statistically representative of the entire U.S. building stock.
- Benchmarking on two tasks evaluating generalization: zero-shot STLF and transfer learning for STLF.

We provide an index-based PyTorch Dataset for large-scale pretraining, easy data loading for multiple real building energy consumption datasets as PyTorch Tensors or Pandas DataFrames, simple (persistence) to advanced (transformer) baselines, metrics management, and more.


Read more about BuildingsBench in our [documentation](https://nrel.github.io/BuildingsBench/).

### Load a benchmark dataset


```python
import torch
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


## Download the datasets and metadata

Download the tar files to disk and untar, which will create a directory called `BuildingsBench` with the datasets.

The files are accessible for download [here](https://data.openei.org/submissions/5859).

The benchmark datasets are < 1GB in size in total, but the pretraining dataset is ~110GB in size.
The file `metadata.tar.gz` has files that are necessary for running pretraining (such as index files for the Buildings-900K PyTorch Dataset) and the benchmark tasks.


## Run tests

Verify your installation by running unit tests:

```bash
python -m unittest
```

## Usage

We provide scripts in the `./scripts` directory for pretraining and to run the benchmark tasks (zero-shot STLF and transfer learning), either with [our provided baselines](https://nrel.github.io/BuildingsBench/API/models/buildings_bench-models/) or your own model.

Our benchmark assumes each model takes as input a dictionary of torch tensors with the following keys:

```python
{
    'load': torch.Tensor,               # (batch_size, seq_len, 1)
    'building_type': torch.LongTensor,  # (batch_size, seq_len, 1)
    'day_of_year': torch.FloatTensor,   # (batch_size, seq_len, 1)
    'hour_of_day': torch.FloatTensor,   # (batch_size, seq_len, 1)
    'day_of_week': torch.FloatTensor,   # (batch_size, seq_len, 1)
    'latitude': torch.FloatTensor,      # (batch_size, seq_len, 1)
    'longitude': torch.FloatTensor,     # (batch_size, seq_len, 1)
}
```

To use these scripts with your model you'll need to register your model with our platform.

### Registering your model

Make sure to have installed the benchmark in editable mode: `pip install -e .`

1. Create a file called `your_model.py` with your model's implementation, and make your model a subclass of the base model in `./buildings_bench/models/base_model.py`. Make sure to implement the abstract methods: `forward`, `loss`, `load_from_checkpoint`, `predict`, `unfreeze_and_get_parameters_for_finetuning`.
2. Place this file under `./buildings_bench/models/your_model.py.`
3. Import your model class and add your model's name to the `model_registry` dictionary in `./buildings_bench/models/__init__.py`.
4. Create a TOML config file under `./configs/your_model.toml` with each keyword argument your model expects in its constructor (i.e., the hyperparameters for your model) and any additional args for the script you want to run.

The TOML config file should look something like this:

```toml
[model]
# your model's keyword arguments

[pretrain]
# override any of the default pretraining argparse args here

[zero_shot]
# override any of the default zero_shot argparse args here

[transfer_learning]
# override any of the default transfer_learning argparse args here
```
See `./configs/TransformerWithTokenizer-L.toml` for an example.

### Pretraining 

`python3 scripts/pretrain.py --config your_model.toml`

This script is implemented with PyTorch `DistributedDataParallel`, so it can be launched with `torchrun`. See `./scripts/pretrain.sh` for an example.

### Zero-shot STLF

`python3 scripts/zero_shot.py --config your_model.toml --checkpoint /path/to/checkpoint.pt`

### Transfer Learning for STLF

`python3 scripts/transfer_learning_torch.py --config your_model.toml --checkpoint /path/to/checkpoint.pt`  

## BuildingsBench Leaderboard

### Zero-shot STLF

Eval over all real buildings for all available years. Lower is better.

| Model | Commercial NRMSE (%) |  Commercial RPS | Residential NRMSE (%) | Residential RPS | 
| --- | --- | --- | --- | --- |
| [Transformer-L (Gaussian)]() | 13.86 | 5.15 | 83.87 | 0.082 | 
| [Transformer-M (Gaussian)]() | 14.01 | 4.90| 83.17 | 0.085 |
| [Transformer-S (Gaussian)]() | 22.07 | 8.09 | 83.97 | 0.078|
| [Transformer-L (Tokens)]() | 14.82 | 4.81 | 101.7 | 0.072 |
| [Transformer-M (Tokens)]() | 14.54 | 4.67 | 102.1 | 0.071 |
| [Transformer-S (Tokens)]() | 14.88 | 5.17 | 103.2  |0.071 |
| Persistence Ensemble | 17.17 | 5.39 | 80.11 | 0.067 |
| Previous Day Persistence | 17.41 | - | 102.44 | - |
| Previous Week Persistence | 19.96 | - | 103.51 | - |

### Transfer Learning for STLF

Results are over a sub-sample of 100 residential and 100 commercial buildings--see the list of buildings in the datasets metadata directory: `BuildingsBench/metadata/transfer_learning_residential_buildings.csv` and `BuildingsBench/metadata/transfer_learning_commercial_buildings.csv`.
Models are provided with the first 6 months of consumption data for fine-tuning and tested with a 24-hour sliding window on the next 6 months.

| Model | Commercial NRMSE (%) |  Commercial RPS | Residential NRMSE (%) | Residential RPS |
| --- | --- | --- | --- | --- |
| **Pretrained + Fine-tuned** | | | | |
| Transformer (Gaussian) | 13.36 | 3.50 | 82.17 | 0.061 |
| Transformer (Tokens) |  13.86 | 3.73 | 95.55 | 0.060 |
| **Fine-tuned from scratch** | | | | |
| Transformer (Gaussian) | 39.26 |13.24 | 94.25 | 0.080 |
| Transformer (Tokens) |  44.62 | 27.18 | 108.61 | 5.76 |
| LightGBM | 16.63 | - | 83.97 | - |
| DLinear | 39.22 | - | 98.56 | - | 
| Linear Regression | 47.43 | - | 102.17 | - |
| **Persistence** | | | | |
| Persistence Ensemble | 17.41 | 5.16 | 80.13 | 0.058 |
| Previous Day Persistence | 16.98 | - | 101.78 | - |
| Previous Week Persistence | 18.93 | - | 104.38 | - |