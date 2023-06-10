 [![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

# Welcome to BuildingsBench!

## Overview 

BuildingsBench is a platform for
- Large-scale pretraining with the simulated Buildings-900K dataset for short-term load forecasting (STLF).
- Benchmarking on two tasks evaluating generalization: zero-shot STLF and transfer learning for STLF.

We provide an index-based PyTorch Dataset for large-scale pretraining, easy-to-use PyTorch and Pandas dataloaders for multiple real building energy consumption datasets, simple to advanced (transformer) baselines, metrics management, and more.


Read more about BuildingsBench in our [documentation](https://nrel.github.io/BuildingsBench/).

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


### Run tests

Verify your installation by running unit tests:

```bash
python -m unittest
```

## Download the datasets

Download the tar files to disk and untar, which will create a directory called `BuildingsBench` with the datasets.

The files are accessible for download [here](https://data.openei.org/submissions/5859).


## Usage

We provide scripts in the `./scripts` directory for pretraining and to run the benchmark tasks (zero-shot STLF and transfer learning), either with our provided baselines or your own model.

Our benchmark assumes each model takes as input a dictionary of torch tensors with the following keys:

```python
{
    'load': torch.Tensor,  # (batch_size, seq_len, 1)
    'building_type': torch.LongTensor,  # (batch_size, 1)
    'day_of_year': torch.FloatTensor,  # (batch_size, 1)
    'hour_of_day': torch.FloatTensor,  # (batch_size, 1)
    'day_of_week': torch.FloatTensor,  # (batch_size, 1)
    'latitude': torch.FloatTensor,  # (batch_size, 1)
    'longitude': torch.FloatTensor,  # (batch_size, 1)
}
```

To use these scripts with your model you'll need to register your model with our platform.

### Registering your model

Make sure to have installed the benchmark in editable mode: `pip install -e .`

1. Create a file called `your_model.py` with your model's implementation, and make your model a subclass of the base model in `./buildings_bench/models/base_model.py`. Make sure to implement the abstract methods: `forward`, `loss`, `load_from_checkpoint`, `predict`, `unfreeze_and_get_parameters_for_finetuning`.
2. Place this file under `./buildings_bench/models/your_model.py.`
3. Import your model class and add your model's name to the `model_registry` dictionary in `BuildingsBench/buildings_bench/models/__init__.py`.
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
