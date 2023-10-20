# Welcome to BuildingsBench

## Overview 

BuildingsBench is a platform for:

- Large-scale pretraining with the synthetic Buildings-900K dataset for short-term load forecasting (STLF). Buildings-900K is statistically representative of the entire U.S. building stock and is extracted from the NREL [End-Use Load Profiles database](https://www.nrel.gov/buildings/end-use-load-profiles.html).
- Benchmarking on two tasks evaluating generalization: zero-shot STLF and transfer learning for STLF.

We provide an index-based PyTorch [Dataset](https://nrel.github.io/BuildingsBench/API/data/buildings_bench-data/#the-buildings-900k-pytorch-dataset) for large-scale pretraining, easy data loading for multiple real building energy consumption datasets as [PyTorch Tensors](https://nrel.github.io/BuildingsBench/API/data/buildings_bench-data/#torchbuildingdatasetsfromcsv) or [Pandas DataFrames](https://nrel.github.io/BuildingsBench/API/data/buildings_bench-data/#pandasbuildingdatasetsfromcsv), from simple persistence to advanced transformer baselines, [metrics management](https://nrel.github.io/BuildingsBench/API/utilities/buildings_bench-evaluation/), a [tokenizer](https://nrel.github.io/BuildingsBench/API/utilities/buildings_bench-tokenizer/) based on KMeans for load time series, and more.

Read more about BuildingsBench in our [paper](https://arxiv.org/abs/2307.00142).


## Getting started 

### Installation

If you aren't going to pretrain or evaluate models and just want access to the provided dataloaders, model code, metrics computation, etc., install the package with:

```bash
pip install buildings_bench
```

#### Full installation

Otherwise, clone this repository and install it in editable mode in a virtual environment or a conda environment.

1. Create an environment with `python>=3.8`, for example: `conda create -n buildings_bench python=3.8`.
2. Install the package in editable mode with
```bash
git clone https://github.com/NREL/BuildingsBench.git
cd BuildingsBench
pip install -e ".[benchmark]"
```

#### Installing faiss-gpu

Due to a PyPI limitation, we have to install `faiss-gpu` (for KMeans) by directly downloading the wheel from [https://github.com/kyamagu/faiss-wheels/releases/](https://github.com/kyamagu/faiss-wheels/releases/).
Download the wheel for the python version you are using, then install it in your environment.

For example:

```bash
wget https://github.com/kyamagu/faiss-wheels/releases/download/v1.7.3/faiss_gpu-1.7.3-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

pip install faiss_gpu-1.7.3-cp38-cp38-manylinux2014_x86_64.whl
```

#### [Optional] Installing LightGBM

If running the LightGBM baseline, you will need to install LightGBM.

1. Follow instructions [here](https://pypi.org/project/lightgbm/) for your OS. 
2. Then install `skforecast` with `pip install skforecast==0.8.1`.

#### Environment variables

Set the environment variable `BUILDINGS_BENCH` to the path where the data directory `BuildingsBench` is located (created when untarring the data files). **This is not the path to the code repository.**

```bash
export BUILDINGS_BENCH=/path/to/BuildingsBench
```

##### Wandb 

If using `wandb`, set the following:

- `WANDB_ENTITY`: your wandb username
- `WANDB_PROJECT`: the name of your wandb project for this benchmark


### Run tests

Verify your installation by running unit tests:

```bash
python -m unittest
```

## Next steps

1. [Download and get familiar with the datasets](https://nrel.github.io/BuildingsBench/datasets/)
2. [Learn how to download a pretrained model and run it on a building dataset](https://github.com/NREL/BuildingsBench/blob/main/tutorials/pretrained_models.ipynb)
3. [Learn how to run a custom model on the benchmark](https://github.com/NREL/BuildingsBench/blob/main/tutorials/registering_your_model_with_the_benchmark.ipynb)
4. [Computing metrics and interpreting the results](https://github.com/NREL/BuildingsBench/blob/main/tutorials/aggregate_benchmark_results.ipynb)

[List of all tutorials](https://nrel.github.io/BuildingsBench/tutorials/).

## Citation

If you use BuildingsBench in your research, please cite our preprint:

```
@article{emami2023buildingsbench,
  title={BuildingsBench: A Large-Scale Dataset of 900K Buildings and Benchmark for Short-Term Load Forecasting},
  author={Emami, Patrick and Sahu, Abhijeet and Graf, Peter},
  journal={arXiv preprint arXiv:2307.00142},
  year={2023}
}
```
