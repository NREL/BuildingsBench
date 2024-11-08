We provide scripts in the `./scripts` directory for pretraining and to run the benchmark tasks (zero-shot STLF and transfer learning), either with our provided baselines or your own model.

PyTorch checkpoint files for our trained models are available for download as a single tar file  [here](https://oedi-data-lake.s3.amazonaws.com/buildings-bench/v1.0.0/compressed/checkpoints.tar.gz) or as individual files on S3 [here](https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=buildings-bench%2Fv1.0.0%2Fcheckpoints%2F).


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

As of v2.0.0, models can also optionally take a `temperature` timeseries tensor as input. 

Arguments for each model are specified in a TOML configuration file in the `./buildings_bench/configs` directory.
If you just want to modify the arguments for a provided model type, you can do so either by modifying the provided TOML config file or by creating a new TOML config file.
To add your own custom model, you'll need to follow a few steps to register your model with our platform.

## Registering your model

Please see this [step-by-step tutorial](https://github.com/NREL/BuildingsBench/blob/main/tutorials/registering_your_model_with_the_benchmark.ipynb) for a Jupyter Notebook version of the following instructions.

Make sure to have installed the benchmark in editable mode: `pip install -e .[benchmark]`

1. Create a file called `your_model.py` with your model's implementation, and make your model a subclass of the base model in `./buildings_bench/models/base_model.py`. Make sure to implement the abstract methods: `forward`, `loss`, `load_from_checkpoint`, `predict`, `unfreeze_and_get_parameters_for_finetuning`.
2. Place this file under `./buildings_bench/models/your_model.py.`
3. Import your model class and add your model's name to the `model_registry` dictionary in `./buildings_bench/models/__init__.py`.
4. Create a TOML config file under `./buildings_bench/configs/your_model.toml` with each keyword argument your model expects in its constructor (i.e., the hyperparameters for your model) and any additional args for the script you want to run.

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
See `./buildings_bench/configs/TransformerWithTokenizer-S.toml` for an example.

## Pretraining

#### Without SLURM

The script `pretrain.py` is implemented with PyTorch `DistributedDataParallel` so it must be launched with `torchrun` from the command line and the argument `--disable_slurm` must be passed.
See `./scripts/pretrain.sh` for an example.


```bash
#!/bin/bash

export WORLD_SIZE=1
NUM_GPUS=1

torchrun \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    scripts/pretrain.py --model TransformerWithGaussian-S --disable_slurm
```

The argument `--disable_slurm` is not needed if you are running this script on a Slurm cluster as a batch job.

This script will automatically log outputs to `wandb` if the environment variables `WANDB_ENTITY` and `WANDB_PROJECT` are set. Otherwise, pass the argument `--disable_wandb` to disable logging to `wandb`.

#### With SLURM

To launch pretraining as a SLURM batch job:

```bash
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

srun python3 scripts/pretrain.py \
        --model TransformerWithGaussian-S
```


## Zero-shot Evaluation

The script `scripts/zero_shot.py` and the script for transfer learning `scripts/transfer_learning_torch.py` do not use `DistributedDataParallel` so they can be run without `torchrun`.

`python3 scripts/zero_shot.py --model TransformerWithGaussian-S --checkpoint /path/to/checkpoint.pt`

## Transfer Learning Evaluation

`python3 scripts/transfer_learning_torch.py --model TransformerWithGaussian-S --checkpoint /path/to/checkpoint.pt`  

## Weather Timeseries

An important data source for forecasting building energy usage is the external weather condition. This significantly impacts energy usage in buildings, for example, when high temperatures lead to increases in cooling demand. In BuildingsBench v2.0.0, we have added weather timeseries data for each building in Buildings-900K and for each test building in the BuildingBench evaluation suite. In particular, we support pretraining and evaluation with a temperature timeseries input. The outdoor *temperature* is the most impactful weather feature for load forecasting. 

In detail, a forecasting model can be provided with both the past one week of temperature timeseries data as well as the temperature for the 24 hour prediction horizon. We note that some of the building datasets in our benchmark have more variables available beyond just temperature.

Summary of available weather data: 

* Buildings-900K weather: For each PUMA and year (`amy2018`, `tmy3`), there is a corresponding weather csv file. That is, a residential building in the same PUMA has the same `amy2018` weather timeseries as a commercial building in that PUMA for `amy2018`. This file has hourly annual weather data with the following 7 variables: Dry Bulb Temperature (°C), Relative Humidity (%), Wind Speed (m/s), Wind Direction (Deg),Global Horizontal Radiation (W/m2), Direct Normal Radiation (W/m2), Diffuse Horizontal Radiation (W/m2). We note that these weather files are the same ones used by the EnergyPlus simulator to create these synthetic load timeseries. 

* BDG-2 and SMART datasets weather: These datasets provide per-building hourly temperature and humidity timeseries, which we include.  

* Other BuildingsBench evaluation datasets: The Electricity, Borealis, IDEAL, LCL, Sceaux do not provide weather data. We collected the temperature timeseries ourselves from the National Oceanic and Atmospheric Administration's (NOAA) Integrated Surface Database (ISD), managed by the National Centers for Environmental Information (NCEI).  

An important caveat is that we are *not* using 24-hour weather *forecasts* as inputs to our load forecasting model. Rather, we are providing the models with the actual day-ahead weather that was recorded. In reality, we do not know tomorrow's weather and so our models must normally rely on a (potentially inaccurate) weather forecast. 

#### Training and evaluation with weather data

To train or evaluate a model that uses temperature timeseries inputs, create a new model configuration TOML file in the `buildings_bench/configs` folder to include the `weather_inputs` key:

```toml
[model]

weather_inputs = ['temperature']

```

The `weather_inputs` key is a list of strings that correspond to the weather variables you want to include in your model and load from the corresponding datasets. 

This will automatically add keys to the model's batch dictionary with the same names as the weather variables:

```python
{
    'load': torch.Tensor,               # (batch_size, seq_len, 1)
    'building_type': torch.LongTensor,  # (batch_size, seq_len, 1)
    'day_of_year': torch.FloatTensor,   # (batch_size, seq_len, 1)
    'hour_of_day': torch.FloatTensor,   # (batch_size, seq_len, 1)
    'day_of_week': torch.FloatTensor,   # (batch_size, seq_len, 1)
    'latitude': torch.FloatTensor,      # (batch_size, seq_len, 1)
    'longitude': torch.FloatTensor,     # (batch_size, seq_len, 1)
    'temperature': torch.FloatTensor,   # (batch_size, seq_len, 1)
}
```

Then, launch model training in the usual way: 

`python3 scripts/zero_shot.py --model TransformerWithGaussian-t-S`

```bash
export WORLD_SIZE=1

torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    scripts/pretrain.py \
    --model TransformerWithGaussian-t-S \
    --disable_slurm
```

We provide default small (S), medium (M), and large (L) model configs for models that expect `temperature` timeseries inputs: `TransformerWithGaussian-t-*`, `temperature` and `humidity` inputs: `TransformerWithGaussian-th-*`, and all available weather variables for the synthetic Buildings-900K data: `TransformerWithGaussian-weather-*`.