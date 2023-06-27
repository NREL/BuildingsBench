## Running The Benchmark

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

To use these scripts with your model you'll need to register your model with our platform. 

### Registering your model

Please see this [step-by-step tutorial](https://github.com/NREL/BuildingsBench/blob/main/tutorials/registering_your_model_with_the_benchmark.ipynb) for a Jupyter Notebook version of the following instructions.

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