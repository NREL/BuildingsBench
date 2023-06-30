# BuildingsBench Scripts

## Evaluation
- `pretrain.py`: Launch pretraining on Buildings-900K.
- `pretrain.sh`: Example `torchrun` launch script.
- `zero_shot.py`: Evaluate a model on BuildingsBench benchmark on the zero-shot STLF task.
- `transfer_learning_torch.py`: Evaluate a PyTorch model on the BuildingsBench benchmark transfer learning task.
- `transfer_learning_lightgbm.py`: Evaluate LightGBM on the BuildingsBench benchmark transfer learning task.

## Benchmark data creation
- `./process_raw_data`
  - `create_buildings900K.py`: Launch a PySpark job to process the raw EULP database and save the Buildings-900K Parquet files.
  - `download_and_process_buildingsbench.py`: Replicate the preprocessing applied to the BuildingsBench evaluation datasets.
- `create_index_files.py`: Create index files for the Buildings-900K PyTorch Dataset.
- `create_index_files_with_less_buildings.py`: Create index files for the minified Buildings-900K datasets with less buildings for analytical studies on data scaling.
- `fit_tokenizer.py`: Train a new tokenizer. Requires `faiss-gpu` to be installed.
- `fit_scaler_transforms.py`: Fit the Box-Cox and StandardScaler transforms on the Buildings-900K training set.
- `subsample_buildings_for_transfer_learning.py`: Replicate the steps taken to select 100 residential and 100 commercial buildings for the BuildingsBench transfer learning task.
