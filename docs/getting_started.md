## Datasets 

### Buildings-900K

This is a dataset of energy timeseries derived from simulations of 900K building energy models. The timeseries are hourly and span an entire year. Each building energy model was simulated for two distinct weather years, so in total there are 1.8M timeseries. The weather timeseries and building metadata are also provided. 

### BuildingsBench datasets

In our work, we used Buildings-900K for large-scale pretraining and the BuildingsBench real building datasets for zero-shot forecasting and transfer learning. The BuildingsBench benchmark is a collection of 7 datasets containing smart meter electricity timeseries of real, individual residential and commercial buildings. We also provide temperature timeseries that can be used to condition load forecasts.  

### Accessing the data

The pretraining dataset and evaluation data is available for download [here](https://data.openei.org/submissions/5859) as tar files, or can be accessed via AWS S3 [here](https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=buildings-bench). The benchmark datasets are < 1GB in size in total and the pretraining data is ~110GB in size. 

The pretraining data is divided into 4 compressed files

- `comstock_amy2018.tar.gz`
- `comstock_tmy3.tar.gz`
- `resstock_amy2018.tar.gz`
- `resstock_tmy3.tar.gz`

and one compressed file for the metadata

- `metadata.tar.gz`

The evaluation datasets are compressed into a single file

- `BuildingsBench.tar.gz`

Download all files to a folder on a storage device with at least 250GB of free space. Then, decompress all of the downloaded files. There will be a new subdirectory called `BuildingsBench`. **This is the data directory, which is different than the Github code repository, although both folders are named "BuildingsBench".**

## Dataset directory organization

```bash
BuildingsBench/
├── Buildings-900K/end-use-load-profiles-for-us-building-stock/2021/ # Buildings-900K pretraining data.
    ├── comstock_amy2018_release_1/
        ├── timeseries_individual_buildings/
            ├── by_puma_midwest/
                ├── upgrade=0/
                    ├── puma={puma_id}/*.parquet
                    ├── ...
            ├── by_puma_northeast
            ├── ...
        ├── weather/
            ├── {county_id}.csv
            ├── ...
        ├── metadata/ # Individual building simulation metadata.
            ├── metadata.parquet
    ├── ... # Other pretraining datasets (comstock_tmy3, resstock_amy2018, resstock_tmy3)            
├── BDG-2/ # Building Data Genome Project 2. This is real building smart meter data with outliers removed. 
    ├── {building_id}={year}.csv # The .csv files for the BDG-2 dataset,
    ├── ... # Other buildings in BDG-2.
    ├── weather_{building_id}.csv # Weather data for each building in BDG-2.
├── ... # Other evaluation datasets (Borealis, Electricity, etc.)
├── buildingsbench_with_outliers/ # Copy of the BuildingsBench smart meter data  *with outliers*
├── LICENSES/ # Licenses for each evaluation dataset redistributed in BuildingsBench. 
├── metadata/ # Metadata for the evaluation suite.
    ├── benchmark.toml # Metadata for the benchmark. For each dataset, we specify: `building_type`: `residential` or `commercial`, `latlon`: a List of two floats representing the location of the building(s), `conus_location`: The name of the county or city in the U.S. where the building is located, or a county/city in the U.S. of similar climate to the building's true location (N.b. we do not have nor provide the exact location of buildings), `actual_location`: The name of the county/city/country where the building is actually located which is different from `conus_location` when the building is located outside of CONUS (these values are for book-keeping and can be set to dummy values), `url`: The URL where the dataset was obtained from.
    ├── building_years.txt # List of .csv files included in the benchmark. Each line is of the form `{dataset}/{building_id}={year}.csv`.
    ├── withheld_pumas.tsv # List of PUMAs withheld from the training/validation set of Buildings-900K, which we use as synthetic test data.
    ├── map_of_pumas_in_census_region*.csv # Maps PUMA IDs to their geographical centroid (lat/lon).
    ├── spatial_tract_lookup_table.csv # Mapping between census tract identifiers and other geographies.
    ├── list_oov.py # Python script to generate a list of buildings that are OOV for the Buildings-900K tokenizer.
    ├── oov.txt # List of buildings that are OOV for the Buildings-900K tokenizer.
    ├── transfer_learning_commercial_buildings.txt # List of 100 commercial buildings from the benchmark we use for evaluating transfer learning.
    ├── transfer_learning_residential_buildings.txt # List of 100 residential buildings from the benchmark we use for evaluating transfer learning.
    ├── transfer_learning_hyperparameter_tuning.txt # List of 2 held out buildings (1 commercial, 1 residential) that can be used for hyperparameter tuning.
    ├── train*.idx # Index files for fast dataloading of Buildings-900K. This file uncompressed is ~16GB. 
    ├── val*.idx # Index files for fast dataloading of Buildings-900K.
    ├── transforms # Directory for storing data transform info.
        ├── weather/ # Directory where weather variable normalization parameters are stored.
```

## Dataset Updates

- Version 2.0.0:
    - Added the building simulation metadata files, which contain attributes for the EnergyPlus building energy model used to run the simulation. See `Buildings-900K/end-use-load-profiles-for-us-building-stock/2021/resstock_amy2018_release_1/metadata/metadata.parquet` for an example.
    - Added weather timeseries data. See this [description](https://nrel.github.io/BuildingsBench/running/#weather-timeseries) for more information.


## Buildings-900K parquet file format

The pretraining dataset Buildings-900K is stored as a collection of parquet files. Each parquet file corresponds to a single PUMA, or Public Use Microdata Area, which is a geographic unit used by the U.S. Census Bureau. The parquet file contains the energy timeseries for all buildings assigned to that PUMA.  
Each PUMA-level parquet file in Buildings-900K is stored in a directory with a unique PUMA ID. For example, all residential buildings with weather-year `amy2018` in the northeast census region and PUMA ID `puma_id` can be found under: `Buildings-900K/end-use-load-profiles-for-us-building-stock/2021/resstock_amy2018_release_1/timeseries-individual-buildings/by_puma_northeast/upgrade=0/puma={puma_id}/*.parquet`. 

In the parquet file, the first column is the timestamp and each subsequent column is the energy consumption in kWh for a different building in that. These columns are named by building id. The timestamp is in the format `YYYY-MM-DD HH:MM:SS`. The energy consumption is in kWh.
The parquet files are compressed with snappy. Sort by the timestamp after loading.

```python
import pyarrow.parquet as pq

bldg_id = '00001'
df = pq.read_table('puma={puma_id}', columns=['timestamp', bldg_id]).to_pandas().sort_values(by='timestamp')
```

## Exploring the data

See our dataset quick start [Jupyter notebook](https://github.com/NREL/BuildingsBench/blob/main/tutorials/dataset_quick_start.ipynb)

## CSV file format

We use a simpler CSV file format to store smart meter timeseries data for real buildings, which make up most of the data in the evaluation suite. Most CSV files in the benchmark are named `building_id=year.csv` and correspond to a single building's energy consumption time series. The first column is the timestamp (the Pandas index), and the second column is the energy consumption in kWh. The timestamp is in the format `YYYY-MM-DD HH:MM:SS`. The energy consumption is in kWh. 

Certain datasets have multiple buildings in a single file. In this case, the first column is the timestamp (the Pandas index), and each subsequent column is the energy consumption in kWh for a different building. These columns are named by building id. The timestamp is in the format `YYYY-MM-DD HH:MM:SS`. The energy consumption is in kWh.

## Adding a new dataset

For a new CSV dataset named `{dataset}`

- Create a directory called  `{dataset}` of CSV files with filenames `{building_id}={year}.csv`.
- Add the line `{dataset}/{building_id}={year}.csv` for each file to the `building_years.txt` file.
- Add the appropriate metadata for the dataset to `benchmark.toml` under the `buildings_bench.{dataset}` tag.
- Add `{dataset}` to the benchmark registry in ``buildings_bench/data/__init__.py``.

You can now use the provided torch and pandas dataloaders to load this dataset by name `{dataset}`.

## Out-of-vocab test consumption values

Hourly consumption values > 5100 kWh are larger than the maximum values seen during pretraining on Buildings-900K.
We consider these "out-of-vocab" and remove such buildings from evaluation. 
This prevents errors due to extrapolation, which is not the focus of this benchmark.
See `list_oov.py` for the code we use to generate a list of OOV buildings.