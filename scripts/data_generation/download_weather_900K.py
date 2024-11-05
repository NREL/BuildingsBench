import numpy as np
from pathlib import Path
import requests
import pandas as pd
import os
from tqdm import tqdm

def url_retrieve(url: str, outfile: Path):
    R = requests.get(url, allow_redirects=True)
    if R.status_code != 200:
        return

    outfile.write_bytes(R.content)

def main():
    lookup_df = pd.read_csv(Path(os.environ.get('BUILDINGS_BENCH', ''), 'metadata', 'spatial_tract_lookup_table.csv'))
    county = lookup_df['nhgis_2010_county_gisjoin']
    county_list = np.unique(county)
    print('Number of counties:', county_list.shape)

    time_series_dir = Path(os.environ.get('BUILDINGS_BENCH', ''), 'Buildings-900K', 'end-use-load-profiles-for-us-building-stock', '2021')
    building_years = [('comstock_tmy3_release_1', 'tmy3', 'tmy3'), ('resstock_tmy3_release_1', 'tmy3', 'tmy3'), 
                      ('comstock_amy2018_release_1', 'amy2018', '2018'), ('resstock_amy2018_release_1', 'amy2018', '2018')] 
 
    for by in building_years:
        print('Downloading', by[0])
        output_dir = time_series_dir / by[0] / 'weather/'
        for ct in tqdm(county_list):
            url_retrieve(f'https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/{by[0]}/weather/{by[1]}/{ct}_{by[2]}.csv', Path(f'{output_dir}/{ct}.csv'))

   
if __name__ == '__main__':
    main()
