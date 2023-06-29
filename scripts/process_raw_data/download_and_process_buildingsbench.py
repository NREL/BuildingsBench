"""
Basic preprocessing steps for real building data

1. Resample sub-hourly kW data to hourly kWh data
2. Remove buildings with more than 10% of missing values
3. Linearly interpolate missing hour data
4. Save to csv
"""
import pandas as pd
import numpy as np
import glob
from pathlib import Path 
import os
import urllib
import random
    
    
def download_electricity_data(savedir):
    url = 'https://archive.ics.uci.edu/static/public/321/'
    urllib.request.urlretrieve(url, 'electricityloaddiagrams20112014.zip')
    os.system('unzip electricityloaddiagrams20112014.zip')
    os.system('rm electricityloaddiagrams20112014.zip')
    os.system(f'mv LD2011_2014.txt {savedir}')

def download_bdg2_data(savedir):
    url = 'https://github.com/buds-lab/building-data-genome-project-2'
    print(f'use git lfs to download the data from the git repo {url} '
          f' then copy it to {savedir}')


def download_sceaux_data(savedir):
    url = 'https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set'
    print(f'Log into Kaggle then download the dataset from {url}. Unzip the file archive.zip and copy '
          f'the file household_power_consumption.txt to {savedir}')

def download_smart_data(savedir):
    url_Homes = ['https://lass.cs.umass.edu/smarttraces/2017/HomeA-electrical.tar.gz',
                 'https://lass.cs.umass.edu/smarttraces/2017/HomeB-electrical.tar.gz',
                 'https://lass.cs.umass.edu/smarttraces/2017/HomeC-electrical.tar.gz',
                 'https://lass.cs.umass.edu/smarttraces/2017/HomeD-electrical.tar.gz',
                 'https://lass.cs.umass.edu/smarttraces/2017/HomeF-electrical.tar.gz',
                 'https://lass.cs.umass.edu/smarttraces/2017/HomeG-electrical.tar.gz',
                 'https://lass.cs.umass.edu/smarttraces/2017/HomeH-electrical.tar.gz',
    ]
    # untar each 
    for url in url_Homes:
        os.system(f'wget {url}')
        os.system(f'tar -xzf {url.split("/")[-1]}')
        os.system(f'rm {url.split("/")[-1]}')
        # move each to savedir
        os.system(f'mv {url.split("/")[-1].split("-")[0]} {savedir}')

def download_borealis_data(savedir):
    url = 'https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/R4SVBF'
    print(f'Navigate to {url} and download the .zip datasest. Unzip the file and copy '
          f'the folder 6secs_load_measurement_dataset to {savedir}')
    
def download_ideal_data(savedir):
    url = 'https://datashare.ed.ac.uk/bitstream/handle/10283/3647/household_sensors.zip'
    os.system(f'wget {url}')
    os.system(f'unzip household_sensors.zip')
    os.system(f'rm household_sensors.zip')
    os.system(f'mv sensordata {savedir}')
    
def download_lcl_data(savedir):
    url = 'https://data.london.gov.uk/download/smartmeter-energy-use-data-in-london-households/04feba67-f1a3-4563-98d0-f3071e3d56d1/Partitioned%20LCL%20Data.zip'
    os.system(f'wget {url}')
    os.system(f'unzip Partitioned\ LCL\ Data.zip')
    os.system(f'rm Partitioned\ LCL\ Data.zip')
    os.system(f'mv Small\ LCL\ Data/ {savedir}')

if __name__ == '__main__':
    dataset_dir = Path(os.environ.get('BUILDINGS_BENCH', ''))

    print('WARNING: This script will try to automatically download and process the datasets in BuildingsBench.'
          ' Please be sure to have followed the instructions for downloading the data from the following datasets, '
          'which are not able to be automatically downloaded: ')
    print('>>> BDG-2')
    download_bdg2_data(dataset_dir / 'BDG-2')
    print('>>> Sceaux')
    download_sceaux_data(dataset_dir / 'Sceaux')
    print('>>> Borealis')
    download_borealis_data(dataset_dir / 'Borealis')    


    ########################################################
    print('Electricity...')
    ########################################################       
    (dataset_dir / 'Electricity').mkdir(parents=True, exist_ok=True)
    download_electricity_data(dataset_dir / 'Electricity')

    df = pd.read_csv(dataset_dir / 'Electricity' / 'LD2011_2014.txt', sep=';', index_col=0)
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')
    df = df.asfreq('15min')
    df = df.sort_index()

    # Set the type of each column to float
    # Convert commas in each string to a period
    for col in df.columns:
        # if column entries are type str
        if df[col].dtype == 'object' and col != 'index':
            df[col] = df[col].apply(lambda x: float(x.replace(',', '.') if type(x) is str else x))

    df = df.astype('float32')

    # Resample 15 min -> hourly
    df = df.resample(rule='H', closed='left', label='right').mean()

    # Group buildings by year
    years = [2011, 2012, 2013, 2014]
    for year in years:
        bldgs = df[df.index.year == year]
        # Replace zeros with nan for this dataset, which uses zeros to indicate missing values
        bldgs = bldgs.replace(0, np.nan)
        # Drop buildings with more than 10% of missing values
        bldgs = bldgs.dropna(thresh=len(bldgs)*0.9, axis=1)
        # linearly interpolate nan values
        bldgs = bldgs.interpolate(method='linear', axis=0, limit=24*7, limit_direction='both')
        # Replace any remaining nan values with zeros (> 1 week)
        bldgs = bldgs.fillna(0)
        # Name the index column 'timestamp'
        bldgs.index.name = 'timestamp'
        bldgs = bldgs.asfreq('1H')
        # Save to csv
        bldgs.to_csv(dataset_dir / 'Electricity' / f'LD2011_2014_clean={year}.csv', index=True, header=True)

    ########################################################
    print('BDG-2...')
    ########################################################

    (dataset_dir / 'BDG-2').mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(dataset_dir / 'electricity_cleaned.csv', sep=',')
         # set timestamp as index
        df = df.set_index('timestamp')
        df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')

        sites = ['Bear_', 'Fox_', 'Panther_', 'Rat_']
        years = [2016, 2017]
        for s in sites:
            # regex filter for Panther
            df_site = df.filter(regex=s)
        
            for year in years:
                bldgs = df_site[df_site.index.year == year]
                # Drop columns with more than 10% of missing values
                bldgs = bldgs.dropna(thresh=len(bldgs)*0.9, axis=1)
                # linearly interpolate nan values (1 week limit)
                bldgs = bldgs.interpolate(method='linear', axis=0, limit=24*7, limit_direction='both')
                # Replace any remaining nan values with zeros (> 1 week)
                bldgs = bldgs.fillna(0)
                sit = s.replace('_', '')
                bldgs.to_csv(dataset_dir / 'BDG-2' / f'{sit}_clean={year}.csv', header=True, index=True)
    except:
        download_bdg2_data(dataset_dir / 'BDG-2')
        print('skipping BDG-2...')


    ########################################################
    print('Sceaux...')
    ########################################################

    try:
        (dataset_dir / 'Sceaux').mkdir(parents=True, exist_ok=True)
        house = pd.read_csv(dataset_dir / 'Sceaux' / 'household_power_consumption.txt', sep=';', header=0)

        # Combine Date and Time columns
        house['timestamp'] = house['Date'] + ' ' + house['Time']
        # Set timestamp as index
        house = house.set_index('timestamp')
        # Convert index to datetime
        house.index = pd.to_datetime(house.index, format='%d/%m/%Y %H:%M:%S')
        # Drop Date and Time columns
        house = house.drop(['Date', 'Time'], axis=1)

        # Replace ? values with NaN
        house = house.replace('?', np.nan)
        house = house.astype('float32')
        house = house.asfreq('1min')
        house = house.sort_index()
        # resample to hourly data, ignoring nan values
        house = house.resample(rule='H', closed='left', label='right').mean()
        years = [2007, 2008, 2009, 2010]
        for year in years:
            h = house[house.index.year == year]
            # Linearly interpolate missing values
            h = h.interpolate(method='linear', axis=0, limit=24*7, limit_direction='both')
            # Fill remaining missing values with zeros
            h = h.fillna(0)
            # Keep Global_active_power column
            h = h[['Global_active_power']]
            # Save to csv
            h.to_csv(dataset_dir / 'Sceaux' / f'Sceaux_clean={year}.csv', header=True, index=True)
    except:
        download_sceaux_data(dataset_dir / 'Sceaux')
        print('skipping Sceaux...')

    ########################################################
    print('SMART...')
    ########################################################
    
    (dataset_dir / 'SMART').mkdir(parents=True, exist_ok=True)
    download_smart_data(dataset_dir / 'SMART')
    homes = [('HomeB', 'meter1'), ('HomeC', 'meter1'), ('HomeD', 'meter1'), ('HomeF', 'meter2'), ('HomeG', 'meter1'), ('HomeH', 'meter1')]
    years = [2014, 2015, 2016]

    power_column = 'use [kW]'

    # Load data
    for idx in range(len(homes)):
        home = homes[idx][0]
        meter = homes[idx][1]
        for year in years:
            try:
                # join on timestamp index
                house_ = pd.read_csv(dataset_dir / 'SMART' / f'{home}/{year}/{home}-{meter}_{year}.csv')
            except:
                continue
            # Rename Date & Time to timestamp
            house_ = house_.rename(columns={'Date & Time': 'timestamp'})
            # Set timestamp as index
            house_ = house_.set_index('timestamp')
            # Convert index to datetime
            house_.index = pd.to_datetime(house_.index, format='%Y-%m-%d %H:%M:%S')
            # Replace missing values with NaN
            house_ = house_.replace('?', np.nan)
            # Resample to hourly
            house_ = house_.resample(rule='H', closed='left', label='right').mean()

            # only keep use [kW] column
            if 'use [kW]' in house_.columns:
                house_ = house_[['use [kW]']]
                power_column = 'use [kW]'
            if 'Usage [kW]' in house_.columns:
                house_ = house_[['Usage [kW]']]
                power_column = 'Usage [kW]'

            house_ = house_.rename(columns={power_column: 'power'})
            
            missing_frac = house_['power'].isnull().sum() / house_.shape[0]
            # Calculate fraction of missing values
            #print(f'Fraction of NaN values in {home}-{meter} {year}: {missing_frac}')
            # Calculate fraction of 0's
            zeros_frac = (house_['power'] == 0.0).sum() / house_.shape[0]
            #print(f'Fraction of 0 values in {home}-{meter} {year}: {zeros_frac}')

            if missing_frac <= 0.1 and zeros_frac <= 0.1:
                # Linearly interpolate missing values
                house_ = house_.interpolate(method='linear', axis=0, limit=24*7, limit_direction='both') 
                # Fill remaining missing values with zeros
                house_ = house_.fillna(0)
                house_.to_csv(dataset_dir / 'SMART' / f'{home}_clean={year}.csv', header=True, index=True)

    ########################################################
    print('Borealis...')
    ########################################################

    (dataset_dir / 'Borealis').mkdir(parents=True, exist_ok=True)

    try:
        homes = glob.glob(str(dataset_dir / 'Borealis' / '6secs_load_measurement_dataset' / '*.csv'))
        years = [2010, 2011, 2012]
   
        # Load data
        for home in homes:
            house_ = pd.read_csv(dataset_dir / 'Borealis'  / '6secs_load_measurement_dataset' / f'{home}.csv')

            # Set timestamp as index
            house_ = house_.set_index('timestamp')
            # drop row 1, which has 0 value
            house_ = house_.drop(house_.index[0])
            # Convert index to datetime
            house_.index = pd.to_datetime(house_.index, format='%Y-%m-%d %H:%M:%S')
            # only keep power column
            house_ = house_[['power']]
            # Resample to hourly
            house_ = house_.resample(rule='H', closed='left', label='right').mean()
            # convert from Wh to kHh
            house_ = house_ / 1000

            for year in years:
                h = house_[house_.index.year == year]
                # count number of rows
                if h.shape[0] < 168: # if < 1 week, throw out building-year
                    continue
                missing_frac = h["power"].isnull().sum() / h.shape[0]

                if missing_frac <= 0.1:
                    # Linearly interpolate
                    h = h.interpolate(method='linear', axis=0, limit=24*7, limit_direction='both')
                    # Fill missing values with zeros
                    h = h.fillna(0)
                    # Save to csv
                    h.to_csv(dataset_dir / 'Borealis' / f'{home}_clean={year}.csv', header=True, index=True)

    except:
        download_borealis_data(dataset_dir / 'Borealis')
        print('skipping Borealis...')

  
    ########################################################
    print('IDEAL...')
    ########################################################
    (dataset_dir / 'IDEAL').mkdir(parents=True, exist_ok=True)
    
    print('WARNING: the raw IDEAL dataset requires over 140GB of storage. Skip?')
    skip = input('Skip IDEAL? (y/n): ')
    if skip == 'y':
        print('skipping IDEAL...')
    else:
        print('this may take a while...')
        download_ideal_data(dataset_dir / 'IDEAL')
        homes_gz = glob.glob(str(dataset_dir / 'IDEAL' / 'sensordata' / 'home*_electric-mains_electric-combined.csv.gz'))
        # gunzip each file
        for h in homes_gz:
            os.system('gunzip ' + h)
        homes = glob.glob(str(dataset_dir / 'IDEAL' / 'sensordata' / 'home*_electric-mains_electric-combined.csv'))

        years = [2016, 2017, 2018]

        for h in homes:
            homeid = Path(h).name.split('_')[0]
            home = pd.read_csv(h, names=['power'], index_col=0, header=None)
            # drop the first row
            home = home.drop(home.index[0])
            # convert index to datetime
            home.index = pd.to_datetime(home.index, format='%Y-%m-%d %H:%M:%S')
            # convert from Wh to kHh
            home = home / 1000
            # resample to 1 hour
            home = home.resample(rule='H', closed='left', label='right').mean() 

            for year in years:
                ho = home[home.index.year == year]
                if ho.shape[0] < 168: # skip empty
                    continue
                missing_frac = (ho['power'].isnull().sum() / ho.shape[0])
                #print(f'{h} {year} missing frac = {missing_frac}')
                if missing_frac <= 0.1:
                    # linearly interpolate
                    ho = ho.interpolate(method='linear', axis=0, limit=24*7, limit_direction='both')
                    # Replace remaining nan with 0
                    ho = ho.fillna(0)
                    # save
                    ho.to_csv( dataset_dir / 'IDEAL' / f'{homeid}_clean={year}.csv', header=True, index=True)
            del home

    ########################################################
    print('LCL...')
    ########################################################

    all_buildings = {'MAC001964', 'MAC003533', 'MAC003527', 'MAC005461', 'MAC000339', 'MAC000767', 'MAC001706', 'MAC003238', 'MAC000065', 'MAC002146', 'MAC001945', 'MAC003968', 'MAC001764', 'MAC000710', 'MAC000712', 'MAC005444', 'MAC000699', 'MAC000677', 'MAC005101', 'MAC000790', 'MAC002841', 'MAC005496', 'MAC003511', 'MAC002290', 'MAC004665', 'MAC005116', 'MAC000774', 'MAC003366', 'MAC004672', 'MAC000702', 'MAC002227', 'MAC002143', 'MAC000051', 'MAC001953', 'MAC000633', 'MAC002141', 'MAC005494', 'MAC005129', 'MAC004906', 'MAC000451', 'MAC003915', 'MAC000761', 'MAC000470', 'MAC000783', 'MAC005441', 'MAC001954', 'MAC000354', 'MAC002121', 'MAC004631', 'MAC000467', 'MAC000779', 'MAC003807', 'MAC001576', 'MAC003526', 'MAC005094', 'MAC004667', 'MAC000722', 'MAC005464', 'MAC001566', 'MAC003360', 'MAC005499', 'MAC004659', 'MAC000765', 'MAC003548', 'MAC002872', 'MAC003198', 'MAC004997', 'MAC003185', 'MAC005567', 'MAC000044', 'MAC001740', 'MAC005469', 'MAC005089', 'MAC004664', 'MAC001826', 'MAC005107', 'MAC005438', 'MAC000141', 'MAC000106', 'MAC002292', 'MAC003298', 'MAC003516', 'MAC005500', 'MAC000690', 'MAC003540', 'MAC002247', 'MAC003517', 'MAC003315', 'MAC005467', 'MAC004919', 'MAC003977', 'MAC000706', 'MAC003338', 'MAC001651', 'MAC000725', 'MAC005113', 'MAC003852', 'MAC001814', 'MAC003848', 'MAC000402', 'MAC001963', 'MAC000384', 'MAC004662', 'MAC001694', 'MAC000383', 'MAC004905', 'MAC001547', 'MAC002219', 'MAC004634', 'MAC002242', 'MAC005133', 'MAC005430', 'MAC005443', 'MAC004923', 'MAC003510', 'MAC002876', 'MAC000788', 'MAC000755', 'MAC000766', 'MAC005439', 'MAC003782', 'MAC001614', 'MAC000751', 'MAC003268', 'MAC001602', 'MAC003895', 'MAC001682', 'MAC005505', 'MAC000082', 'MAC002300', 'MAC004635', 'MAC002124', 'MAC001856', 'MAC002232', 'MAC003321', 'MAC001758', 'MAC005112', 'MAC005456', 'MAC004991', 'MAC002153', 'MAC001949', 'MAC003256', 'MAC003823', 'MAC000713', 'MAC001970', 'MAC001979', 'MAC003530', 'MAC004914', 'MAC001754', 'MAC000328', 'MAC000794', 'MAC005097', 'MAC001645', 'MAC000707', 'MAC004653', 'MAC005501', 'MAC003557', 'MAC001947', 'MAC002838', 'MAC000401', 'MAC002847', 'MAC003954', 'MAC000679', 'MAC001655', 'MAC000701', 'MAC001975', 'MAC001958', 'MAC004983', 'MAC005561', 'MAC000048', 'MAC000693', 'MAC000772', 'MAC002875', 'MAC004663', 'MAC004936', 'MAC001778', 'MAC004994', 'MAC004650', 'MAC001755', 'MAC004993', 'MAC004928', 'MAC000640', 'MAC003553', 'MAC000695', 'MAC001737', 'MAC004646', 'MAC000683', 'MAC005462', 'MAC005007', 'MAC003542', 'MAC005003', 'MAC003951', 'MAC001743', 'MAC002866', 'MAC000700', 'MAC001536', 'MAC000680', 'MAC003247', 'MAC000786', 'MAC000757', 'MAC002160', 'MAC004916', 'MAC005485', 'MAC001966', 'MAC005447', 'MAC000666', 'MAC000344', 'MAC003827', 'MAC001944', 'MAC001823', 'MAC001766', 'MAC002281', 'MAC003897', 'MAC003379', 'MAC003211', 'MAC002222', 'MAC001667', 'MAC004637', 'MAC000793', 'MAC001760', 'MAC001715', 'MAC004643', 'MAC004973', 'MAC002240', 'MAC004652', 'MAC000046', 'MAC003334', 'MAC003769', 'MAC003973', 'MAC000773', 'MAC002867', 'MAC002842', 'MAC004927', 'MAC001960', 'MAC005450', 'MAC004992', 'MAC000754', 'MAC001950', 'MAC003546', 'MAC000692', 'MAC005555', 'MAC000676', 'MAC000787', 'MAC004989', 'MAC000406', 'MAC002268', 'MAC001948', 'MAC001806', 'MAC000768', 'MAC004907', 'MAC002873', 'MAC001739', 'MAC002151', 'MAC004999', 'MAC000408', 'MAC005104', 'MAC001830', 'MAC001735', 'MAC004638', 'MAC003532', 'MAC004998', 'MAC000641', 'MAC003259', 'MAC000776', 'MAC005460', 'MAC005123', 'MAC000703', 'MAC003170', 'MAC001959', 'MAC005004', 'MAC003512', 'MAC001825', 'MAC005562', 'MAC005454', 'MAC000665', 'MAC001768', 'MAC000464', 'MAC001773', 'MAC001738', 'MAC001718', 'MAC005001', 'MAC004644', 'MAC001751', 'MAC001693', 'MAC003783', 'MAC001556', 'MAC003522', 'MAC005431', 'MAC000763', 'MAC003257', 'MAC002238', 'MAC002852', 'MAC003892', 'MAC003919', 'MAC004912', 'MAC000681', 'MAC002230', 'MAC000109', 'MAC001554', 'MAC001777', 'MAC005502', 'MAC003788', 'MAC001761', 'MAC001965', 'MAC002277', 'MAC002850', 'MAC003869', 'MAC004904', 'MAC002125', 'MAC005504', 'MAC000675', 'MAC003507', 'MAC000708', 'MAC000014', 'MAC002123', 'MAC000716', 'MAC002870', 'MAC003347', 'MAC002139', 'MAC003508', 'MAC004647', 'MAC004913', 'MAC003515', 'MAC004666', 'MAC005092', 'MAC000447', 'MAC003865', 'MAC005119', 'MAC001750', 'MAC003178', 'MAC003535', 'MAC000760', 'MAC001744', 'MAC005475', 'MAC003358', 'MAC003816', 'MAC000461', 'MAC000052', 'MAC000369', 'MAC002346', 'MAC005096', 'MAC001769', 'MAC001542', 'MAC003372', 'MAC005437', 'MAC002225', 'MAC001973', 'MAC002138', 'MAC003771', 'MAC004908', 'MAC003501', 'MAC000739', 'MAC000762', 'MAC001956', 'MAC005471', 'MAC005491', 'MAC001650', 'MAC003916', 'MAC000678', 'MAC002860', 'MAC002264', 'MAC003808', 'MAC000682', 'MAC000689', 'MAC004654', 'MAC003829', 'MAC000352', 'MAC001978', 'MAC000782', 'MAC002849', 'MAC002159', 'MAC004673', 'MAC000698', 'MAC005125', 'MAC005476', 'MAC002324', 'MAC002162', 'MAC000336', 'MAC002878', 'MAC003514', 'MAC003552', 'MAC003337', 'MAC002236', 'MAC002883', 'MAC005459', 'MAC002336', 'MAC000758', 'MAC005098', 'MAC004911', 'MAC005497', 'MAC002846', 'MAC003538', 'MAC005131', 'MAC000764', 'MAC003975', 'MAC003333', 'MAC002856', 'MAC003165', 'MAC003201', 'MAC001746', 'MAC000342', 'MAC001564', 'MAC001770', 'MAC002880', 'MAC002163', 'MAC003272', 'MAC002265', 'MAC003331', 'MAC004656', 'MAC003556', 'MAC002882', 'MAC005117', 'MAC000694', 'MAC005455', 'MAC005457', 'MAC002334', 'MAC001612', 'MAC000771', 'MAC001943', 'MAC003221', 'MAC004931', 'MAC002280', 'MAC002862', 'MAC002877', 'MAC002851', 'MAC000792', 'MAC003269', 'MAC004630', 'MAC005484', 'MAC000017', 'MAC000653', 'MAC002854', 'MAC001855', 'MAC002155', 'MAC005005', 'MAC002840', 'MAC000784', 'MAC005103', 'MAC005132', 'MAC003551', 'MAC005481', 'MAC002246', 'MAC001955', 'MAC002157', 'MAC005472', 'MAC005466', 'MAC004915', 'MAC002136', 'MAC005453', 'MAC004934', 'MAC001636', 'MAC005487', 'MAC000795', 'MAC002344', 'MAC001700', 'MAC001849', 'MAC001974', 'MAC001968', 'MAC002127', 'MAC001765', 'MAC005128', 'MAC003180', 'MAC001586', 'MAC001941', 'MAC002150', 'MAC003286', 'MAC001686', 'MAC005451', 'MAC003277', 'MAC003519', 'MAC003504', 'MAC003914', 'MAC004922', 'MAC005118', 'MAC004660', 'MAC001680', 'MAC002130', 'MAC005120', 'MAC002144', 'MAC003862', 'MAC002149', 'MAC000366', 'MAC000421', 'MAC004651', 'MAC003521', 'MAC003509', 'MAC000064', 'MAC003899', 'MAC002129', 'MAC003960', 'MAC003982', 'MAC004987', 'MAC005126', 'MAC000107', 'MAC000672', 'MAC001772', 'MAC002858', 'MAC000635', 'MAC005433', 'MAC000456', 'MAC003988', 'MAC005446', 'MAC000770', 'MAC000075', 'MAC003520', 'MAC004626', 'MAC000778', 'MAC000418', 'MAC003202', 'MAC001972', 'MAC002152', 'MAC005090', 'MAC002226', 'MAC005440', 'MAC000711', 'MAC004925', 'MAC004978', 'MAC005088', 'MAC001767', 'MAC005442', 'MAC000686', 'MAC004929', 'MAC001707', 'MAC002133', 'MAC005503', 'MAC003262', 'MAC001749', 'MAC000738', 'MAC002120', 'MAC004975', 'MAC000709', 'MAC001851', 'MAC000714', 'MAC002291', 'MAC003518', 'MAC004649', 'MAC000071', 'MAC001736', 'MAC002853', 'MAC004671', 'MAC003308', 'MAC001741', 'MAC001977', 'MAC002145', 'MAC001734', 'MAC004933', 'MAC005109', 'MAC003285', 'MAC001840', 'MAC000392', 'MAC000684', 'MAC001853', 'MAC000728', 'MAC003911', 'MAC001763', 'MAC002843', 'MAC005436', 'MAC001961', 'MAC000005', 'MAC000696', 'MAC001626', 'MAC003245', 'MAC005479', 'MAC001642', 'MAC003534', 'MAC004645', 'MAC000769', 'MAC000015', 'MAC001762', 'MAC000781', 'MAC004629', 'MAC001660', 'MAC001742', 'MAC003890', 'MAC005130', 'MAC003175', 'MAC004974', 'MAC004976', 'MAC004627', 'MAC001833', 'MAC004924', 'MAC000697', 'MAC002137', 'MAC002128', 'MAC001704', 'MAC001946', 'MAC004636', 'MAC004918', 'MAC000777', 'MAC001980', 'MAC004642', 'MAC005115', 'MAC003967', 'MAC001610', 'MAC005099', 'MAC001967', 'MAC000687', 'MAC001688', 'MAC003984', 'MAC005000', 'MAC003506', 'MAC002217', 'MAC001747', 'MAC002857', 'MAC004632', 'MAC001771', 'MAC000705', 'MAC000685', 'MAC005480', 'MAC000780', 'MAC001835', 'MAC000139', 'MAC005452', 'MAC003547', 'MAC004909', 'MAC000031', 'MAC004984', 'MAC001748', 'MAC002871', 'MAC003537', 'MAC000688', 'MAC000433', 'MAC005095', 'MAC004658', 'MAC002865', 'MAC003989', 'MAC002135', 'MAC001658', 'MAC002863', 'MAC003543', 'MAC005468', 'MAC001745', 'MAC004920', 'MAC000759', 'MAC005490', 'MAC000124', 'MAC001596', 'MAC004981', 'MAC005566', 'MAC001733', 'MAC001753', 'MAC004628', 'MAC003365', 'MAC004986', 'MAC004996', 'MAC001940', 'MAC005483', 'MAC005425', 'MAC003947', 'MAC000303', 'MAC005114', 'MAC001597', 'MAC001712', 'MAC001559', 'MAC002142', 'MAC003554', 'MAC001731', 'MAC005486', 'MAC002243', 'MAC005108', 'MAC005435', 'MAC001732', 'MAC001752', 'MAC001756', 'MAC002318', 'MAC001847', 'MAC005087', 'MAC005428', 'MAC004995', 'MAC001538', 'MAC005105', 'MAC000076', 'MAC002117', 'MAC003265', 'MAC005458', 'MAC000704', 'MAC004648', 'MAC004930', 'MAC002161', 'MAC005463', 'MAC003820', 'MAC004980', 'MAC004985', 'MAC005478', 'MAC000691', 'MAC001962', 'MAC005002', 'MAC001759', 'MAC005100', 'MAC003529', 'MAC000465', 'MAC003274', 'MAC005493', 'MAC000673', 'MAC001590', 'MAC003549', 'MAC000719', 'MAC001971', 'MAC001730', 'MAC002844', 'MAC002859', 'MAC001757', 'MAC002158', 'MAC003500', 'MAC003196', 'MAC000146', 'MAC004982', 'MAC000439', 'MAC003903', 'MAC004935', 'MAC000088', 'MAC000395', 'MAC005426'}
    all_LCL = glob.glob(str(dataset_dir / 'Small\ LCL\ Data' 'LCL-June2015v2*.csv'))

    (dataset_dir / 'LCL').mkdir(parents=True, exist_ok=True)
    random.shuffle(all_LCL)
    num_bldgs = 0
    years = [2012, 2013]
    for lcl_file in all_LCL:
        df_ = pd.read_csv(lcl_file)
        unique_bldgs = df_['LCLid'].unique()
        # only keep buildings that are in all_buildings
        unique_bldgs = all_buildings.intersection(set(unique_bldgs))
        # remove buildings from all_buildings that are in unique_bldgs
        all_buildings = all_buildings.difference(unique_bldgs)

        for ub in unique_bldgs:
            df = df_[df_.LCLid == ub]
            df.DateTime = pd.to_datetime(df.DateTime, format='%Y-%m-%d %H:%M:%S')
            # Set timestamp as index
            df = df.set_index('DateTime')
            df = df[~df.index.duplicated(keep='first')]
            df = df.asfreq('30min')
            df = df.rename(columns={'KWH/hh (per half hour) ': 'power'})
            # only keep power column
            df = df[['power']]
            df = df.replace('Null', np.nan)
            df = df.astype('float32')
            df = df.resample(rule='H', closed='left', label='right').mean()
             
            for year in years:
                dfy = df[df.index.year == year]
                if dfy.shape[0] < 168:
                    continue
                missing_frac = dfy['power'].isnull().sum() / dfy.shape[0]
                if missing_frac <= 0.1:
                    dfy = dfy.interpolate(method='linear', axis=0, limit=24*7, limit_direction='both')
                    dfy = dfy.fillna(0.)
                    dfy.to_csv(save_path / f'{ub}_clean={year}.csv')
                    num_bldgs += 1
            if num_bldgs == 713:
                break
            print(f'num bldgs: {num_bldgs}') 
        if num_bldgs == 713:
            break