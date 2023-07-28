area_coor = {
    'edinburg' : [55.96, -3.19, 55.95, -3.18],
    'massachusetts' : [42.41, -71.39, 42.4, -71.38],
    'portugal': [42, -9.5, 37, -6],
    'london' : [51.51, -0.12, 51.5,-0.11],
    'sceaux' : [48.78, 2.29, 48.77, 2.3],
    'waterloo' : [43.47, -80.52, 43.46, -80.51]
}

def get_hourly_weather_pandas(start_date: str, end_date: str, area: list, variable=['2m_temperature'], download_path=None):
    import xarray as xr
    from urllib.request import urlopen
    import cdsapi
    import pandas as pd
    
    # start the client
    cds = cdsapi.Client()
    # dataset you want to read
    dataset = 'reanalysis-era5-single-levels'
    
    params = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': variable,
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
        'date': f'{start_date}/{end_date}',
        'area': area
    }
    # retrieves the path to the file
    fl = cds.retrieve(dataset, params)
    
    if download_path != None:
        fn = '_'.join(start_date + end_date + variable) + '_raw.nc'
        fl.download(download_path + '/' + fn)
        print(f"Saved raw file: {download_path + '/' + fn}")
            
    # load into memory
    with urlopen(fl.location) as f:
        ds = xr.open_dataset(f.read())
    
    # average values across latitude and longitude
    mean_ds = ds.mean(dim=['latitude', 'longitude'])
    
    # to pandas dataframe
    df = mean_ds.to_pandas()
    df.columns = variable
    
    return df
