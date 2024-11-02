import os

# same process for PV
os.chdir(r'I:\global_wind_future')

import numpy as np
import xarray as xr
import datetime




def calculate_percentiles(daily_data, doy, block, days):
    # Define the 15-day sliding window
    window_days = [(doy - i) % days + 1 for i in range(-7, 8)]
    window_data = daily_data.sel(time=daily_data['time'].isin(window_days))['wind_energy_density'].values
    
    if block == 0:
        p1 = np.percentile(window_data, 1, axis=2).copy()
        p5 = np.percentile(window_data, 5, axis=2).copy()
        p10 = np.percentile(window_data, 10, axis=2).copy()
        del window_data
        return (doy, block, p1, p5, p10)
    
    elif block == 1:
        p1 = np.percentile(window_data, 1, axis=2).copy()
        p5 = np.percentile(window_data, 5, axis=2).copy()
        p10 = np.percentile(window_data, 10, axis=2).copy()
        del window_data
        return (doy, block, p1, p5, p10)





def get_thres(climate_model):
    #
    # set path    
    file_path = os.path.join(r'I:\global_wind_future\daily_map\land_and_ocean', climate_model) 
    
    lats = np.arange(90, -60, -0.25) - 0.25/2
    lons = np.arange(-180, 180, 0.25) + 0.25/2
    
    # Set number of days in a year based on the climate model
    days = 360 if climate_model in ['KACE-1-0-G', 'UKESM1-0-LL'] else 365
    
    # Initialize the thresholds DataArray
    # ave = xr.DataArray(np.zeros((days, 600, 1440),dtype=np.float32),
    #                             dims = ['doy', 'lat', 'lon'],
    #                             coords={'doy': range(1, days+1), 
    #                                     'lat': lats, 
    #                                     'lon': lons })  
    
    thresholds = xr.DataArray(np.zeros((3, days, 600, 1440),dtype=np.float32),
                                dims = ['thrs', 'doy', 'lat', 'lon'],
                                coords={'thrs':['1th', '5th', '10th'], 
                                        'doy': range(1, days+1), 
                                        'lat': lats, 
                                        'lon': lons })  
    
    # Block to relieve memory pressure
    for block in range(2):
        
        daily_data = []
        t = datetime.datetime.now()
        
        for year in range(1985,2015):
            file_name = 'global_wind_power_historical_' + str(year) + '_' + climate_model + '.nc'
            data = xr.open_dataset( os.path.join(file_path, file_name) )
            
            if block == 0:
                data = data.sel(x=slice(-180,0))
            if block == 1:
                data = data.sel(x=slice(0,180))

            data['time'] = data['time'].dt.dayofyear
            daily_data.append(data)
            
        daily_data = xr.concat(daily_data, dim='time')      
        
        time = daily_data['time'].values
        daily_data = daily_data['wind_energy_density'].values
        
        print('finish: concat', datetime.datetime.now() - t)   
 
        
        for doy in range(1, days+1):
            
            t = datetime.datetime.now()
            #
            # Get the window of 7-8 days before and after the current dayofyear, a total of 15 days
            window_days = [(doy - i) % days +1 for i in range(-7, 8)]             
            time_index = np.nonzero(np.isin(time, window_days))[0]
            window_data = daily_data[:, :, time_index]

            
            if block == 0:
                # ave[doy-1, :, :720] = np.mean(window_data, axis = 2)
                thresholds[:, doy-1, :, :720] = np.percentile(window_data, [1, 5, 10], axis = 2)
                print(doy, datetime.datetime.now() - t)
                
            if block == 1:
                # ave[doy-1, :, 720:] = np.mean(window_data, axis = 2)
                thresholds[:, doy-1, :, 720:] = np.percentile(window_data, [1, 5, 10], axis = 2)
                print(doy, datetime.datetime.now() - t)   
                
        del window_data, daily_data
        
    # save_path = r'I:\global_wind_future\low_output_by_percentile\threshold'                
    # file_name =  f'ave_{climate_model}.nc'
    # ave.to_netcdf( os.path.join(save_path, file_name)
    #                      ,encoding={'__xarray_dataarray_variable__': {'zlib': True, 'complevel': 6}}) 
                
    save_path = r'I:\global_wind_future\low_output_by_percentile\threshold_1985_2014'                
    file_name =  f'threshold_{climate_model}.nc'
    thresholds.to_netcdf( os.path.join(save_path, file_name)
                          ,encoding={'__xarray_dataarray_variable__': {'zlib': True, 'complevel': 6}}) 



 
def main():
    climate_models = ['ACCESS-ESM1-5','BCC-CSM2-MR', 
                      'CanESM5','CMCC-ESM2','EC-Earth3',
                      'FGOALS-g3', 'GFDL-ESM4', 'INM-CM4-8', 
                      'IPSL-CM6A-LR', 'MIROC6',
                      'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM', 'UKESM1-0-LL' ,
                      #'CNRM-ESM2-1','GISS-E2-1-G',
                      'KACE-1-0-G']
    
    for climate_model in climate_models:
        main(climate_model)



        