# -*- coding: utf-8 -*-

import os
os.chdir(r'...')

import xarray as xr
import numpy as np
import datetime
import random
import glob


def get_rolling_time_index(doy, time, days, windows_size):
    window_days = np.array([(doy - i) % days for i in range(- windows_size//2+ windows_size%2, windows_size//2 + windows_size%2)])
    window_days[window_days == 0] = 365
    
    time_index = np.nonzero(np.isin(time, window_days))[0]
    return time_index




def calculate_clim_mean_by_doy(window_data):
    """
    Process a single day of the year.
    """    
    # e.g. Get the window of 7 and 8 days before and after the current dayofyear, a total of 15 days
    #window_days = np.array([(doy - i) % days for i in range(- windows_size//2+ windows_size%2, windows_size//2 + windows_size%2)])
    #window_days[window_days == 0] = 365
    
    #time_index = np.nonzero(np.isin(time, window_days))[0]
    #window_data = daily_data[:, :, time_index]  
     
    clim_mean = np.mean(window_data.values, axis=2)        
    return clim_mean




def calculate_percentiles_by_doy(doy, window_data, remove_seasonal_cycle=False, clim_mean=None, valid_mask_for_rsc = None):
    """
    Process a single day of the year.
    """    
    # e.g. Get the window of 7-8 days before and after the current dayofyear, a total of 15 days
    #window_days = np.array([(doy - i) % days for i in range(- windows_size//2+ windows_size%2, windows_size//2 + windows_size%2)])
    #window_days[window_days == 0] = 365
    #time_index = np.nonzero(np.isin(time, window_days))[0]
    #window_data = daily_data[:, :, time_index]  
    
    # 10th percentile
    if remove_seasonal_cycle:        
        anomalies = window_data - clim_mean.sel(time=window_data.time)        
        if valid_mask_for_rsc is not None:
            percentiles = np.percentile(window_data.values, 10, axis=2) 
            percentiles[valid_mask_for_rsc] = np.percentile(anomalies.values[valid_mask_for_rsc,:], 10, axis=1) + clim_mean.sel(time=doy).values[valid_mask_for_rsc]
        else:
            percentiles = np.percentile(anomalies.values, 10, axis=2) + clim_mean.sel(time=doy).values
    else:               
        percentiles = np.percentile(window_data.values, 10, axis=2)        
    return percentiles




def compute_bias(power, data, threshold, f_exp):
    """f'(p, w)"""    
    data = data.transpose("time", "lat", "lon")
    data = data.sel(time = data.time<=365)  
    threshold = threshold.sel(time = data.time)
    if power == 'pv':
        # Eliminate polar nights in high-latitude regions. When the threshold is less than or equal to 0, it will not be included in the statistics.  
        valid_mask = threshold > 0
        f_obs = ((data < threshold)  & valid_mask).sum("time") / valid_mask.sum("time")
        
    if power == 'wind':
        # A threshold of less than 0 is usually due to bias/error introduced by strong variability / seasonal variability
        valid_mask = threshold != 0
        f_obs = ((data < threshold)  & valid_mask).sum("time") / valid_mask.sum("time")
    return ((f_obs - f_exp) / f_exp) 
    

    
    
def compute_elp_days(data, threshold):
    """f'(p, w)"""
    data = data.transpose("time", "lat", "lon")
    
    data = data.sel(time = data.time<=365)    
    threshold = threshold.sel(time = data.time)
    
    # Eliminate polar nights in high-latitude regions. When the threshold is less than or equal to 0, it will not be included in the statistics.   
    valid_mask = threshold > 0
    f_obs = ((data < threshold)  & valid_mask).sum("time")    
    return f_obs
    
    
    
    
def generate_resampled_datasets(time_coords):    
    resampled_dates = []
    years = np.arange(1985, 2015)
    random.seed(42)
    for year_to_remove in years:
        # remove current year
        remaining_years = [y for y in years if y != year_to_remove]
        year_dates = time_coords[time_coords.dt.year.isin(remaining_years)]
        
        # select one of the remaining years for supplementation
        for supp_year in remaining_years:        
            modified_year_dates = xr.concat([year_dates, time_coords[time_coords.dt.year == supp_year]], dim='time')        
            resampled_dates.append((year_to_remove, modified_year_dates))
    
    return resampled_dates     
    
    
    
    

def get_thres(power, climate_model, windows_size, baseline = False, bias = False, remove_seasonal_cycle=False, valid_mask_for_rsc = None):
    """
    Calculate thresholds for a given climate model.
    
    Parameters:
    -----------
    climate_model : str
        The name of the climate model.
    window_size : int
        The size of the sliding window (default is 15 days).
    """
    # Set path
    if power == 'wind':
        file_path = r"./daily_power_production/wind/land_and_ocean"
        
    if power == 'pv': 
        file_path = r"./daily_power_production/pv"
    
    # Set number of days in a year based on the climate model
    days = 360 if climate_model in ['KACE-1-0-G', 'UKESM1-0-LL'] else 365
    var_name = f"{power}_power_production" 
    

    daily_data = []
    t = datetime.datetime.now()
    for year in range(1985, 2015):
        file_name = f'global_{power}_power_historical_{year}_{climate_model}.nc'
        data = xr.open_dataset(os.path.join(file_path, file_name))    
        daily_data.append(data)
    
    daily_data = xr.concat(daily_data, dim='time')
    
    
    lats = daily_data.lat.values
    lons = daily_data.lon.values   
    print('Finish: concat', datetime.datetime.now() - t)
    
    
    if baseline:
        resample_years = generate_resampled_datasets(daily_data.time)   
    else:
        resample_years = [[0, np.arange(1985, 2015)]]
    
    # Create a list of delayed tasks    
    for r in range(len(resample_years)):
        year_to_remove = resample_years[r][0]
        resampled_date = resample_years[r][1]
        resampled_data = daily_data.sel(time=resampled_date)
        resampled_data['time'] = resampled_data['time'].dt.dayofyear
        
        for ws in windows_size:
            # Save the thresholds
            save_path = r'./ELP_event/threshold'

            if baseline:
                folder = 'threshold_for_baseline'
                if remove_seasonal_cycle:
                    file_name = f'{power}_threshold_{climate_model}_ws{ws}_r{year_to_remove}_remove_seasonal_cycle{r}.nc'
                    file_name_clim_mean = f'{power}_clim_ave_{climate_model}_ws{ws}_r{year_to_remove}_remove_seasonal_cycle{r}.nc'
                    if bias:
                        file_name_bias = f'{power}_bias_{climate_model}_ws{ws}_r{year_to_remove}_remove_seasonal_cycle{r}.nc'
                    
                else:
                    file_name = f'{power}_threshold_{climate_model}_ws{ws}_r{year_to_remove}_{r}.nc'
                    file_name_clim_mean = f'{power}_clim_ave_{climate_model}_ws{ws}_r{year_to_remove}_{r}.nc'
                    if bias:
                        file_name_bias = f'{power}_bias_{climate_model}_ws{ws}_r{year_to_remove}_{r}.nc'
                                    
            else:
                folder = 'threshold_for_future'
                if remove_seasonal_cycle:
                    file_name = f'{power}_threshold_{climate_model}_ws{ws}_remove_seasonal_cycle.nc'
                    file_name_clim_mean = f'{power}_clim_ave_{climate_model}_ws{ws}_remove_seasonal_cycle.nc'
                    if bias:
                        file_name_bias = f'{power}_bias_{climate_model}_ws{ws}_remove_seasonal_cycle.nc'
                    
                else:
                    file_name = f'{power}_threshold_{climate_model}_ws{ws}.nc'
                    file_name_clim_mean = f'{power}_clim_ave_{climate_model}_ws{ws}.nc'
                    if bias:
                        file_name_bias = f'{power}_bias_{climate_model}_ws{ws}.nc'  
                        
                        
            if os.path.exists(os.path.join(save_path, folder, file_name)):
                continue
                

            time_index = [get_rolling_time_index(doy, resampled_data['time'].values, days, ws) for doy in range(1, days+1)]
            
            if power == 'pv':
                time_index_c = [get_rolling_time_index(doy, resampled_data['time'].values, days, 1) for doy in range(1, days+1)]
                clim_mean = [calculate_clim_mean_by_doy(resampled_data[var_name][:,:,time_index_c[doy-1]]) for doy in range(1, days+1)]
                
            if power == 'wind':
                clim_mean = [calculate_clim_mean_by_doy(resampled_data[var_name][:,:,time_index[doy-1]]) for doy in range(1, days+1)]

            clim_mean = xr.DataArray(np.array(clim_mean, dtype=np.float32),
                                      dims = ['time', 'lat', 'lon'],
                                      name = 'clim_ave_1985-2014',
                                      coords={
                                              'time': range(1, days+1),
                                              'lat': lats,
                                              'lon': lons}
                                      )            
            print(f'finished clim_ave: {datetime.datetime.now() - t}')
            
            if remove_seasonal_cycle:
                thresholds = [calculate_percentiles_by_doy(doy, resampled_data[var_name][:,:,time_index[doy-1]], remove_seasonal_cycle, clim_mean, valid_mask_for_rsc) for doy in range(1, days+1)]

            else:
                thresholds = [calculate_percentiles_by_doy(doy, resampled_data[var_name][:,:,time_index[doy-1]]) for doy in range(1, days+1)]



            # Initialize the thresholds DataArray
            thresholds = xr.DataArray(np.array(thresholds, dtype=np.float32),
                                      dims = ['time', 'lat', 'lon'],
                                      name = 'threshold_10th_1985-2014',
                                      coords={
                                              'time': range(1, days+1),
                                              'lat': lats,
                                              'lon': lons}
                                      )    
            
            if bias:
                bias = compute_bias(power, resampled_data[var_name], thresholds, 0.1)    


                # Initialize the bias DataArray
                bias = xr.DataArray(np.array(bias, dtype=np.float32),
                                          dims = ['lat', 'lon'],
                                          name = 'bias',
                                          coords={
                                                  'lat': lats,
                                                  'lon': lons}
                                          )


            print(f'finished in {datetime.datetime.now() - t}')

       
 
            clim_mean.to_netcdf(os.path.join(save_path, folder, file_name_clim_mean),
                               encoding={'clim_ave_1985-2014': {'zlib': True, 'complevel': 6}})    

            thresholds.to_netcdf(os.path.join(save_path, folder, file_name),
                                 encoding={'threshold_10th_1985-2014': {'zlib': True, 'complevel': 6}})
            if bias:
                bias.to_netcdf(os.path.join(save_path, folder, file_name_bias),
                               encoding={'bias': {'zlib': True, 'complevel': 6}})    
     



def average_r_files(power, climate_model, windows_size, years_to_remove, baseline=True, bias = False, remove_seasonal_cycle=False):
    if baseline:        
        save_path='./ELP_event/threshold'
        folder = 'threshold_for_baseline'
        
        target_folder = os.path.join(save_path, folder)
    
        def get_pattern(filetype, ws, year_to_remove):
            if remove_seasonal_cycle:
                return f'{power}_{filetype}_{climate_model}_ws{ws}_r{year_to_remove}_remove_seasonal_cycle*.nc'
            else:
                return f'{power}_{filetype}_{climate_model}_ws{ws}_r{year_to_remove}_*.nc'
    
        def process_filetype(filetype, ws, year_to_remove):
            pattern = get_pattern(filetype, ws)
            file_list = sorted(glob.glob(os.path.join(target_folder, pattern)))
        
            # read files
            datasets = [xr.open_dataset(f) for f in file_list]
            combined = xr.concat(datasets, dim='r')
            mean_ds = combined.mean(dim='r')
    
            if remove_seasonal_cycle:
                output_name = f'{power}_{filetype}_{climate_model}_ws{ws}_r{year_to_remove}_remove_seasonal_cycle.nc'
            else:
                output_name = f'{power}_{filetype}_{climate_model}_ws{ws}_r{year_to_remove}.nc'
            output_path = os.path.join(target_folder, output_name)
            mean_ds.to_netcdf(output_path)
    
    
        # process
        for y in years_to_remove:
            for ws in windows_size:
                process_filetype('threshold', ws, y)
                process_filetype('clim_ave', ws, y)
                if bias:
                    process_filetype('bias', ws, y)
    
     
    
    
    
    
def get_elp_days(power, climate_model, scenario, years, windows_size, remove_seasonal_cycle=False):
    # Set path
    if power == 'wind':
        file_path = r"./daily_power_production/wind/land_and_ocean"
        
    if power == 'pv': 
        file_path = r"./daily_power_production/pv"
    
    # Set number of days in a year based on the climate model
    # days = 360 if climate_model in ['KACE-1-0-G', 'UKESM1-0-LL'] else 365
    
    
    path = r'./ELP_event/threshold'    
    for ws in windows_size:
        
        period_data = []
        for year in years:
            if scenario == 'historical':
                folder = 'threshold_for_baseline'
                
                if remove_seasonal_cycle:
                    file_name = f'{power}_threshold_{climate_model}_ws{ws}_r{year}_remove_seasonal_cycle.nc'
                    
                else:
                    file_name = f'{power}_threshold_{climate_model}_ws{ws}_r{year}.nc'            
                
            else:
                folder = 'threshold_for_future'           
            
                if remove_seasonal_cycle:
                    file_name = f'{power}_threshold_{climate_model}_ws{ws}_remove_seasonal_cycle.nc'
                    
                else:
                    file_name = f'{power}_threshold_{climate_model}_ws{ws}.nc'
                
            thresholds = xr.open_dataset(os.path.join(path, folder, file_name))['threshold_10th_1985-2014']   
            
            # get power data
            file_name = f'global_{power}_power_{scenario}_{year}_{climate_model}.nc'
            data = xr.open_dataset(os.path.join(file_path, file_name))        
            data['time'] = data['time'].dt.dayofyear
            lats = data.lat.values
            lons = data.lon.values            

            var_name = f"{power}_power_production"       
            elp_days = compute_elp_days(data[var_name], thresholds)             
            elp_days = xr.DataArray(np.array(elp_days, dtype=np.float32),
                                              dims = ['lat', 'lon'],
                                              name = 'total_elp_days',
                                              coords={
                                                      'lat': lats,
                                                      'lon': lons}
                                              )
            period_data.append(elp_days)  
            
        period_data = xr.concat(period_data, dim='time').sum(dim='time')

        # save  
        save_path = r'./ELP_event/threshold'
        period = str(years.min()) + '_' + str(years.max())
        if remove_seasonal_cycle:
            file_name = f'{power}_elp_days_{climate_model}_ws{ws}_{period}_remove_seasonal_cycle.nc'
        else:
            file_name = f'{power}_elp_days_{climate_model}_ws{ws}_{period}.nc'


        period_data.to_netcdf(os.path.join(save_path, file_name),
                           encoding={'total_elp_days': {'zlib': True, 'complevel': 6}})    
                    
    
    
    
# ----- sensitivity analysis 
# power = 'wind'
# climate_model = 'ACCESS-ESM1-5'
# get_thres(power, climate_model, windows_size= [5,10,15,31], bias = True)
# get_thres(power, climate_model, windows_size= [5,10,15,31], bias = True, remove_seasonal_cycle=True)

# average_r_files(power, climate_model, windows_size= [5,10,15,31], np.arange(1985,2015), baseline=True, bias = True)
# average_r_files(power, climate_model, windows_size= [5,10,15,31], np.arange(1985,2015), baseline=True, bias = True, remove_seasonal_cycle=True)

# get_elp_days(power, climate_model, 'historical', np.arange(1985,2015), windows_size= [5,10,15,31])
# get_elp_days(power, climate_model, 'ssp370', np.arange(2071,2101), windows_size= [5,10,15,31])
# get_elp_days(power, climate_model, 'historical', np.arange(1985,2015), windows_size= [5,10,15,31], remove_seasonal_cycle=True)
# get_elp_days(power, climate_model, 'ssp370', np.arange(2071,2101), windows_size= [5,10,15,31], remove_seasonal_cycle=True)


# ----- calculate threshold
# power = 'wind' # or 'pv'
# climate_models = [
#                   'ACCESS-ESM1-5',
#                     'BCC-CSM2-MR',
#                     'CanESM5', 
#                     'CMCC-ESM2',
#                     'EC-Earth3', 
#                     'FGOALS-g3', 
#                     'GFDL-ESM4', 
#                     'INM-CM4-8', 
#                   'IPSL-CM6A-LR', 
#                     'KACE-1-0-G', 
#                     'MIROC6',
#                   'CNRM-ESM2-1',
#                       'GISS-E2-1-G', 
#                     'MPI-ESM1-2-HR',
#                     'UKESM1-0-LL', 
#                       'MRI-ESM2-0', 
#                       'NorESM2-MM'
#                  ]
# import gc
# for climate_model in climate_models:
#     if power == 'wind':
#         valid_mask = xr.open_dataset(r'./ELP_event/threshold/threshold_for_future/valid_mask_for_remove_seasonality_for_wind_power.nc')['valid_mask'].values == 1
#         get_thres(power, climate_model, windows_size= [15], baseline = True, bias = False, remove_seasonal_cycle=True, valid_mask_for_rsc = valid_mask)
#         average_r_files(power, climate_model, windows_size= [15], np.arange(1985,2015), baseline=True, bias = False, remove_seasonal_cycle=True)
#
#     else:
#         get_thres(power, climate_model, windows_size= [15], baseline = True, bias = False, remove_seasonal_cycle=True)
#         average_r_files(power, climate_model, windows_size= [15], np.arange(1985,2015), baseline=True, bias = False, remove_seasonal_cycle=True)
#     gc.collect()
    