# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import datetime
import xarray as xr
import warnings
warnings.filterwarnings('ignore')




# In[extreme low output]

def renumber_continuous(nums):
    # Generate event number
    nums = np.array(nums)
    diffs = np.diff(nums)
    groups = np.cumsum( np.concatenate(([1], diffs != 1)) )
    return groups




from joblib import Parallel, delayed
def process_chunk(chunk):
    
    group_keys = ['event_ID', 'lon_index', 'lat_index']
    
    # duration
    du = chunk.groupby(group_keys).size().reset_index(name='duration_days')
    
    # intensity
    d_power = chunk.groupby(group_keys)[['power', 'ave', 'thrs_10th', 'thrs_5th', 'thrs_1th']].sum().reset_index()
    
    # combine
    result = pd.merge(du, d_power, on=group_keys)
    return result




def main(climate_model, scenario, year, threshold, threshold1, threshold2, aves, remove_seasonal_no_power = True):
    # Generate the event information table
    #
    # define save path and name
    save_path = os.path.join(r'I:\global_wind_future\low_output_by_percentile\extreme_low_output', climate_model) 
    save_name = 'event_' +  scenario + '_' + str(year) + '_' + climate_model + '.npy'

    # read daily power generation curve  
    file_path = os.path.join(r'I:\global_wind_future\daily_map\land_and_ocean', climate_model) #land_and_ocean
    file_name = 'global_wind_power_' + scenario + '_' + str(year) + '_' + climate_model + '.nc'

    daily_data = xr.open_dataset( os.path.join(file_path, file_name) )['wind_energy_density']
    days = len(daily_data['time'])
    lat_length, lon_length = len(daily_data.y), len(daily_data.x)
    
    power = daily_data.values
    if days == 366:
        power = power[:,:,~((daily_data['time'].dt.month==2) & (daily_data['time'].dt.day==29))]
        days = 365
    
 
    # Generate time, latitude and longitude indexes
    doy = np.arange(0, days, 1)
    lat = np.arange(0, lat_length, 1)
    lon = np.arange(0, lon_length, 1)
    lons, lats, doys  = np.meshgrid( lon, lat, doy)         
   

    judge = np.where((power<threshold)|(power<0), 1, 0)
    lon_index = lons[judge==1]
    lat_index = lats[judge==1]
    doy_index = doys[judge==1]
    power_event = power[judge==1]
    
    ave = aves[judge==1]
    thr = threshold[judge==1]
    thr1 = threshold1[judge==1]
    thr2 = threshold2[judge==1]
    
    
    df = np.concatenate((lon_index.reshape(-1,1), lat_index.reshape(-1,1), ave.reshape(-1,1),\
                         thr.reshape(-1,1), thr1.reshape(-1,1), thr2.reshape(-1,1),\
                              doy_index.reshape(-1,1), power_event.reshape(-1,1)), axis=1, dtype=np.float16) 
     
    df = pd.DataFrame(df, columns = ['lon_index','lat_index', 'ave',
                                          'thrs_10th','thrs_5th','thrs_1th',
                                          'doy','power'])   
    del lon_index, lat_index, ave, thr, thr1, thr2, doy_index, power_event, threshold, threshold1, threshold2, aves, doys
    
 
    #    
    # if remove_seasonal_no_power:
    # Data with 10 quantile < 0 is deleted        
    #     df = df[df['thrs_10th']>0]
    #     df.reset_index(drop = True, inplace = True)     

    df['power'][df['power']<0] = 0 
     
    #为事件添加编号
    df = df.sort_values(by=['lon_index', 'lat_index']).reset_index(drop=True)
    df['event_ID'] = renumber_continuous(df['doy'])       


    chunk_size = 2000000
    
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]
    
    # Each block is processed in parallel
    results = Parallel(n_jobs=-1)(delayed(process_chunk)(chunk) for chunk in chunks)
    
    # Result of merging all blocks
    extreme_event = pd.concat(results, ignore_index=True)
    extreme_event = extreme_event[extreme_event['duration_days']<360].reset_index(drop=True)
    print(extreme_event['duration_days'].max())
    del df        
   

    if not os.path.exists(save_path):
        os.makedirs(save_path)
            
    save_name = 'event_' +  scenario + '_' + str(year) + '_' + climate_model + '.npy'
    

    # np.float16 will be failed to save the event ID, but will save memory  
    df = np.array(extreme_event, dtype=np.float16)
    if np.isinf(np.max(df[:,1:])):
        df = np.array(extreme_event, dtype=np.float32)
        
    np.save(os.path.join(save_path, save_name), df )
    del extreme_event, df
        
        

            

#
# main 
#



scenarios = ['historical','ssp126', 'ssp245', 'ssp370']
start_years = [1985, 2015, 2015, 2015]
end_years = [1985, 2100, 2100, 2100]


climate_models = ['ACCESS-ESM1-5','BCC-CSM2-MR', 
                  'CanESM5','CMCC-ESM2','EC-Earth3',
                  'FGOALS-g3', 'GFDL-ESM4', 'INM-CM4-8', 
                  'IPSL-CM6A-LR',  'MIROC6',
                  #'CNRM-ESM2-1','GISS-E2-1-G', 'KACE-1-0-G',
                  'MPI-ESM1-2-HR', 'UKESM1-0-LL',
                  'MRI-ESM2-0', 'NorESM2-MM'] 


for c in range(0, len(climate_models)):
    
    climate_model = climate_models[c]
    thrs_name =  r'I:\global_wind_future\low_output_by_percentile\threshold_1985_2014\threshold_' + climate_model + '.nc'
    thresholds = xr.open_dataset(thrs_name)
    
    threshold = thresholds.sel(thrs='10th')['__xarray_dataarray_variable__'].values
    threshold1 = thresholds.sel(thrs='5th')['__xarray_dataarray_variable__'].values
    threshold2 = thresholds.sel(thrs='1th')['__xarray_dataarray_variable__'].values  
    
    threshold = np.transpose(threshold, (1, 2, 0))
    threshold1 = np.transpose(threshold1, (1, 2, 0))
    threshold2 = np.transpose(threshold2, (1, 2, 0))
    
    
    ave_name =  r'I:\global_wind_future\low_output_by_percentile\threshold\ave_' + climate_model + '.nc'
    aves = xr.open_dataset(ave_name)
    aves = aves['__xarray_dataarray_variable__'].values
    aves = np.transpose(aves, (1, 2, 0))

    for s in range(0,len(scenarios)):
        scenario = scenarios[s]
        start_year = start_years[s]
        end_year = end_years[s]
        time_range = np.arange(start_year,end_year+1)  
        
        for year in time_range:
            t = datetime.datetime.now()
            main(climate_model, scenario, year, threshold, threshold1, threshold2, aves)
            print(climate_model, scenario, year, datetime.datetime.now()-t)
    del threshold, threshold1, threshold2, aves
            



# In[annual low_output map]


def main(climate_model, scenario, year, ave):
    
    save_path = os.path.join(r'I:\global_wind_future\low_output_by_percentile\annual_low_output', climate_model) 
    save_name = 'low_output_' +  scenario + '_' + str(year) + '_' + climate_model + '.nc'
  
    if not os.path.exists(os.path.join(save_path)):
        os.makedirs(save_path)

    if os.path.exists(os.path.join(save_path, save_name)):
        print('file exists')


    file_path = os.path.join(r'I:\global_wind_future\daily_map\land_and_ocean', climate_model) 
    file_name = 'global_wind_power_' + scenario + '_' + str(year) + '_' + climate_model + '.nc'
   
    daily_data = xr.open_dataset( os.path.join(file_path, file_name) )
    daily_data = daily_data.transpose('time', 'y', 'x')['wind_energy_density']
    
    lons = daily_data.x
    lats = daily_data.y
    time = daily_data['time']
   
    power = daily_data.values
    if len(time) == 366:
        power = power[~((daily_data['time'].dt.month==2) & (daily_data['time'].dt.day==29)),:,:]


    power = ave - power
    power[power<=0] = 0
    power = power.sum(0)

    annual_low_output = xr.Dataset( {'annual_low_output': (["y", "x"], power)},     
                                    coords = {'y': lats,
                                              'x': lons} )

    annual_low_output.to_netcdf( os.path.join(save_path, save_name)
                                , encoding={'annual_low_output': {'zlib': True, 'complevel': 6}})         



#
# main 
#

scenarios = ['historical','ssp126', 'ssp245', 'ssp370']
start_years = [1985, 2015, 2015, 2015]
end_years = [1985, 2100, 2100, 2100]

climate_models = ['ACCESS-ESM1-5','BCC-CSM2-MR', 
                  'CanESM5','CMCC-ESM2','EC-Earth3',
                  'FGOALS-g3', 'GFDL-ESM4', 'INM-CM4-8', 
                  'IPSL-CM6A-LR', 'MIROC6',
                  #'CNRM-ESM2-1','GISS-E2-1-G', 'KACE-1-0-G', 
                  'MPI-ESM1-2-HR',
                  'MRI-ESM2-0', 'NorESM2-MM', 'UKESM1-0-LL'] 


for c in range(0, len(climate_models)):
    
    climate_model = climate_models[c]
    ave_name =  r'I:\global_wind_future\low_output_by_percentile\threshold\ave_' + climate_model + '.nc'
    ave = xr.open_dataset(ave_name)
    ave = ave['__xarray_dataarray_variable__'].values
    

    for s in range(0, len(scenarios)):
        scenario = scenarios[s]
        start_year = start_years[s]
        end_year = end_years[s]
        time_range = np.arange(start_year,end_year+1)  
        
        for year in time_range:
            t = datetime.datetime.now()
            main(climate_model, scenario, year, ave)
            print(climate_model, scenario, year, datetime.datetime.now()-t)
            


        
# In[share of extreme events: related to historical total low output @selected ]

from scipy.sparse import coo_matrix
def reconstruct_2d_array(index_value_pairs, num_rows, num_cols):
    rows, cols, values = zip(*index_value_pairs)
    return coo_matrix((values, (rows, cols)), shape=(num_rows, num_cols)).toarray()



def main(climate_model, scenario, start_year, end_year, low_output):
    
    event_path = r'I:\global_wind_future\low_output_by_percentile\extreme_low_output' 
    
    
    num_rows = 600
    num_cols = 1440    
    
    elos = np.full((end_year - start_year +1, num_rows, num_cols), 0, dtype = np.float32)
    for year in range(start_year, end_year +1):
        file_name = 'event_' + scenario + '_' + str(year) + '_' + climate_model + '.npy'
        d = np.load(os.path.join(event_path, climate_model, file_name))[:,1:]    
        d = pd.DataFrame(d,  columns = ['lon_index', 'lat_index', 'duration_days', 
                                        'power', 'ave','thrs_10th', 'thrs_5th', 'thrs_1th'])  
        
        dd = d.groupby(['lon_index', 'lat_index']).sum().reset_index()
        
        lon_index = dd['lon_index'].astype(int)
        lat_index = dd['lat_index'].astype(int)
        value = dd['ave'] - dd['power'] 
        index_value_pairs = list(zip(lat_index, lon_index, value))       
        power =  reconstruct_2d_array(index_value_pairs, num_rows, num_cols)      

        elos[year-start_year,:] = power / low_output
        
        
    lats = np.arange(89.875,-60,-0.25)
    lons = np.arange(-179.875, 180, 0.25)
      
    maps = xr.Dataset( { 'share_of_extreme_events': (['year',"y", "x"], 
                                                     elos)},     
                        coords = {'year':np.arange(start_year, end_year +1),
                                  'y': lats,
                                  'x': lons} )
      
    save_path = r'I:\global_wind_future\low_output_by_percentile\annual_low_output'                
    file_name =  climate_model + '_' + scenario + '_' + str(start_year) + '_' + str(end_year) +'_share_of_extreme_events_yearly.nc'
    maps.to_netcdf( os.path.join(save_path, climate_model, file_name), 
                   encoding={'share_of_extreme_events': {'zlib': True, 'complevel': 6}})         





#
# main 
#

scenarios = ['historical','ssp126','ssp126','ssp245','ssp245','ssp370','ssp370']
start_years = [1985, 2031, 2071, 2031, 2071, 2031, 2071]
end_years = [2014, 2060, 2100, 2060, 2100, 2060, 2100]

# used for event info plot(time series)
scenarios = ['historical', 'ssp126', 'ssp245', 'ssp370']
start_years = [1985, 2015, 2015, 2015]
end_years = [2014, 2100, 2100, 2100]

climate_models = ['ACCESS-ESM1-5','BCC-CSM2-MR', 
                  'CanESM5','CMCC-ESM2','EC-Earth3',
                  'FGOALS-g3', 'GFDL-ESM4', 'INM-CM4-8', 
                  'IPSL-CM6A-LR', 'MIROC6',
                  #'CNRM-ESM2-1','GISS-E2-1-G', 'KACE-1-0-G',
                  'MPI-ESM1-2-HR', 'UKESM1-0-LL',
                  'MRI-ESM2-0', 'NorESM2-MM'] 


for c in range(0, len(climate_models)):
    
    climate_model = climate_models[c]
    
    s = 0
    scenario = scenarios[s]
    start_year = start_years[s]
    end_year = end_years[s]    
    
    low_output_path = r'I:\global_wind_future\low_output_by_percentile\annual_low_output'   
    low_output = np.full((end_year - start_year +1, 600, 1440), 0, dtype = np.float32)
    for year in range(start_year, end_year +1):
        file_name = 'low_output_' + scenario + '_' + str(year) + '_' + climate_model + '.nc'
        low_output[year-start_year,:] = xr.open_dataset(os.path.join(low_output_path, climate_model, file_name))['annual_low_output'].values
        
        
    for s in range(0,len(scenarios)):
        scenario = scenarios[s]
        start_year = start_years[s]
        end_year = end_years[s]
 
        t = datetime.datetime.now()
        main(climate_model, scenario, start_year, end_year, low_output.mean(0))
        print(climate_model, scenario, datetime.datetime.now()-t)        
                
        
        
        
# In[extreme low-outputs]        


from scipy.sparse import coo_matrix
def reconstruct_2d_array(index_value_pairs, num_rows, num_cols):
    rows, cols, values = zip(*index_value_pairs)
    return coo_matrix((values, (rows, cols)), shape=(num_rows, num_cols)).toarray()


# annual_extreme_low_output
def main(climate_model, scenario, start_year, end_year, low_output):
       
    save_path = r'I:\global_wind_future\low_output_by_percentile\annual_low_output'                
    file_name =  climate_model + '_' + scenario + '_' + str(start_year) + '_' + str(end_year) +'_share_of_extreme_events.nc'
    maps = xr.open_dataset( os.path.join(save_path, climate_model, file_name)) 
    
    extreme_low_output = maps * low_output
    extreme_low_output = extreme_low_output.rename({'share_of_extreme_events': 'extreme_low_output'})
    
    save_path = r'I:\global_wind_future\low_output_by_percentile\annual_low_output'                
    file_name =  climate_model + '_' + scenario + '_' + str(start_year) + '_' + str(end_year) +'_extreme_low_output.nc'
    extreme_low_output.to_netcdf( os.path.join(save_path, climate_model, file_name), 
                                 encoding={'extreme_low_output': {'zlib': True, 'complevel': 6}})     
    
      
        
#
# main 
#

scenarios = ['historical','ssp126','ssp126','ssp245','ssp245','ssp370','ssp370']
start_years = [1985, 2031, 2071, 2031, 2071, 2031, 2071]
end_years = [2014, 2060, 2100, 2060, 2100, 2060, 2100]

climate_models = ['ACCESS-ESM1-5','BCC-CSM2-MR', 
                  'CanESM5','CMCC-ESM2','EC-Earth3',
                  'FGOALS-g3', 'GFDL-ESM4', 'INM-CM4-8', 
                  'IPSL-CM6A-LR', 'MIROC6',
                  #'CNRM-ESM2-1','GISS-E2-1-G', 'KACE-1-0-G',
                  'MPI-ESM1-2-HR', 'UKESM1-0-LL',
                  'MRI-ESM2-0', 'NorESM2-MM'] 


for c in range(9,len(climate_models)):
    
    climate_model = climate_models[c]
    
    s = 0
    scenario = scenarios[s]
    start_year = start_years[s]
    end_year = end_years[s]    
    
    low_output_path = r'I:\global_wind_future\low_output_by_percentile\annual_low_output'   
    low_output = np.full((end_year - start_year +1, 600, 1440), 0, dtype = np.float32)
    for year in range(start_year, end_year +1):
        file_name = 'low_output_' + scenario + '_' + str(year) + '_' + climate_model + '.nc'
        low_output[year-start_year,:] = xr.open_dataset(os.path.join(low_output_path, climate_model, file_name))['annual_low_output'].values
        
        
    for s in range(0,len(scenarios)):
        scenario = scenarios[s]
        start_year = start_years[s]
        end_year = end_years[s]
 
        t = datetime.datetime.now()
        main(climate_model, scenario, start_year, end_year, low_output.mean(0))
        print(climate_model, scenario, datetime.datetime.now()-t)        

        
        
        
        
        