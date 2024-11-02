import os


import xarray as xr
import numpy as np
import pandas as pd
import datetime
import dask.dataframe as dd

import warnings
warnings.filterwarnings('ignore')



num_rows = 600
num_cols = 1440
   

from scipy.sparse import coo_matrix
def reconstruct_2d_array(index_value_pairs, num_rows, num_cols):
    rows, cols, values = zip(*index_value_pairs)
    return coo_matrix((values, (rows, cols)), shape=(num_rows, num_cols)).toarray()



def initialize_zones(power):
    #
    #### define configure : PV
    #    
    if power == 'pv':
        event_path = r'G:\global_photovoltatic_future\low_output_by_percentile\extreme_events'
        climate_models = ['ACCESS-ESM1-5','BCC-CSM2-MR' 
                          ,'CanESM5','CMCC-ESM2','EC-Earth3'
                          ,'FGOALS-g3', 'GFDL-ESM4', 'INM-CM4-8' 
                          ,'IPSL-CM6A-LR', 'KACE-1-0-G', 'MIROC6'
                          ,'CNRM-ESM2-1','GISS-E2-1-G' 
                          ,'MPI-ESM1-2-HR', 'UKESM1-0-LL'
                          ,'MRI-ESM2-0', 'NorESM2-MM'
                          ]  
        
        pv_no_utilization = xr.open_dataset(r'I:\limited_area\for_pv.tif')['band_data'].values[0,:]
        development_zone_pv = pv_no_utilization < 0.5
        pv_install = xr.open_dataset(r'G:\global_pv_installations\count.tif')['band_data'].values[0,:] # km
        
        # built_up
        built_up_lat = np.where(pv_install!=0)[0]
        built_up_lon = np.where(pv_install!=0)[1]
        built_up_zones = pd.DataFrame(np.array([built_up_lat, built_up_lon]).T , columns=['lat_index', 'lon_index'])
        built_up_zones['land_class'] = 1#'built_up'
        
        # develop_area
        develop_area_lat = np.where(development_zone_pv)[0]
        develop_area_lon = np.where(development_zone_pv)[1]
        develop_zones = pd.DataFrame(np.array([develop_area_lat, develop_area_lon]).T , columns=['lat_index', 'lon_index'])
        develop_zones['land_class'] = 2#'develop'
        zones = pd.concat((built_up_zones, develop_zones))
        zones = zones.drop_duplicates(subset=['lat_index', 'lon_index'])
        del built_up_lat, built_up_lon, built_up_zones, develop_area_lat, develop_area_lon, develop_zones
    
    #
    #### define configure : wind
    #    
    if power == 'wind':
        event_path = r'I:\global_wind_future\low_output_by_percentile\extreme_events'
        climate_models = ['ACCESS-ESM1-5','BCC-CSM2-MR' 
                          ,'CanESM5','CMCC-ESM2','EC-Earth3'
                            ,'FGOALS-g3', 'GFDL-ESM4', 'INM-CM4-8' 
                            ,'IPSL-CM6A-LR', 'MIROC6'
                          #,'CNRM-ESM2-1','GISS-E2-1-G' , 'KACE-1-0-G'
                            ,'MPI-ESM1-2-HR', 'UKESM1-0-LL'
                            ,'MRI-ESM2-0', 'NorESM2-MM'
                          ] 
        
        dis2coastline = xr.open_dataset(r'I:\global_wind_future\dis2coastline.tif')['band_data'].values[0,:] # km
        land = xr.open_dataset(r'I:\global_wind_future\land.tif')['band_data'].values[0,:]
        
        wind_no_utilization = xr.open_dataset(r'I:\limited_area\for_wind.tif')['band_data'].values[0,:]
        wind_no_utilization[(land==0)] = 1
        development_zone_wind = wind_no_utilization < 0.5
        offshore = (land==0) & (dis2coastline<200*1000)
        global_all = (land == 1) | offshore
        
        
        # built up
        turbine_install = xr.open_dataset(r'I:\global_wind_turbine\count.tif')['band_data'].values[0,:] # km
        turbine_install[np.isnan(turbine_install)] = 0
     
        
        # 1: built_up
        built_up_lat = np.where(turbine_install!=0)[0]
        built_up_lon = np.where(turbine_install!=0)[1]
        built_up_zones = pd.DataFrame(np.array([built_up_lat, built_up_lon]).T , columns=['lat_index', 'lon_index'])
        built_up_zones['land_class'] = 1 #'built_up'
        
        # 2: develop_area
        develop_area_lat = np.where(development_zone_wind)[0]
        develop_area_lon = np.where(development_zone_wind)[1]
        develop_zones = pd.DataFrame(np.array([develop_area_lat, develop_area_lon]).T , columns=['lat_index', 'lon_index'])
        develop_zones['land_class'] = 2 #'develop'
        
        # 3: offshore
        offshore_lat = np.where(offshore)[0]
        offshore_lon = np.where(offshore)[1]
        offshore_zones = pd.DataFrame(np.array([offshore_lat, offshore_lon]).T , columns=['lat_index', 'lon_index'])
        offshore_zones['land_class'] = 3 #'offshore'
        
                
        zones = pd.concat((built_up_zones, develop_zones, offshore_zones))
        zones = zones.drop_duplicates(subset=['lat_index', 'lon_index'])
        del built_up_lat, built_up_lon, built_up_zones, develop_area_lat, develop_area_lon, develop_zones, offshore_lat, offshore_lon, offshore_zones
        # land_class = ['built_up', 'develop', 'offshore','None']
        
    lon_index = zones['lon_index']
    lat_index = zones['lat_index']
    value = zones['land_class']
    index_value_pairs = list(zip(lat_index, lon_index, value))       
    land_class =  reconstruct_2d_array(index_value_pairs, num_rows, num_cols)  
    return event_path, climate_models, land_class, global_all, land, offshore





# read elps, yearly
def elps(power, scenario, period):
    if power == 'wind':
        base_path = r'I:\global_wind_future\low_output_by_percentile\annual_low_output'
        climate_models = ['ACCESS-ESM1-5','BCC-CSM2-MR','CanESM5','CMCC-ESM2','EC-Earth3',
                          'FGOALS-g3', 'GFDL-ESM4', 'INM-CM4-8', 'IPSL-CM6A-LR', 'MIROC6',
                          #'CNRM-ESM2-1','GISS-E2-1-G',  'KACE-1-0-G',
                          'MPI-ESM1-2-HR', 'UKESM1-0-LL','MRI-ESM2-0', 'NorESM2-MM']    
    if power == 'pv':
        base_path = r'H:\global_photovoltatic_future\low_output_by_percentile\annual_low_output'
        climate_models = ['ACCESS-ESM1-5','BCC-CSM2-MR','CanESM5','CMCC-ESM2','EC-Earth3',
                          'FGOALS-g3', 'GFDL-ESM4', 'INM-CM4-8', 'IPSL-CM6A-LR', 'KACE-1-0-G', 'MIROC6',
                          'CNRM-ESM2-1','GISS-E2-1-G', 
                          'MPI-ESM1-2-HR', 'UKESM1-0-LL', 
                          'MRI-ESM2-0', 'NorESM2-MM']  
        
    var_name = 'share_of_extreme_events'
    dataset = []
    for i in range(len(climate_models)):
        climate_model = climate_models[i]
        
        file_name = climate_model + '_' + scenario + '_' + period + '_' + var_name + '_yearly.nc'   
        d = xr.open_dataset(os.path.join(base_path, climate_model, file_name))
        dataset.append(d) 
        
    elps = xr.concat(dataset, dim=xr.DataArray(climate_models, dims='climate_model', name='climate_model'))
    return elps





def grided_statistic(d):
    groupby_subset = ['lon_index', 'lat_index']
    
    # convert pandas DataFrame to dask DataFrame
    ddf = dd.from_pandas(d, npartitions=8)  # npartitions can be revised base your computer      
    
    # frequency
    freq_event = ddf.groupby(groupby_subset)['power'].count().compute().reset_index()
    
    # Calculate total days \ Maximum duration days
    days = ddf.groupby(groupby_subset)['duration_days'].agg(['sum','max']).compute().reset_index()    
        
    # intensity
    ddf['ab_intensity'] = (ddf['ave'] - ddf['power']) 
    ddf['re_intensity'] = (ddf['ave'] - ddf['power']) / ddf['ave']
    mask0 = (ddf['ave'] > 0)
    mean_intensity = ddf[mask0].groupby(groupby_subset)[['ab_intensity', 're_intensity']].mean().compute().reset_index()   
    
    # Combine all calculations
    data_mode = pd.merge(freq_event, days, on=groupby_subset, how='left')    
    data_mode = pd.merge(data_mode, mean_intensity, on=groupby_subset, how='left')

    
    data_mode.columns = ['lon_index','lat_index', 'freq', 'total_days', 'max_duration', 
                         'mean_ab_intensity', 'mean_re_intensity'] 
    data_mode = data_mode.fillna(0)
    return data_mode
    


#%% global time-series
def get_global_time_series():
    power = 'wind' # same process for PV
    event_path, climate_models, land_class, global_all, land, offshore = initialize_zones(power)
    
        
    scenarios = ['historical', 'ssp126', 'ssp245',  'ssp370']
    periods  = ['1985_2014', '2015_2100', '2015_2100', '2015_2100']    
    
    
    indicators = ['ELPS', 
                  'frequency', 
                  'total_days', 
                  'max_duration', 
                  'mean_duration', 
                  'mean_re_intensity'
                 ]
    
    
    climate_models = ['ACCESS-ESM1-5','BCC-CSM2-MR', 
                      'CanESM5','CMCC-ESM2','EC-Earth3',
                      'FGOALS-g3', 'GFDL-ESM4', 'INM-CM4-8', 
                      'IPSL-CM6A-LR', 'MIROC6',
                      #'CNRM-ESM2-1','GISS-E2-1-G', 'KACE-1-0-G'
                      'MPI-ESM1-2-HR', 'UKESM1-0-LL',
                      'MRI-ESM2-0', 'NorESM2-MM'] 
    
    if 'dfs' in vars():
        del dfs
    
    for i in range(0, len(scenarios)):
        scenario = scenarios[i]
        period = periods[i]
        
        # elps    
        data_elps = elps(power, scenario, period)
        data_elps = data_elps.rename({'share_of_extreme_events':'ELPS'})
    
        # event_info
        file_name = 'event_info_' + scenario + '_' + period +'.nc'
        data = xr.open_dataset( os.path.join(event_path, file_name) )
        data['mean_duration'] = data['total_days'] / data['frequency']
        
        data = xr.merge([data_elps,data])
        data = data.sel(climate_model = data.climate_model.isin(climate_models))
        
        years = data.year.data
        
            
        for indicator in indicators:
            data_indicator = data[indicator].values
    
            global_all_area = np.nanmean(data_indicator[:,:, global_all], axis = 2)
            land_area = np.nanmean(data_indicator[:,:, land==1], axis = 2)
            offshore_area = np.nanmean(data_indicator[:,:, offshore], axis = 2)
            bulit_up = np.nanmean(data_indicator[:,:, land_class==1], axis = 2)
            onshore_developed = np.nanmean(data_indicator[:,:, land_class==2], axis = 2)
            offshore_developed = np.nanmean(data_indicator[:,:, land_class==3], axis = 2)   
            
    
             
            df1 = pd.DataFrame(np.array([np.nanmean(global_all_area, axis=0), 
                                         np.nanmean(land_area, axis=0), 
                                         np.nanmean(offshore_area, axis=0), 
                                         np.nanmean(bulit_up, axis=0), 
                                         np.nanmean(onshore_developed, axis=0), 
                                         np.nanmean(offshore_developed, axis=0)
                                         ]                                    
                                        ).T,
                              columns = ['global_all','onshore','offshore', 'bulit_up', 'onshore_developed', 'offshore_developed'])
            df1['year'] = years
            df1['scenario'] = scenario
            df1['indicator'] = indicator
            df1['climate_model'] = 'mme'
            
            
            
            df2 = pd.DataFrame(np.array([np.nanpercentile(global_all_area, 90, axis=0),
                                         np.nanpercentile(land_area, 90, axis=0),
                                         np.nanpercentile(offshore_area, 90, axis=0),
                                         np.nanpercentile(bulit_up, 90, axis=0),
                                         np.nanpercentile(onshore_developed, 90, axis=0),
                                         np.nanpercentile(offshore_developed, 90, axis=0)
                                         ]).T,
                              columns = ['global_all','onshore','offshore','bulit_up','onshore_developed','offshore_developed'])
            df2['year'] = years
            df2['scenario'] = scenario
            df2['indicator'] = indicator
            df2['climate_model'] = '90p'
            
            
            
            df3 = pd.DataFrame(np.array([np.nanpercentile(global_all_area, 10, axis=0),
                                         np.nanpercentile(land_area, 10, axis=0),
                                         np.nanpercentile(offshore_area, 10, axis=0),
                                         np.nanpercentile(bulit_up, 10, axis=0),
                                         np.nanpercentile(onshore_developed, 10, axis=0),
                                         np.nanpercentile(offshore_developed, 10, axis=0),
                                         ]).T,
                              columns = ['global_all','onshore','offshore','bulit_up','onshore_developed','offshore_developed'])
            df3['year'] = years
            df3['scenario'] = scenario
            df3['indicator'] = indicator
            df3['climate_model'] = '10p'        
            
            df = pd.concat((df1,df2,df3))
            
    
            if 'dfs' not in vars():
                dfs = df.copy()
            else:
                dfs = pd.concat((dfs, df))
                
            print(scenario, period, indicator)
    
            
    dfs.to_csv(r'I:\global_wind_future\low_output_by_percentile\extreme_events\time_series_1985-2100.csv')    




#%% global maps
def get_global_maps():
    power = 'wind' # same process for PV
    event_path, climate_models, land_class, global_all, land, offshore = initialize_zones(power)
    
    
    climate_models = ['ACCESS-ESM1-5','BCC-CSM2-MR', 
                      'CanESM5','CMCC-ESM2','EC-Earth3',
                      'FGOALS-g3', 'GFDL-ESM4', 'INM-CM4-8', 
                      'IPSL-CM6A-LR', 'MIROC6',
                      #'CNRM-ESM2-1','GISS-E2-1-G', 'KACE-1-0-G'
                      'MPI-ESM1-2-HR', 'UKESM1-0-LL',
                      'MRI-ESM2-0', 'NorESM2-MM'] 
    
    
    scenarios = ['historical', 'ssp126', 'ssp245', 'ssp370']
    periods  = ['1985_2014', '2015_2100', '2015_2100', '2015_2100']
    
    columns = ['freq', 
               'total_days', 
               'max_duration', 
               'mean_ab_intensity', 
               'mean_re_intensity'
               ]
    
    
    
    t = datetime.datetime.now()
    for c in range(0,len(climate_models)):
        climate_model = climate_models[c]
    
        for i in range(2,len(scenarios)):
            scenario = scenarios[i]
            period = periods[i]
            
            start_year = int(period[:4])
            end_year = int(period[-4:])
            
            maps = np.full((len(columns), end_year - start_year+1, num_rows, num_cols), np.nan, dtype = np.float32)
            for year in range(start_year, end_year+1):
                
                file_name = 'event_' + scenario + '_' + str(year) + '_' + climate_model + '.npy'
                d = np.load(os.path.join(event_path, climate_model, file_name))[:,1:]    
                d = pd.DataFrame(d,  columns = ['lon_index', 'lat_index', 'duration_days', 
                                                'power', 'ave','thrs_10th', 'thrs_5th', 'thrs_1th']) 
    
                data = grided_statistic(d)
    
                lon_index = data['lon_index'].astype(int)
                lat_index = data['lat_index'].astype(int)
                
                for col in range(0, len(columns)):
                    value = data[columns[col]]
                    index_value_pairs = list(zip(lat_index, lon_index, value))      
                    maps[col, year - start_year, :] =  reconstruct_2d_array(index_value_pairs, num_rows, num_cols)    
                    
                print(climate_model, scenario, year, datetime.datetime.now() - t)
                
            lats = np.arange(89.875,-60,-0.25)
            lons = np.arange(-179.875, 180, 0.25)
            maps = xr.Dataset( { 'frequency': (["year", "y", "x"], maps[0]), 
                                   'total_days': (["year", "y", "x"], maps[1]),  
                                   'max_duration':  (["year", "y", "x"], maps[2]),   
                                   'mean_ab_intensity':  (["year", "y", "x"], maps[3]),
                                   'mean_re_intensity':  (["year", "y", "x"], maps[4])
                                   },     
                                coords = {'y': lats,
                                          'x': lons,
                                          "year": range(start_year, end_year+1)} )
            
            
            save_path = r'I:\global_wind_future\low_output_by_percentile\extreme_events\event_info'                
            file_name =  climate_model + '_' + scenario + '_' + period +'.nc'
            maps.to_netcdf( os.path.join(save_path, file_name), 
                           encoding= { 'frequency': {'zlib': True, 'complevel': 6},
                                        'total_days': {'zlib': True, 'complevel': 6},
                                        'max_duration':  {'zlib': True, 'complevel': 6},
                                        'mean_ab_intensity':  {'zlib': True, 'complevel': 6},
                                        'mean_re_intensity':  {'zlib': True, 'complevel': 6} 
                                        }
                         ) 
            
            print(climate_model, scenario, period, datetime.datetime.now() - t)
     
                  
