# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import datetime
import pvlib
import warnings

class PVsystem:
    '''
    input parameters:  
        year: int                                                            
        rs：W/m2, daily solar radiation 
        temp_*: air temperature, 2m above surface  K
        wind_speed: m/s
        lon: degree  (-180 - 180)    
        lat: degree  (-90 - 90)  
    '''   
    
    def __init__(self,year, rs, temp_mean, temp_min, temp_max, wind_speed, lon, lat):           
        self.rs = abs(rs) * 86400 # --> daily total J/m2
        self.temp_mean = temp_mean - 273.15
        self.temp_min = temp_min - 273.15
        self.temp_max = temp_max - 273.15
        self.wind_speed = wind_speed        
        self.lon = lon
        self.lat = lat

        
        self.year = year 
        self.timezone = self.TimeZone()
        self.time = self.generating_hour_time()
        self.local_time = self.time.tz_convert(self.timezone)
        self.dayofyear = self.time.dayofyear
        
        self.hourangle = self.HourAngle()
        self.sunpath = self.Sunpath()
        # self.sunrise_hour, self.sunset_hour = self.SunTime()
        
        self.her = self.HourlyExtraterrestrialRadiation()
        self.der = self.dailyExtraterrestrialRadiation()
        
        self._module = pvlib.pvsystem.retrieve_sam('SandiaMod').Canadian_Solar_CS5P_220M___2009_ # or 'CECMod'  
        self._inverter = pvlib.pvsystem.retrieve_sam('SandiaInverter').PV_Powered__PVP2500__240V_


        
        
        
    def TimeZone(self):
        #function: Divide the time zone by longitude
        lon = self.lon
        
        if lon < -180 or lon > 360:
            lon = 0
        if lon > 180:
            lon -= 360       
            
        timezone = round(lon / 15)
        hours = int(timezone)         

        utc_str = f"UTC{hours:+03d}:00"             
        return utc_str  
    
    
    
    def generating_hour_time(self):
        # start from 01.01 0:00 local 
        # timezone = self.TimeZone()
        year = self.year
        start_time = datetime.datetime(year, 1, 1, 0) 
        end_time = datetime.datetime(year, 12, 31, 23) 
        times = pd.date_range(start_time, end_time, freq="h").tz_localize('UTC')
        
        if len(self.rs) == 365 and len(times) == 366*24:
            times = times[times.date != datetime.date(year,2,29)]
            
        if len(self.rs) == 360 and len(times) == 366*24:
            times = times[times.date != datetime.date(year,1,31)]
            times = times[times.date != datetime.date(year,3,31)]
            times = times[times.date != datetime.date(year,7,31)]
            times = times[times.date != datetime.date(year,5,31)]
            times = times[times.date != datetime.date(year,8,31)]
            times = times[times.date != datetime.date(year,10,31)]
            
        if len(self.rs) == 360 and len(times) == 365*24:
            times = times[times.date != datetime.date(year,1,31)]
            times = times[times.date != datetime.date(year,3,31)]
            times = times[times.date != datetime.date(year,7,31)]
            times = times[times.date != datetime.date(year,5,31)]
            times = times[times.date != datetime.date(year,8,31)]
        times = pd.DatetimeIndex(times)    
        return times
 
        

    def HourAngle(self):   
        # function: Calculate the solar hour Angle
        # return: solar hour angle (degree)
    
        # dayofyear = self.time.dayofyear
        
        # eot--> :Soteris A. Kalogirou, "Solar Energy Engineering Processes and Systems, 
        #            2nd Edition" Elselvier/Academic Press (2009).
        eot = pvlib.solarposition.equation_of_time_pvcdrom(self.dayofyear)/60 # min --> hour
        t = int(self.timezone[3:6]) * 15 # The central longitude of the corresponding time zone
        
        w = self.time.hour + 4 * (self.lon - t) / 60 + eot # It takes the 4 minutes to rotate 1 degree.
        ha = (w - 12) * 15
        return np.array(ha)
    
    

    def Sunpath(self):   
        # function: Calculate the zenith, elevation, equation_of_time
        # ! Must be localized or UTC will be assumed.
        ephem_data = pvlib.location.Location(self.lat, self.lon).get_solarposition(self.local_time)
        ephem_data['time'] = ephem_data.index.month*10000 + ephem_data.index.day*100 + ephem_data.index.hour
        ephem_data.sort_values(by = 'time',inplace = True)
        ephem_data.index = self.time
        return ephem_data
    
    
    def SunTime(self):  
        # function: Calculate the sunrise and sunset time (hour)
        
        with warnings.catch_warnings():
             warnings.simplefilter('ignore')    
             # Must be localized to the Location
             # !!return UTC time
             suntime = pvlib.solarposition.sun_rise_set_transit_spa(self.local_time, self.lat, self.lon) 
             # ordered by localtime 00：00-23：00
             suntime['time'] = suntime.index.month*10000 + suntime.index.day*100 + suntime.index.hour
             suntime.sort_values(by = 'time',inplace = True)
             suntime.index = self.time
        
        sunrise_hour = pd.DatetimeIndex(suntime['sunrise'])
        sunset_hour = pd.DatetimeIndex(suntime['sunset'])
   
        return sunrise_hour, sunset_hour   

    

    def dailyExtraterrestrialRadiation(self):
        # function: Calculate the daily Extraterrestrial Radiation (J/m2)
        # GB/T37525—2019

        ISC = 1366.1 #solar constant (W/m2)
        lat = self.lat          
        
        # declination angle
        da = pvlib.solarposition.declination_spencer71(self.dayofyear) 
        
        # sunset hour angle
        x = np.array(-np.tan(lat/180 * np.pi) * np.tan(da))
        x[x>1] = 1
        x[x<-1] = -1
        ws = np.arccos(x) * 180 / np.pi
                  
        E0 = 1 + 0.033 * np.cos(2 * np.pi * self.dayofyear / 365)
        I0 = 24 * 3600 / np.pi * ISC * E0 * ( np.cos( lat / 180 * np.pi) * np.cos(da) *  np.sin( ws / 180 * np.pi)
                                       + np.pi * ws / 180 * np.sin(lat / 180 * np.pi) * np.sin(da) )
        return np.array(I0)
        
    
    
    def HourlyExtraterrestrialRadiation(self):
        # function: Calculate the hourly extraterrestrial horizontal solar irradiation (J/m2)
        # GB/T37525—2019
  
        ISC = 1366.1 #solar constant (W/m2)
        lat = self.lat

        # declination angle (radians)
        da = pvlib.solarposition.declination_spencer71(self.dayofyear) 
        # hour angle
        HourAngle = self.hourangle
        w1 = HourAngle
        w2 = np.hstack((HourAngle[1:], HourAngle[-1:] + 15))
        # error revise
        error = ((w2-w1)<14) | ((w2-w1)>16)
        w2[error] = w1[error] + 15
        
        E0 = 1 + 0.033 * np.cos(2 * np.pi * self.dayofyear / 365)
        I0 = 12 * 3600 / np.pi * ISC * E0 *( np.cos(lat / 180 * np.pi) * np.cos(da) * ( np.sin(w2 / 180 * np.pi) - np.sin(w1 / 180 * np.pi))
                                       + np.pi * (w2 - w1) / 180 * np.sin( lat / 180 * np.pi) * np.sin(da) )    
            
        return np.array(I0)
    
    
        
    
    
    def Pramod_ratio(self):
        """ downscale the daily rsds data to the hourly scale. """
        # REF : Pandey PK, Soupir ML. A new method to estimate average hourly global solar radiation on the horizontal surface. 
        #       Atmospheric Research 114, 83-90 (2012).
        rd = 0.18 + 0.72/( 1 + ((self.dayofyear - 181)/180)**2)
        rh = 0.5 + 0.34/( 1 + ((self.time.hour - 12.1)/3.8)**2)
        r = rd*rh
        
        her = self.her.copy() * np.array(r)
        her[her<0] = 0
        # UTC to match the daily rsds data 
        her = pd.DataFrame(her)
        her.index = self.time
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            factor = np.array(her[0]) / np.array(np.repeat(her.groupby(her.index.date).sum()[0],24))         
        return factor
    
    
    

    def kd(self):
        # function: using BRL model to caculate the fraction of diffuse solar radiation from global solar radiation
        # para Source: Liu et.al.2022 Spatial estimation of the optimum PV tilt angles in China by
        #              incorporating ground with satellite data. Renewable Energy        
        # Lauret et al.2013 Bayesian statistical analysis applied to solar radiation modelling. 
        beta0 = -5.32
        beta1 = 7.2
        beta2 = -0.03
        beta3 = -0.0047
        beta4 = 1.72
        beta5 = 1.08
 
        rs = self.rs
        hourly_d_rs = np.repeat(rs, 24)

        

        factor = self.Pramod_ratio()
        rs_hourly = factor * hourly_d_rs
        
        
        # para
        I0 = np.where(self.der == 0, 1, self.der)

        
        # daily clearness index        
        Kt = np.array(hourly_d_rs)/I0
        
        # hourly clearness index
        kt = np.clip(rs_hourly / self.her, 0, 1)


        #apparent solar time
        AST = (self.hourangle / 15) + 12
        
        #solar altitude angle
        alpha = np.array(self.sunpath['apparent_elevation'])
        
        #persistence of the sky conditions
        phi = kt

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            k = 1 / ( 1 + np.exp( beta0 + beta1 * kt + beta2 * AST + beta3 * alpha + beta4 * Kt + beta5 * phi)) 
        return k, rs_hourly  
   
    
   
    def Airmass(self):
        #function: Calculate absolute (not pressure-adjusted) airmass at sea level.       
        solar_zenith_angle = self.sunpath['apparent_zenith']
        airmass_relative = pvlib.atmosphere.get_relative_airmass(solar_zenith_angle)
        airmass = pvlib.atmosphere.get_absolute_airmass(airmass_relative)
        airmass[(pd.isna(airmass))| (solar_zenith_angle>=89)] = 1
        return airmass
            

    def Fixed(self):
        # funciton: no tracking system (panel tilt, azimuth: south)
        # panel azimuth angle = 180 ,facing South
        # tile: plane tilt angle 
        # dni:Direct Normal Irradiance. [W/m2]
        # ghi:Global horizontal irradiance. [W/m2]
        # dhi:Diffuse horizontal irradiance. [W/m2]

        k, rs_hourly = self.kd()
        lat = self.lat
        
        # Sunpath = self.sunpath
        solar_zenith_angle = self.sunpath['apparent_zenith']
        solar_azimuth_angle = self.sunpath['azimuth']

        dhi = rs_hourly * k / 3600 #To convert to watts per square metre (W m-2).
        ghi = (rs_hourly / 3600 - dhi) 
        
        theta = np.cos( solar_zenith_angle / 180 * np.pi)    
        theta[(theta>0)&(theta<0.1)] = 0.1 
        dni = ghi / np.cos( theta / 180 * np.pi)    
        

        ISC = 1366.1 #solar constant (W/m2)
        E0 = ISC * (1 + 0.033 * np.cos(2 * np.pi * self.dayofyear / 365)) #W/m2
        dni[dni>E0] = E0[dni>E0]
        dni[dni<0] = 0
               
        if lat >= 0:
            panel_azimuth = 180 #facing south
            #source: World estimates of PV optimal tilt angles and ratios of sunlight incident
                     #upon tilted and tracked PV panels relative to horizontal panels.2018
            tilt = 1.3793 + lat * (1.2011 + lat * (-0.014404 + lat * 0.000080509))
        else:
            panel_azimuth = 0 #facing north
            tilt = -0.41657 + lat * (1.4216 + lat * (0.024051 + lat * 0.00021828))
            tilt = -tilt    
            
        ipoa = pvlib.irradiance.get_total_irradiance( tilt,
                                                      panel_azimuth, 
                                                      solar_zenith_angle, solar_azimuth_angle,            
                                                      dni = dni, ghi = ghi, dhi = dhi, model = 'king',
                                                      surface_type = 'concrete' )     
        aoi = pvlib.irradiance.aoi(tilt,panel_azimuth,solar_zenith_angle, solar_azimuth_angle)
        ipoa['aoi'] = aoi    
        return ipoa
    
    

    def dailyTemp2hourlyTemp(self, temp_mean, temp_max, temp_min): 
        
        time_arange = np.array([24] + list(np.arange(1, 24, 1)))
        a = 2 * np.pi / 24 * time_arange
        cos_terms = (
            0.4632 * np.cos(a - 3.805) +
            0.0984 * np.cos(2 * a - 0.360) +
            0.0168 * np.cos(3 * a - 0.822) +
            0.0138 * np.cos(4 * a - 3.513)
        )
        # Calculate the hourly temperature deviation
        t = (temp_max - temp_min) * cos_terms
        
        max_t, min_t = t.max(), t.min()
        ratio_max = (temp_max - temp_mean) / max_t if max_t != 0 else 1
        ratio_min = (temp_min - temp_mean) / min_t if min_t != 0 else 1
        
        t = np.where(t > 0, t * ratio_max, t * ratio_min)
        return t + temp_mean  
            
  
    def PVSystem_sapm_celltemp(self, poa_global):     
        
        temp = np.array([self.dailyTemp2hourlyTemp(self.temp_mean[i], self.temp_max[i], self.temp_min[i])
                            for i in range(len(self.temp_mean))
                        ]).reshape(-1)
        
        
        # function: cell temperature
        a, b, deltaT = (-2.98, -0.0471, 1)  # glass_glass/cell, close roof mount
        wind_speed_h = np.repeat(self.wind_speed, 24)
        temp_cell = pvlib.temperature.sapm_cell(poa_global, temp, wind_speed_h, a, b, deltaT, irrad_ref=1000.)
        return temp_cell
    

    
    def sapm(self):
        '''
        The Sandia PV Array Performance Model (SAPM) 
        '''
   
        Fixed = self.Fixed()
        airmass_absolute = self.Airmass()
        effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(Fixed['poa_direct'], 
                                                                        Fixed['poa_diffuse'], 
                                                                        airmass_absolute, 
                                                                        Fixed['aoi'], 
                                                                        self._module)
        
        
        temp_cell = self.PVSystem_sapm_celltemp(Fixed['poa_global']) 
        para = pvlib.pvsystem.sapm( effective_irradiance, temp_cell, self._module)
        return para

         
    def DC2AC(self):
        # function: Convert DC power and voltage to AC power using Sandia's Grid-Connected PV Inverter model.
        #  Returns
        #  -------
        #  power_ac : numeric [ Wh /d ]
        #  AC power output. [ Wh /d ]

        dc_para = self.sapm()
        p_dc = dc_para['i_mp'] * dc_para['v_mp']
        power_ac = pvlib.inverter.sandia(dc_para['v_mp'], p_dc, self._inverter) 
        
        # to daily total
        daily_power_ac = power_ac.groupby(power_ac.index.dayofyear).sum() 
        return daily_power_ac

    
    
    def Area_ajust(self):
        # funtion: available area / module area
        # modules = pvlib.pvsystem.retrieve_sam('SandiaMod') # or 'CECMod'
        # module = modules.Canadian_Solar_CS5P_220M___2009_
        Area_ajust = 1 / self._module.Area               
        return Area_ajust

    

    
    
    
    
    
# In[1]
def main(inputs):
    pixel_type, year, rsds, wind_speed, temp_mean, temp_min, temp_max, lon, lat = inputs[:] 
    if pixel_type == 1:
        f=PVsystem(year,
                   rsds,
                   temp_mean,
                   temp_min, 
                   temp_max, 
                   wind_speed,
                   lon,lat)

        power = f.DC2AC() * f.Area_ajust() #unit: wh
        return np.array(power, dtype=np.float32)
    else:
        power=np.array([0]*len(rsds), dtype=np.float32)
        return power

    

