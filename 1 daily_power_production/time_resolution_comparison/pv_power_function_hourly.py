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
        time: numpy.datetime64                                                             
        rs：J/m2, houly solar radiation array 
        temp_air: air temperature, 2m above surface  K
        wind_speed: m/s
        lon: degree  (-180 - +180)    
        lat: degree  (-90-90)  

    '''    
    
    def __init__(self, year, time, rs, temp_air, wind_speed, lon, lat):               
        self.rs = rs   
        self.temp_air = temp_air - 273.15
        self.wind_speed = wind_speed        
        self.lon = lon
        self.lat = lat

        
        self.year = year   
        self.timezone = self.TimeZone()
        self.utc_time = pd.to_datetime(time).tz_localize(None).tz_localize("UTC")
        self.local_time = self.utc_time.tz_convert(self.timezone)
        self.local_doy = self.local_time.dayofyear
        
      
          
    def TimeZone(self):
        #function: Divide the time zone by longitude
        lon = self.lon
        
        if lon < -180 or lon > 360:
            lon = 0
        if lon > 180:
            lon -= 360       
            
        timezone = round(lon / 15)
        hours = int(timezone)  # 小时部分        

        utc_str = f"UTC{hours:+03d}:00"             
        return utc_str  
    
   

    def HourAngle(self):   
        # function: Calculate the solar hour Angle
        # GB/T37525—2019
        # return: solar hour angle (degree)
               
        # eot--> Soteris A. Kalogirou, "Solar Energy Engineering Processes and Systems, 
        #        2nd Edition" Elselvier/Academic Press (2009).        
        eot = pvlib.solarposition.equation_of_time_pvcdrom(self.local_doy)/60 # min --> hour        
        
        #
        # lon in pvlib input format: -180 - +180
        t = int(self.timezone[3:6])*15 # The central longitude of the corresponding time zone
        w = self.local_time.hour + 4* (self.lon - t) / 60 + eot # It takes the 4 minutes to rotate 1 degree.
        ha = (w -12) * 15
        return np.array(ha, dtype=np.float32)
    
    
    
    def Sunpath(self):   
        # function: Calculate the zenith, elevation, equation_of_time
        # source:pvlib package     
        
        #lon in pvlib input format: -180 - +180
        ephem_data = pvlib.location.Location(self.lat, self.lon).get_solarposition(self.local_time)
        return ephem_data
    
    
    
    
    def SunTime(self):  
        # function: Calculate the sunrise and sunset time (hour)
        # source:pvlib package
        with warnings.catch_warnings():
             warnings.simplefilter('ignore')            
             suntime = pvlib.solarposition.sun_rise_set_transit_spa(self.local_time, self.lat, self.lon)

        #极昼与极夜设空值
        sunrise_hour=np.array( [np.nan if pd.isnull(suntime['sunrise'].iloc[x]) else suntime['sunrise'].iloc[x].timetuple().tm_hour
                               for x in range(len(suntime))] )
        sunset_hour=np.array( [np.nan if pd.isnull(suntime['sunset'].iloc[x]) else suntime['sunset'].iloc[x].timetuple().tm_hour
                               for x in range(len(suntime))] )
        return sunrise_hour, sunset_hour   
    
    
    
    def dailyExtraterrestrialRadiation(self):
        # function: Calculate the daily Extraterrestrial Radiation (J/m2)
        # GB/T37525—2019
        # doy : the day of year
        # timezone = self.TimeZone()
        ISC = 1366.1 # solar constant (W/m2)
        lat = self.lat          
    
        # declination angle
        da = pvlib.solarposition.declination_spencer71(self.local_doy) 
        
        # sunset hour angle
        x=np.array(-np.tan(lat/180*np.pi)*np.tan(da))
        x[x>1]=1
        x[x<-1]=-1
        ws = np.arccos(x) *180 /np.pi
                  
        E0 = 1 + 0.033 * np.cos(2 * np.pi * self.local_doy/365)
        I0 = 24*3600/np.pi * ISC * E0 *( np.cos(lat/180*np.pi) * np.cos(da) *  np.sin(ws/180*np.pi)
                                        + np.pi*ws/180 * np.sin(lat/180*np.pi)*np.sin(da) )

        return np.array(I0, dtype=np.float32)
        
    
    
    def HourlyExtraterrestrialRadiation(self):
        # function: Calculate the hourly extraterrestrial horizontal solar irradiation (J/m2)
        # GB/T37525—2019     
        ISC = 1366.1 #solar constant (W/m2)
        lat = self.lat
        
        # declination angle (radians)
        da = pvlib.solarposition.declination_spencer71(self.local_doy) 
        
        # hour angle
        HourAngle = self.HourAngle()
        w1 = HourAngle
        w2 = np.hstack((HourAngle[1:],HourAngle[-1:]+15))

        error = ((w2-w1)<14) | ((w2-w1)>16)
        w2[error] = w1[error]+15
        
        E0 = 1 + 0.033 * np.cos(2*np.pi*self.local_doy/365)
        I0 = 12 * 3600/np.pi * ISC * E0 *( np.cos(lat/180*np.pi) * np.cos(da) * ( np.sin(w2/180*np.pi) - np.sin(w1/180*np.pi))
                                         + np.pi*(w2-w1)/180 * np.sin(lat/180*np.pi)*np.sin(da) )
            
        return np.array(I0, dtype=np.float32)
    

    
    
    def kd(self):
        # function: using BRL model to caculate the fraction of diffuse solar radiation from global solar radiation
        # para Source: Liu et.al.2022 Spatial estimation of the optimum PV tilt angles in China by
        #             incorporating ground with satellite data. Renewable Energy        
        # Lauret et al.2013 Bayesian statistical analysis applied to solar radiation modelling. 
        beta0 = -5.32 
        beta1 = 7.28
        beta2 = -0.03
        beta3 = -0.0047 
        beta4 = 1.72 
        beta5 = 1.08 
        
        
        rs = self.rs 
        sunrise_hour,sunset_hour = self.SunTime()
        # para
        I0 = self.dailyExtraterrestrialRadiation()
        I0[I0==0] = 1 
        #
        # daily clearness index
        daily_rs = rs.reshape(24, int(len(rs)/24))
        daily_rs = np.sum(daily_rs,axis=0)

        hourly_d_rs=[]
        for i in range(len(daily_rs)):
            hourly_d_rs.extend([daily_rs[i]]*24)
        Kt = np.array(hourly_d_rs)/I0
        
        # hourly clearness index
        kt  = rs / self.HourlyExtraterrestrialRadiation()
        kt[kt>1] = 1
        kt[kt<0] = 0

        # apparent solar time
        AST = (self.HourAngle()/15) + 12
        
        # solar altitude angle
        alpha = np.array(self.Sunpath()['apparent_elevation'])
        
        # persistence of the sky conditions
        timezone = self.TimeZone()
        utcoffset = int(timezone[4:6])
        indexs = np.arange(utcoffset, utcoffset+len(rs),1,dtype=np.int32)
        hour_index = indexs%24 
        phi = np.convolve(kt, np.array([1/2, 1/2]), mode='full')[:-1]
        

        phi[hour_index==sunrise_hour] = kt[hour_index==sunrise_hour]
        phi[hour_index==sunset_hour] = kt[hour_index==sunset_hour]
        phi[hour_index<sunrise_hour] = 0
        phi[hour_index>sunset_hour] = 0
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            k = 1/(1+np.exp(beta0 + beta1*kt + beta2*AST + beta3*alpha + beta4*Kt + beta5*phi))             
        return k    
    
    
    
    

    def Airmass(self):
        # function: Calculate absolute (not pressure-adjusted) airmass at sea level.       
        solar_zenith_angle = self.Sunpath()['apparent_zenith']
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

        k = self.kd()
        rs = self.rs
        lat = self.lat
        
        Sunpath = self.Sunpath()
        solar_zenith_angle = Sunpath['apparent_zenith']
        solar_azimuth_angle = Sunpath['azimuth']

        dhi = rs*k/3600 #To convert to watts per square metre (W m-2).
        ghi = (rs/3600-dhi) 
        
        theta = np.cos( solar_zenith_angle / 180 * np.pi)    
        theta[(theta>0)&(theta<0.1)] = 0.1 # 防止dni在天顶角接近90°时过大失真
        dni = ghi / np.cos( theta / 180 * np.pi)            
    

        ISC = 1366.1 #solar constant (W/m2)
        E0 = ISC * (1 + 0.033 * np.cos(2 * np.pi * self.local_doy / 365)) #W/m2
        dni[dni>E0] = E0[dni>E0]
        dni[dni<0] = 0
        
        
        if lat>=0:
            panel_azimuth = 180 #facing south
            # source: World estimates of PV optimal tilt angles and ratios of sunlight incident
                      #upon tilted and tracked PV panels relative to horizontal panels.2018
            tilt = 1.3793+lat*(1.2011+lat*(-0.014404+lat*0.000080509))
        else:
            panel_azimuth = 0 #facing north
            tilt = -0.41657+lat*(1.4216+lat*(0.024051+lat*0.00021828))
            tilt = -tilt     
            
        ipoa = pvlib.irradiance.get_total_irradiance( tilt,
                                                       panel_azimuth, #facing south
                                                       solar_zenith_angle, solar_azimuth_angle,            
                                                       dni=dni,ghi=ghi,dhi=dhi,model='king',
                                                       surface_type='concrete')     
        aoi = pvlib.irradiance.aoi(tilt,panel_azimuth,solar_zenith_angle, solar_azimuth_angle)
        ipoa['aoi'] = aoi
        return ipoa
    
     
  
    def PVSystem_sapm_celltemp(self,poa_global):
        # function: cell temperature
        # King model
        a, b, deltaT = (-2.98, -0.0471, 1)  # glass_glass/cell, close roof mount
        temp_air = self.temp_air
        wind_speed = self.wind_speed
        temp_cell = pvlib.temperature.sapm_cell(poa_global, temp_air, wind_speed, a, b, deltaT,irrad_ref=1000.)
        return temp_cell
    
 
    
    def sapm(self,system_type):
        '''
        The Sandia PV Array Performance Model (SAPM) generates 5 points on a
        PV module's I-V curve (Voc, Isc, Ix, Ixx, Vmp/Imp) according to
        SAND2004-3535. Assumes a reference cell temperature of 25°C.
        '''
        modules = pvlib.pvsystem.retrieve_sam('SandiaMod') # or 'CECMod'
        module = modules.Canadian_Solar_CS5P_220M___2009_
   
        if system_type == 'Two_axis':
            TwoAxis = self.TwoAxis()
            poa_global=TwoAxis['poa_global']
            poa_direct=TwoAxis['poa_direct']
            poa_diffuse=TwoAxis['poa_diffuse']
            aoi=TwoAxis['aoi']
            
        if system_type == 'Single_axis':
            SingleAxis = self.SingleAxis()
            poa_global = SingleAxis['poa_global']
            poa_direct = SingleAxis['poa_direct']
            poa_diffuse = SingleAxis['poa_diffuse']
            aoi = SingleAxis['aoi']
            
        if system_type=='Fixed':
            Fixed = self.Fixed()
            poa_global = Fixed['poa_global']
            poa_direct = Fixed ['poa_direct']
            poa_diffuse = Fixed ['poa_diffuse']
            aoi = Fixed['aoi']
            
        airmass_absolute = self.Airmass()
        effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(poa_direct, poa_diffuse, airmass_absolute, aoi, module)
        temp_cell = self.PVSystem_sapm_celltemp(poa_global) 
        para = pvlib.pvsystem.sapm(effective_irradiance, temp_cell, module)
        return para
         
    
    
    def DC2AC(self):
        # function: Convert DC power and voltage to AC power using Sandia's Grid-Connected PV Inverter model.
        invdb = pvlib.pvsystem.retrieve_sam('SandiaInverter')
        inverter = invdb.PV_Powered__PVP2500__240V_
        dc_para = self.sapm('fixed')
        v_dc = dc_para['v_mp']
        p_dc = dc_para['i_mp']*dc_para['v_mp']
        power_ac = pvlib.inverter.sandia(v_dc, p_dc, inverter)#负值为逆变器的消耗电力，夜间
        return power_ac
    
    

    
    def Area_ajust(self):
        # funtion: available roof area / module area
        modules = pvlib.pvsystem.retrieve_sam('SandiaMod') # or 'CECMod'
        module = modules.Canadian_Solar_CS5P_220M___2009_
        Area_ajust=  1 / module.Area               
        return Area_ajust    
    
    

    
# In[1]

def main(inputs):
    land, year, time, rsds, wind, temp, lon, lat = inputs[:]
    if land==1:
        f=PVsystem(year,
                   time,
                   rsds,
                   temp,
                   wind, 
                   lon,
                   lat)

        power_ac = f.DC2AC()
        Area_ajust = f.Area_ajust()
        power= power_ac * Area_ajust # unit: w        
        return np.array(power,dtype=np.float32)
    
    else:
        power=np.array([0]*len(rsds),dtype=np.float32)
        return power

