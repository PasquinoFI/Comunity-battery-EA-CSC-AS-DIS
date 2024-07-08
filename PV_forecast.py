import numpy as np
import pandas as pd
from datetime import datetime,timedelta
from pvlib import location
from pvlib import irradiance
import matplotlib.pyplot as plt

def DWD_forecast(scenario,ahead,time):
    """
    Parameters
    ----------
    scenario : 0-39
        DWD provides 40 different scenarios https://opendata.dwd.de/weather/nwp/icon-eu-eps/grib/
    ahead : 0-48
        forecast horizon
    time : datatime
        current time es.datetime.strptime('2024-03-15 00:00:00', '%Y-%m-%d %H:%M:%S')
        dwd updates the forecasts at 00, 06, 12 and 18. The most recent forecasts will be used as forecast time

    Returns
    -------
    Global Horizontal Irradiance and Temperature forecasts

    """
    
    time += -timedelta(hours=1) # utc -> uct+1
    day = str(time)[:10]
    
    ### remember that dwd databaase is UTC and not UTC+1
    if 0 <= time.hour < 6:
        forecast_time = '00'
        lag = time.hour-0
    if 6 <= time.hour < 12:
        forecast_time = '06'
        lag = time.hour-6
    if 12 <= time.hour < 18:
        forecast_time = '12'
        lag = time.hour-12
    if 18 <= time.hour < 24:
        forecast_time = '18'
        lag = time.hour-18
        
    irr = np.load(f"C:\\Users\\pasqui\\Desktop\\GIT\\Comunity-battery-EA-CSC-AS-DIS\\DWD_irradiance\\Location_Lausanne_time_of_forecast_{day}T{forecast_time}.npy")
    irr = irr[scenario,1+lag:ahead+1+lag]
    temp = np.load(f"C:\\Users\\pasqui\\Desktop\\GIT\\Comunity-battery-EA-CSC-AS-DIS\\DWD_temperature\\Location_Lausanne_time_of_forecast_{day}T{forecast_time}.npy")
    temp = temp[scenario,1+lag:ahead+1+lag]
    index_vector = []
    current_time = time
    while current_time < time + timedelta(hours=ahead):
        index_vector.append(current_time)
        current_time += timedelta(hours=1)
        
    ghi = pd.DataFrame({'ghi [W/m2]':irr})
    ghi.index = index_vector
    ghi.index = ghi.index.shift(freq='1H') # utc -> uct+1
    
    temp = pd.DataFrame({'temp [K]':temp})
    temp.index = index_vector
    temp.index = temp.index.shift(freq='1H') # utc -> uct+1
    
    return(ghi,temp)

def estimate_pv_production(GHI, temp, tilt, azimuth, alpha):
    '''
    Parameters
    ----------
    GHI : Global Horizontal Irradiance [W/m2] dataframe with with one column named 'ghi [W/m2]'. Index should be datetime object in UTC with 15-min interval
    temp : temperature [K] dataframe with one column named 'temp [K]'. Index should be datetime object in UTC with 15-min interval
    tilt : list of panel tilts [0-90] 0: horizontal, 90: vertical
    azimuth : list of orientations [0-360] 0: north, 180: south
    alpha : list of rated capacity [W]

    Returns
    -------
    estimated_PV production [kW] (DC)

    '''

    assert len(GHI) == len(temp),"GHI and temperature timeseries have different length."
    assert len(tilt) == len(azimuth),"Number of tilt and azimuth are different."
    assert len(alpha) == len(azimuth),"Number of alpha and azimuth are different."
    
    tz = 'UTC'
    lat, lon = 46.518374, 6.565068

    # Create location object to store lat, lon, timezone
    site = location.Location(lat, lon, tz=tz)
    
    solar_position = site.get_solarposition(times=GHI.index)
    DNI = irradiance.disc(GHI['ghi [W/m2]'].values,solar_position['zenith'],GHI.index)
    DNI = list(DNI.items())[0][1]
    DHI = GHI['ghi [W/m2]'].values - np.cos(np.deg2rad(solar_position['zenith']))*DNI # DHI form disc model
    HExtra = irradiance.get_extra_radiation(GHI.index);   #extraterrestrial irradiance

    temperature = temp['temp [K]'].values
    
    n_orientations = len(tilt)
    n_samples = len(GHI)
    proxies = np.zeros((n_samples,n_orientations))
    for i in range(n_orientations):
        POA_irradiance = irradiance.get_total_irradiance(
                surface_tilt=tilt[i],
                surface_azimuth=azimuth[i],
                dni=DNI,
                dni_extra = HExtra,
                ghi=GHI['ghi [W/m2]'].values,
                dhi=DHI,
                solar_zenith=solar_position['apparent_zenith'],
                solar_azimuth=solar_position['azimuth'],
                model='haydavies')
        proxies_temp = POA_irradiance['poa_global'].values
        proxies_temp_orig = proxies_temp

        # Correct for temperature
        gamma = -0.43/100
        T_nom = 25 + 273.15
        proxies_temp = proxies_temp*(0.0358+2e-3)
        T_bom = temperature+proxies_temp

        #correct the proxy
        proxies_temp = np.multiply(proxies_temp_orig,1+gamma*(T_bom-T_nom))
        proxies[:,i] = proxies_temp

    proxies = proxies
    estimated_PV = proxies.dot(alpha)
    estimated_PV = pd.DataFrame(index=GHI.index,data=estimated_PV,columns=['prod [W]'])
    return estimated_PV

def pv_production(scenario,time,ahead,tilt,azimuth,alpha,eta_tot):
    """
    this  function combine together the previous two
    """
    ghi,temp = DWD_forecast(scenario,ahead,time)
    ghi_15 = ghi.resample('15T').ffill()
    temp_15 = temp.resample('15T').ffill()
    P_15 = estimate_pv_production(ghi_15, temp_15, tilt, azimuth, alpha)*eta_tot
    P = P_15.resample('60T').ffill()
    P = P.rename(columns={'prod [W]':f"prod_{scenario} [W]"})
    return(P)







if __name__=="__main__": ########################################################################################################
    
    time = datetime.strptime('2024-06-01 00:00:00', '%Y-%m-%d %H:%M:%S')
   # time = datetime.strptime('2024-03-1 00:00:00', '%Y-%m-%d %H:%M:%S')

    pvname,tilt,azimuth,alpha,eta_tot = 'Perun',[10],[180],[13],0.70
    pvname,tilt,azimuth,alpha,eta_tot = 'Solarmax',[10],[180],[16],0.85
    pvname,tilt,azimuth,alpha,eta_tot = 'PVFacade',[90],[180],[14],0.85
    
    ahead = 24
    
    from pull_influx import pull_influx
    meteobox_ghi = pull_influx(time,ahead,'irradiance',pvname) 
    mesured_power = pull_influx(time,ahead,'power',pvname)
    
    if pvname == 'Solarmax':
        mesured_power += - pull_influx(time,ahead,'power','PVFacade')
    mesured_power = mesured_power.resample('60T').ffill()
    
    pro = pv_production(1,time,ahead,tilt,azimuth,alpha,eta_tot)
    print(pro)
    
    
    
#%%
    scenarios = {}
    for s in range(40):    
        
        ghi,temp = DWD_forecast(s,ahead,time)
        ghi_15 = ghi.resample('15T').ffill()
        temp_15 = temp.resample('15T').ffill()
        P_15 = estimate_pv_production(ghi_15, temp_15, tilt, azimuth, alpha)*eta_tot
        P = P_15.resample('60T').ffill()
        P_15_mb = estimate_pv_production(meteobox_ghi, temp_15, tilt, azimuth, alpha)*eta_tot
        P_mb = P_15_mb.resample('60T').ffill()
        P_mb.index = P.index
        P_mb.columns=(['prod2 [W]'])
        scenarios[s] = pd.concat([ghi,temp,P,P_mb], axis=1)
        

    fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,figsize=(18,9),dpi=1000)
    for s in scenarios:
        x = scenarios[s].index.hour
        ax2.plot(x,scenarios[s]['prod [W]'])
        ax1.plot(x,scenarios[s]['ghi [W/m2]'])
        ax3.plot(x,scenarios[s]['temp [K]'])
        ax4.plot(x,scenarios[s]['prod2 [W]'])
    ax1.set_ylabel('[W/m2]')
    ax1.grid()
    ax1.set_title('Irradiance dwd')
    ax2.grid()
    ax2.set_ylabel('kW')
    ax2.set_title('Active Power pvlib (dwd temp and dwd irr)')
    ax3.set_ylabel('[K]')
    ax3.grid()
    ax3.set_title('Temperature dwd')
    ax4.grid()
    ax4.set_ylabel('W')
    ax4.set_title('Active Power pvlib (dwd temp and meteobox irr)')
    
    meteobox_ghi = meteobox_ghi.resample('60T').ffill()
    ax5.plot(x,meteobox_ghi)
    ax6.plot(x,mesured_power)
    ax5.set_title('Irradiance meteobox')
    ax6.set_title('Active Power mesured')
    ax5.grid()
    ax6.grid()
    
    plt.suptitle(str(time)[:10])
    plt.tight_layout()

    plt.show()
    
    
        
    

    
    
    
        
    
    
    