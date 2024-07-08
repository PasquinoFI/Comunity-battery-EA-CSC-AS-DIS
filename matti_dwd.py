# -*- coding: utf-8 -*-
"""
sending dwd forecast to Mattia pc
irradiance and temperature
every 6 hours
evenif at the moment they are updated only ones perday
"""

import socket, json, time
import numpy as np
import pandas as pd
from datetime import datetime,timedelta

def DWD_forecast(ahead,time_forecast):
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
    
    pl = 6 # pulling lag
    
    time_forecast += -timedelta(hours=1) # utc -> uct+1  ### remember that dwd databaase is UTC and not UTC+1
    day = str(time_forecast)[:10]
    
    # per prendere quella piu aggiornata (vengono scaricate ogni 6 ore ma con un pulling lag)
    if 0+pl <= time_forecast.hour < 6+pl:
        #print("00")
        forecast_time = '00'
        lag = time_forecast.hour-0
        # day is correct
    elif 6+pl <= time_forecast.hour < 12+pl:
        #print("06")
        forecast_time = '06'
        lag = time_forecast.hour-6
        # day is correct
    elif 12+pl <= time_forecast.hour < 18+pl:
        #print("12")
        forecast_time = '12'
        lag = time_forecast.hour-12
        # day is correct
        
    elif 18+pl <= time_forecast.hour < 24:
        #print("18")
        forecast_time = '18'
        lag = time_forecast.hour-18
        # day is correct
    else:
        #print("18 d-1")
        forecast_time = '18'
        day = str(time_forecast-timedelta(hours=24))[:10]
        lag = time_forecast.hour+24-18
        
        
    ## per prendere sempre quella delle 18 del giorno prima
# =============================================================================
#     day = str(time_forecast-timedelta(hours=24))[:10]
#     forecast_time = '18'
#     lag = time_forecast.hour+24-18
# =============================================================================
    ##################################################################################################3
    
    
    irr = np.load(f"/home/ubuntu/Documents/code/python/dwd_weather/irradiance_data/Location_Lausanne_time_of_forecast_{day}T{forecast_time}.npy")
    df = {}
    for s in range(len(irr)):
        irr_s = irr[s,lag:ahead+lag]
        index_vector = []
        current_time = time_forecast
        while current_time < time_forecast + timedelta(hours=ahead):
            index_vector.append(current_time)
            current_time += timedelta(hours=1)
        ghi = pd.DataFrame({'ghi [W/m2]':irr_s})
        ghi.index = index_vector
        ghi.index = ghi.index.shift(freq='1H') # utc -> uct+1
        ghi.index = ghi.index.strftime('%Y-%m-%d %H-%M-%S')
        df[s] = ghi['ghi [W/m2]'].to_dict()
    return(df)



# test di invio
ip_to, port_to = '128.178.46.54', 35211  # Matti
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:
    time_forecast = ((datetime.fromtimestamp(time.time()))+timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    ahead=24
    ghi = DWD_forecast(ahead,time_forecast)
    json_string = json.dumps(ghi)
    #json_string = 'ciao bellaaaa'
    sock.sendto(json_string.encode('utf-8'), (ip_to, port_to))



# test
# =============================================================================
# for h in range(30):
#     time_forecast = ((datetime.fromtimestamp(time.time()))+timedelta(hours=-h)).replace(minute=0, second=0, microsecond=0)
#     print(time_forecast)
#     ahead=24
#     ghi = DWD_forecast(ahead,time_forecast)
#     #print(ghi[0])
#     print('\n')
# =============================================================================
