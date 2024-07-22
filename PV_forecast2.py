"""
PV forecast of Perun, PVFacade and Solarmax based on dwd ghi forecast + empirical model
"""

import numpy as np
import pandas as pd
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def empirical_model(filename):
    ### need as input a file.csv with two columns
    # ghi [W/m2]' and 'power [W]'
    # a model to predict power from ghi will be created
    
    pv = pd.read_csv(f"input/{filename}")
    X = pv['ghi [W/m2]'].values.reshape(-1, 1)
    y = pv['power [W]'].values
    poly = PolynomialFeatures(3)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    
    return([poly,model])
    

def DWD_forecast2(scenario,ahead,time):
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
    Global Horizontal Irradiance 

    """
    
    pl = 6 # puling lag
    
    time += -timedelta(hours=1) # utc -> uct+1
    day = str(time)[:10]
    
    ### remember that dwd databaase is UTC and not UTC+1
    if 0+pl <= time.hour < 6+pl:
        forecast_time = '00'
        lag = time.hour-0
    elif 6+pl <= time.hour < 12+pl:
        forecast_time = '06'
        lag = time.hour-6
    elif 12+pl <= time.hour < 18+pl:
        forecast_time = '12'
        lag = time.hour-12
    elif 18+pl <= time.hour < 24:
        forecast_time = '18'
        lag = time.hour-18
    else:
        forecast_time = '18'
        day = str(time-timedelta(hours=24))[:10]
        lag = time.hour+24-18
       
    irr = np.load(f"input\\DWD_irradiance\\Location_Lausanne_time_of_forecast_{day}T{forecast_time}.npy")
    #irr = np.load(f"C:\\Users\\pasqui\\Desktop\\GIT\\Comunity-battery-EA-CSC-AS-DIS\\DWD_irradiance\\Location_Lausanne_time_of_forecast_{day}T{forecast_time}.npy")
    #irr = np.load(f"C:\\Users\\admin\\Desktop\\Comunity-battery-EA-CSC-AS-DIS\\DWD_irradiance\\Location_Lausanne_time_of_forecast_{day}T{forecast_time}.npy")
    irr = irr[scenario,lag:ahead+lag]
    index_vector = []
    current_time = time
    while current_time < time + timedelta(hours=ahead):
        index_vector.append(current_time)
        current_time += timedelta(hours=1)
        
    ghi = pd.DataFrame({'ghi [W/m2]':irr})
    ghi.index = index_vector
    ghi.index = ghi.index.shift(freq='1H') # utc -> uct+1
    
    return(ghi)


def pv_production2(scenario,time,ahead,model):
    """
    this  function combine together the previous two functions
    dwd (ghi) + empirical model (power from ghi) --> power
    """
    
    ghi = DWD_forecast2(scenario,ahead,time)
    predicted_power = []
    for ghi_value in ghi['ghi [W/m2]']:
        ghi_value_poly = model[0].transform([[ghi_value]])
        predicted_power.append(model[1].predict(ghi_value_poly)[0])
    P = pd.DataFrame(predicted_power, index=ghi.index, columns=[f"prod_{scenario} [W]"])
    
    return(P)

def pv_production3(ghi,model,scenario):
    """
    this  function combine together the previous two functions
    dwd (ghi) + empirical model (power from ghi) --> power
    """
    
    predicted_power = []
    for ghi_value in ghi['ghi [W/m2]']:
        ghi_value_poly = model[0].transform([[ghi_value]])
        predicted_power.append(model[1].predict(ghi_value_poly)[0])
    P = pd.DataFrame(predicted_power, index=ghi.index, columns=[f"prod_{scenario} [W]"])
    P.index = pd.to_datetime(P.index, format='%Y-%m-%d %H-%M-%S')
    
    return(P)






if __name__=="__main__": ########################################################################################################

    # test to obtain one scenario of power from dwd + empirical model
    pvname = 'Perun'
    timenow = datetime.strptime('2024-06-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    scenario = 5
    ahead = 24
    model = empirical_model(f"PV_{pvname} realisation_h.csv")
    p = pv_production2(scenario, timenow, ahead, model)
    print(p)
      
    
    
    # to see the model
    PVs= {'Perun': pd.DataFrame(columns=["power [W]","ghi [W/m2]"]),
          'PVFacade': pd.DataFrame(columns=["power [W]","ghi [W/m2]"]),
          'Solarmax': pd.DataFrame(columns=["power [W]","ghi [W/m2]"])}
    
    end_time = datetime.strptime('2024-06-28 00:00:00', '%Y-%m-%d %H:%M:%S')
    
    
    for pvname in PVs:
        
        ####### scomment to pull new data updated from influx and save them to fit new updated models, chose end_time and lookback before to do it
        from pull_influx import pull_influx
        lookback = 370 # days
        timenow = end_time - timedelta(hours=24*lookback)
        while timenow <= end_time:
            try:
                ghi = pull_influx(timenow,24,'irradiance','Perun') 
                P = pull_influx(timenow,24,'power',pvname)
                if pvname == 'Solarmax':
                    P = P-pull_influx(timenow,24,'power','PVFacade')
                day = pd.concat([P,ghi], axis=1)
                PVs[pvname] = pd.concat([PVs[pvname],day], axis=0)
            except:
                pass
            timenow += timedelta(hours=24)
        
        # leva gli errori di misura
        PVs[pvname] = PVs[pvname].dropna()
        PVs[pvname] = PVs[pvname][(PVs[pvname]['power [W]'] <= 15000) & (PVs[pvname]['power [W]'] >= -500) &  (PVs[pvname]['ghi [W/m2]'] >= 0) & (PVs[pvname]['ghi [W/m2]'] <= 1000)]
        PVs[pvname].index = pd.to_datetime(PVs[pvname].index)
        PVs[pvname].to_csv(f"PV_{pvname} realisation.csv")
        PVs[pvname] = PVs[pvname].resample('H').mean()
        PVs[pvname] = PVs[pvname].dropna()
        PVs[pvname].to_csv(f"PV_{pvname} realisation_h.csv")
        #######################################################################################################
        
        
        ### models creation and plotting

        PVs[pvname] = pd.read_csv(f"PV_{pvname} realisation_h.csv")
        X = PVs[pvname]['ghi [W/m2]'].values.reshape(-1, 1)
        y = PVs[pvname]['power [W]'].values
        poly = PolynomialFeatures(3)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        
        fig, (ax1) = plt.subplots(1,1,dpi=1000)
        ax1.scatter(PVs[pvname]['ghi [W/m2]'],PVs[pvname]['power [W]'],alpha=1,s=2)
        
        sorted_indices = np.argsort(X.flatten())
        ax1.plot(X[sorted_indices], y_pred[sorted_indices], color='red', label='Fitted curve')
        ax1.grid()
        ax1.set_xlabel('GHI [W/m2]')
        ax1.set_ylabel('Active Power [W]')
        ax1.set_xlim(0,1000)
        ax1.set_ylim(-500,13000)
        plt.title(f"{pvname} empirical model last 60 days")
        plt.show()
    


