"""
S-D = Reg used in the Skytte models has to be estimated because we know it only with 30' delay
"""

#%%
import numpy as np
import Database_management as dbm
import Graphs as gra
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from scipy.stats import norm


#%%############################################################### fit and forecast 

def RegF_arima(serie_trial,serie_test,lookahead,alpha):
    """
    
    Parameters
    ----------
    Reg : trial serie
    ar : Auto Regresive order
    i : Integrated order
    ma : Moving Average order
    lookahead : forecast horizon length

    Returns
    -------
    forecasts

    """
    
    serie_trial.index.freq = 'H'
    
    # model order selection
    pacfv = pacf(serie_trial)
    acfv = acf(serie_trial)
    ic_up = norm.ppf(1-alpha/2) * 1/np.sqrt(len(serie_trial))
    ar = (abs(pacfv)>ic_up).sum()-1
    ma = (abs(acfv)>ic_up).sum()-1

    model = ARIMA(serie_trial, order=(ar, 0, ma)).fit()
    forecasts = model.forecast(steps=lookahead)
    
    ####################################################### arima testing and graphs
    print(model.summary())
    gra.acf_pacf(serie_trial,alpha,"")
    gra.pdf(model.resid,"Residuals [MWh]","Residuals distribution")
    gra.ARIMAvsRealvsPre(serie_trial[-24:],serie_test,forecasts,lookahead,"")
        
    return(forecasts)
    
def RegF_sarima(serie_trial,serie_test,lookahead,alpha):
    """
    
    Parameters
    ----------
    Reg : trial serie
    ar : Auto Regresive order
    i : Integrated order
    ma : Moving Average order
    lookahead : forecast horizon length

    Returns
    -------
    forecasts

    """
    
    serie_trial.index.freq = 'H'
    
    # model order selection
    pacfv = pacf(serie_trial)
    acfv = acf(serie_trial)
    ic_up = norm.ppf(1-alpha/2) * 1/np.sqrt(len(serie_trial))
    ar = (abs(pacfv)>ic_up).sum()-1
    ma = (abs(acfv)>ic_up).sum()-1

    model = SARIMAX(serie_trial, order=(ar, 0, ma), seasonal_order=(0, 0, 1, 24)).fit()
    forecasts = model.forecast(steps=lookahead)
    
    ####################################################### arima testing and graphs
    #print(model.summary())
    #gra.acf_pacf(serie_trial,alpha,"")
    #gra.pdf(model.resid,"Residuals [MWh]","Residuals distribution")
    #gra.ARIMAvsRealvsPre(serie_trial[-24:],serie_test,forecasts,lookahead,"")
        
    return(forecasts)

def sign_arima(serie_trial,serie_test,lookahead,alpha,ar):
    """
    
    Parameters
    ----------
    Reg : trial serie
    ar : Auto Regresive order
    i : Integrated order
    ma : Moving Average order
    lookahead : forecast horizon length

    Returns
    -------
    forecasts

    """
    
    serie_trial.index.freq = 'H'

    model = ARIMA(serie_trial, order=(ar, 0, 0)).fit()
    forecasts = model.forecast(steps=lookahead)
    
    ####################################################### arima testing and graphs
    #print(model.summary())
    #gra.ARIMAvsRealvsPre(serie_trial[-24:],serie_test,forecasts,lookahead,"")
        
    return(forecasts)


#%%##################################################################### import and clean serie

if __name__ == "__main__":
    
    # Unbalances: PR and Reg
    folder = "C:/Users/pasqui/Desktop/Sbilanciamenti"
    unbalances = dbm.concat_xlsx_Terna(folder,2023,1,2024,1,"Riepilogo_Mensile_Orario")
    unbalances = unbalances.loc[unbalances['Macrozona']=='SUD']
    Reg = -unbalances['Sbil aggregato zonale [MWh]']
    
    # Datacleaning 
    Reg = dbm.cut_smart(Reg,-1000,1000)
            
    #%% ############################################################ STATIONARITY AND AUTOCORRELATION ?!
    
    serie = Reg
    gra.rolling_reg_tot(serie,24,"Daily window")
    gra.rolling_reg_tot(serie,7*24,"Weekly window")
    gra.rolling_reg_tot(serie,30*24,"Monthly window")
    gra.acf_pacf(serie,0.01,"One year")
    
    start = datetime.strptime('2023-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    end = datetime.strptime('2023-01-31 23:00:00', '%Y-%m-%d %H:%M:%S')
    serie = Reg[start:end]
    gra.rolling_reg_m(serie,24*7,24*7,"January 24 - weekly window")
    gra.rolling_reg_m(serie,24,24*7,"January 24 - daily window")
    gra.rolling_reg_m(serie,26,24*7,"January 24 - 6 hours window")
    gra.acf_pacf(serie,0.01,"January 24")
    
    start = datetime.strptime('2024-01-08 00:00:00', '%Y-%m-%d %H:%M:%S')
    end = datetime.strptime('2024-01-14 23:00:00', '%Y-%m-%d %H:%M:%S')
    serie = Reg[start:end]
    gra.rolling_reg_m(serie,24,24,"2th week of Januaray - daily window")
    gra.rolling_reg_m(serie,6,24,"2th week of Januaray - 6 hours window")
    
    gra.acf_pacf(serie,0.01,"2th week of Januaray")
    result = adfuller(serie)
    print('ADF value:', result[0])
    print('P-value:', result[1])
    print('Lags:', result[2])
    print('Observation:', result[3])
    print('Critical values:', result[4])







