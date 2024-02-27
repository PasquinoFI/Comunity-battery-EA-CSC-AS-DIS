"""
S-D = Reg used in the Skytte models has to be estimated because we know it only with 15/30' delay

let's investigate which data we have from Terna
othwerise use an ARIMA
"""


import numpy as np
import pandas as pd
import Database_management as dbm
import Graphs as gra
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats

#%%##################################################################### import series 

# Unbalances: PR and Reg
folder = "C:/Users/pasqui/Desktop/Sbilanciamenti"
unbalances = dbm.concat_xlsx_Terna(folder,2023,1,2024,1,"Riepilogo_Mensile_Orario")
unbalances = unbalances.loc[unbalances['Macrozona']=='SUD']
PR = unbalances['Prezzo di sbilanciamento']
Reg = -unbalances['Sbil aggregato zonale [MWh]']

# DAM: P
folder = "C:/Users/pasqui/Desktop/MercatiElettrici/MGP_Prezzi"
dam = dbm.concat_xml_GME(folder,2023,1,1,2024,1,31,"MGPPrezzi")
P = pd.to_numeric(dam['CNOR'].str.replace(',', '.'), errors='coerce').round(3)

# ora legale di merdaaaa
PR = PR.reindex(P.index)
Reg = Reg.reindex(P.index)

PR_nan = np.where(np.isnan(PR))[0]
for i in PR_nan:
        PR[i] = (PR[i+1] + PR[i-1])/2
Reg_nan = np.where(np.isnan(Reg))[0]
for i in Reg_nan:
        Reg[i] = (Reg[i+1] + Reg[i-1])/2
        
#%%############################################################## datacleaning

# cutting Regulation Price
lc = np.percentile(PR,4) # lower cut
uc = np.percentile(PR,96) # upper cut
PR[PR>uc] = uc
PR[PR<lc] = lc

        
        
#%% ################################################################# ARIMA


#################### STATIONARITY AND AUTOCORRELATION ?!

serie = Reg
# =============================================================================
# gra.rolling_reg_tot(serie,24,"Daily window")
# gra.rolling_reg_tot(serie,7*24,"Weekly window")
# gra.rolling_reg_tot(serie,30*24,"Monthly window")
# =============================================================================
gra.acf_pacf(serie,24*3,6,"One year")

start = datetime.strptime('2023-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
end = datetime.strptime('2023-01-31 23:00:00', '%Y-%m-%d %H:%M:%S')
serie = Reg[start:end]
# =============================================================================
# gra.rolling_reg_m(serie,24*7,24*7,"January 24 - weekly window")
# gra.rolling_reg_m(serie,24,24*7,"January 24 - daily window")
# gra.rolling_reg_m(serie,26,24*7,"January 24 - 6 hours window")
# =============================================================================
gra.acf_pacf(serie,24*3,6,"January 24")

start = datetime.strptime('2024-01-08 00:00:00', '%Y-%m-%d %H:%M:%S')
end = datetime.strptime('2024-01-14 23:00:00', '%Y-%m-%d %H:%M:%S')
serie = Reg[start:end]
# =============================================================================
# gra.rolling_reg_m(serie,24,24,"2th week of Januaray - daily window")
# gra.rolling_reg_m(serie,6,24,"2th week of Januaray - 6 hours window")
# =============================================================================


gra.acf_pacf(serie,24*3,6,"2th week of Januaray")
result = adfuller(serie)
print('ADF value:', result[0])
print('P-value:', result[1])
print('Lags:', result[2])
print('Observation:', result[3])
print('Critical values:', result[4])





#%%########################################################## fitting ARIMA

model = ARIMA(serie, order=(4, 0, 6)).fit()
model.summary()
gra.pdf(model.resid,"Residuals [MWh]","Residuals distribution")


#%%################################## forecast vs test

start_trial = datetime.strptime('2023-12-15 00:00:00', '%Y-%m-%d %H:%M:%S')
#start_trial = datetime.strptime('2024-01-08 00:00:00', '%Y-%m-%d %H:%M:%S')
end_trial = datetime.strptime('2024-01-15 23:00:00', '%Y-%m-%d %H:%M:%S')
serie_trial = Reg[start_trial:end_trial]
start_test = datetime.strptime('2024-01-15 00:00:00', '%Y-%m-%d %H:%M:%S')
end_test = datetime.strptime('2024-01-15 23:00:00', '%Y-%m-%d %H:%M:%S')
serie_test = Reg[start_test:end_test]

model = ARIMA(serie_trial, order=(4, 0, 6)).fit()
model.summary()
gra.pdf(model.resid,"Residuals [MWh]","Residuals distribution")

forecast = model.predict(start=(start_test+timedelta(hours=1)),end=(end_test+timedelta(hours=1)))

gra.ARIMAvsReal(serie_test,forecast,6,"1W trial - 1D test")

mae = round(mean_absolute_error(serie_test, forecast),2)
rmse = round(np.sqrt(mean_squared_error(serie_test, forecast)),2)
mape = round(np.mean(np.abs((serie_test - forecast) / serie_test)),2)
rmspe = round(np.sqrt(np.mean(((serie_test - forecast) / serie_test) ** 2)),2)
print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE:", mape)
print("RMSPE:", rmspe)

errors = forecast.values - serie_test.values
benchmark = serie_trial.mean()
errors_benchmark = benchmark - serie_test

# Eseguire il test di Mincer-Zarnowitz
t_stat, p_value = stats.ttest_rel(errors, errors_benchmark)

# Stampare il risultato del test
print("Mincer-Zarnowitz:")
print("T value:", round(t_stat,4))
print("P-value:", round(p_value,4))











