"""
Skytte model

Price of Regulation ( spot Price, amount of Regulation)
PR(P,Reg) con Reg = S-D (amount announced on the sport market - actual delivery) = - unbalance
PR(P,Reg) = a0 P + 1(Reg>0) [ a1 P + a2 Reg + a3 ] + 1(Reg<0) [ a4 P + a5 Reg + a6 ]  

"""

import numpy as np
import pandas as pd
import Database_management as dbm
import Graphs as gra
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime, timedelta

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



#%%################################################################ preliminary analysis

# Graphs P 
start = '2023-01-09 00:00:00'
end = '2023-01-16 00:00:00'
gra.DAMvsRegulation(P,PR,Reg,start,end,24)

# Annual P vs REG vs 
start = '2023-01-01 00:00:00'
end = '2024-01-31 00:00:00'
gra.PvsPRvsReg(P,PR,Reg,start,end,24*31)

start1 = '2023-01-01 00:00:00'
end1 = '2023-07-1 00:00:00'
title1 = f"{start[:10]} to {end[:10]}"
gra.pdf(Reg[start1:end1],"Regulation [MWh]",title1)
gra.pdf(PR[start1:end1],"Regulation price [€/MWh]",title1)
gra.pdf(P[start1:end1],"Spot price [€/MWh]",title1)

start2 = '2023-08-01 00:00:00'
end2 = '2024-01-31 00:00:00'
title2 = f"{start[:10]} to {end[:10]}"
gra.pdf(Reg[start2:end2],"Regulation [MWh]",title2)
gra.pdf(PR[start2:end2],"Regulation price [€/MWh]",title2)
gra.pdf(P[start2:end2],"Spot price [€/MWh]",title2)



for whis in [1,2,3,4,5]:
    whis = (whis,100-whis)
    gra.duble_boxplot(PR[start1:end1],PR[start2:end2],title1,title2,whis)

#%%############################################################## datacleaning

# cutting Regulation Price
lc = np.percentile(PR,4) # lower cut
uc = np.percentile(PR,96) # upper cut
PR[PR>uc] = uc
PR[PR<lc] = lc

gra.duble_boxplot(PR[start1:end1],PR[start2:end2],title1,title2,whis)

#%%############################################## fitting and testing rolling horizon the model

def Skytte_model(x, c1, c2, c3, c4, c5, c6):
    P,Reg = x
    Reg_P = (Reg >= 0).astype(int)
    Reg_N = (Reg < 0).astype(int)
    PR = P + Reg_N * (c1 * P + c2 * Reg + c3) + Reg_P * (c4 * P + c6 * Reg + c5)
    return PR

def fitting_rolling_horizon(lookback,lookahead,start_test,end_test):
    start = start_test
    end = start + timedelta(hours=23) + timedelta(hours=int(24*(lookahead-1)))
    PR_fitted = np.empty((0,), dtype=float)
    while end <= end_test: 
        end_trial = start
        start_trial = end_trial - timedelta(days=lookback)
        coef, cov = curve_fit(Skytte_model, [P[start_trial:end_trial], Reg[start_trial:end_trial]], PR[start_trial:end_trial])
        
        PR_fitted = np.append(PR_fitted, Skytte_model((P[start:end], Reg[start:end]), *coef))
        start += timedelta(days=lookahead)
        end += timedelta(days=lookahead)
        
    R2 = round(r2_score(PR[start_test:end_test], PR_fitted),3)
    residuals = PR_fitted-PR[start_test:end_test]
    return(PR_fitted,residuals,R2)

# defining test data
# =============================================================================
# start_test = datetime.strptime('2023-01-09 00:00:00', '%Y-%m-%d %H:%M:%S')
# end_test = datetime.strptime('2023-01-15 23:00:00', '%Y-%m-%d %H:%M:%S')
# =============================================================================
# =============================================================================
# start_test = datetime.strptime('2023-02-01 00:00:00', '%Y-%m-%d %H:%M:%S')
# end_test = datetime.strptime('2023-06-30 23:00:00', '%Y-%m-%d %H:%M:%S')
# =============================================================================
# =============================================================================
# start_test = datetime.strptime('2024-01-08 00:00:00', '%Y-%m-%d %H:%M:%S')
# end_test = datetime.strptime('2024-01-14 23:00:00', '%Y-%m-%d %H:%M:%S')
# =============================================================================
# =============================================================================
# start_test = datetime.strptime('2023-07-01 00:00:00', '%Y-%m-%d %H:%M:%S')
# end_test = datetime.strptime('2024-01-31 23:00:00', '%Y-%m-%d %H:%M:%S')
# =============================================================================

# =============================================================================
# start_test = datetime.strptime('2023-10-01 00:00:00', '%Y-%m-%d %H:%M:%S')
# end_test = datetime.strptime('2024-01-28 23:00:00', '%Y-%m-%d %H:%M:%S')
# =============================================================================
start_test = datetime.strptime('2023-12-10 00:00:00', '%Y-%m-%d %H:%M:%S')
end_test = datetime.strptime('2023-12-16 23:00:00', '%Y-%m-%d %H:%M:%S')


PR_fitted,residuals,R2 = fitting_rolling_horizon(6,1,start_test,end_test)

# Graphs PR vs PR* vs Regulation
gra.PRvsPRfitted(PR,PR_fitted,Reg,start_test,end_test)
gra.pdf(residuals,"Residuals [€/MWh]",f"R\u00B2 = {R2}")


#%%###################################################### model statistics (only one model)

# =============================================================================
# start = datetime.strptime('2023-01-09 00:00:00', '%Y-%m-%d %H:%M:%S')
# end = datetime.strptime('2023-01-15 23:00:00', '%Y-%m-%d %H:%M:%S')
# =============================================================================
# =============================================================================
# start = datetime.strptime('2024-01-08 00:00:00', '%Y-%m-%d %H:%M:%S')
# end = datetime.strptime('2024-01-14 23:00:00', '%Y-%m-%d %H:%M:%S')
# =============================================================================
start = datetime.strptime('2023-12-10 00:00:00', '%Y-%m-%d %H:%M:%S')
end = datetime.strptime('2023-12-16 23:00:00', '%Y-%m-%d %H:%M:%S')

coef, cov = curve_fit(Skytte_model, [P[start:end], Reg[start:end]], PR[start:end])
SE = np.sqrt(np.diag(cov))
t_value = coef / SE
model = pd.DataFrame(np.column_stack((t_value,SE,coef)), columns=['t-value','S.E.','Value'])
model = pd.concat([pd.DataFrame({'t-value': ['-'], 'S.E.': ['-'], 'Value': [1]}),model])
model.index=['c0','c1','c2','c3','c4','c5','c6']
print(model.round(3))

# Graphs amount of regulation
P0 = 200
Reg_range = np.arange(-1000,1000,10)
PR_fitted2 = Skytte_model([P0, Reg_range], *coef)
gra.regulating_power_prirce(Reg_range,PR_fitted2,P0)

# Graphs premium readiness
P_range = np.arange(50,400,10)
Premium_down = -model.loc['c1','Value']*P_range - model.loc['c3','Value']
Premium_up = model.loc['c4','Value']*P_range + model.loc['c5','Value']
gra.premium_readiness(P_range,Premium_down,Premium_up)



#%%############################################################## sensitivety analysis

start_test = datetime.strptime('2023-02-01 00:00:00', '%Y-%m-%d %H:%M:%S')
end_test = datetime.strptime('2023-06-30 23:00:00', '%Y-%m-%d %H:%M:%S')
start_test = datetime.strptime('2023-10-01 00:00:00', '%Y-%m-%d %H:%M:%S')
end_test = datetime.strptime('2024-01-28 23:00:00', '%Y-%m-%d %H:%M:%S')

R22 = {}

lookback = np.arange(3,60)
lookahead = np.arange(1,3)

for la in lookahead:
    R22[la] = []
    for lb in lookback:
        PR_fitted,residuals,R2 = fitting_rolling_horizon(int(lb),int(la),start_test,end_test)
        R22[la].append(R2)
    
gra.R2vsDays(R22,lookback,'Sensitivety analysis 01/10/23 - 28/01/24')







