#%%
import numpy as np
import pandas as pd
import Database_management as dbm
import Graphs as gra
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from datetime import datetime, timedelta
from Reg_arima import RegF_arima

#%%##################################################################### import series 

# DAM: P
folder = "C:/Users/pasqui/Desktop/MercatiElettrici/MGP_Prezzi"
dam = dbm.concat_xml_GME(folder,2023,1,1,2024,1,31,"MGPPrezzi")
P = pd.to_numeric(dam['CNOR'].str.replace(',', '.'), errors='coerce').round(3)

# Unbalances: PR and Reg
folder = "C:/Users/pasqui/Desktop/Sbilanciamenti"
unbalances = dbm.concat_xlsx_Terna(folder,2023,1,2024,1,"Riepilogo_Mensile_Orario")
unbalances = unbalances.loc[unbalances['Macrozona']=='SUD']
PR = unbalances['Prezzo di sbilanciamento']
Reg = -unbalances['Sbil aggregato zonale [MWh]']

# Datacleaning 
Reg = dbm.cut(Reg,5)
PR = dbm.cut(PR,5)
P = dbm.cut(P,5)


#%%################################################################ preliminary analysis

# Graphs P 
start = '2023-01-09 00:00:00'
end = '2023-01-16 00:00:00'
gra.DAMvsRegulation(P,PR,Reg,start,end,24)

# Annual P vs REG vs 
start = '2023-01-01 00:00:00'
end = '2024-01-31 00:00:00'
gra.PvsPRvsReg(P,PR,Reg,start,end,24*31)

title = f"{start[:10]} to {end[:10]}"
gra.pdf(Reg[start:end],"Regulation [MWh]",title)
gra.pdf(PR[start:end],"Regulation price [€/MWh]",title)
gra.pdf(P[start:end],"Spot price [€/MWh]",title)


# =============================================================================
# start1 = '2023-01-01 00:00:00'
# end1 = '2023-07-1 00:00:00'
# title1 = f"{start1[:10]} to {end1[:10]}"
# gra.pdf(Reg[start1:end1],"Regulation [MWh]",title1)
# gra.pdf(PR[start1:end1],"Regulation price [€/MWh]",title1)
# gra.pdf(P[start1:end1],"Spot price [€/MWh]",title1)
# =============================================================================

# =============================================================================
# start2 = '2023-08-01 00:00:00'
# end2 = '2024-01-31 00:00:00'
# title2 = f"{start2[:10]} to {end2[:10]}"
# gra.pdf(Reg[start2:end2],"Regulation [MWh]",title2)
# gra.pdf(PR[start2:end2],"Regulation price [€/MWh]",title2)
# gra.pdf(P[start2:end2],"Spot price [€/MWh]",title2)
# =============================================================================

# =============================================================================
# for whis in [1,2,3,4,5]:
#     whis = (whis,100-whis)
#     gra.duble_boxplot(PR[start1:end1],PR[start2:end2],title1,title2,whis)
# =============================================================================

#%%############################################## fitting and testing rolling horizon the model

def Skytte_model(x, c1, c2, c3, c4, c5, c6):
    """
    Price of Regulation (spot Price, amount of Regulation)
    PR(P,Reg) con Reg = S-D (amount announced on the sport market - actual delivery) = - unbalance 
    """
    P,Reg = x
    Reg_P = (Reg >= 0).astype(int)
    Reg_N = (Reg < 0).astype(int)
    PR = P + Reg_N * (c1 * P + c2 * Reg + c3) + Reg_P * (c4 * P + c6 * Reg + c5)
    return PR

def Skytte_rolling_horizon(rollingstep,lookback,lookahead,start_sim,end_sim,arima=False):
    """
    rollingstep: int [hours] frequency of fitting repetition
    lookback: int [hours] length of trial series
    lookahead: int [hours] length of forecast and test series
    start_sim: datetime.strptime('aaaa-mm-gg hh:mm:ss', '%Y-%m-%d %H:%M:%S')
    end_sim: datetime.strptime('aaaa-mm-gg hh:mm:ss', '%Y-%m-%d %H:%M:%S')
    arima: boll 
        if False PRF (the forecast of PR) is forecasted based on P and Reg
        if True PRF is forecasted based on P and RefF, which is an arima forecast of Reg
    """

    # initialise results database
    date_list = []
    date = start_sim
    while date <= end_sim:
        date_list.append(date)
        date += timedelta(hours=rollingstep)
    results = pd.DataFrame(index=date_list)  
    
    # fitting, forecast and save results!    
    for start_test in results.index:    
        end_test = start_test + timedelta(hours=lookahead-1)
        end_trial = start_test - timedelta(hours=1)
        start_trial = end_trial - timedelta(hours=lookback)
        
        # PR(P,Reg) Skytte fitting        
        coef, cov = curve_fit(Skytte_model, [P[start_trial:end_trial], Reg[start_trial:end_trial]], PR[start_trial:end_trial])
        
        # Reg(Reg-1) arima fitting
        if arima:
            RegF = RegF_arima(Reg[start_trial:end_trial],Reg[start_test:end_test],lookahead,alpha=0.01)
        else:        
            RegF = Reg[start_test:end_test]
            
        PRF = Skytte_model((P[start_test:end_test], RegF), *coef)
            
        for h in range(lookahead):
            results.at[start_test,f"PR_{h}"] = PR[start_test+timedelta(hours=h)]
            results.at[start_test,f"PRF_{h}"] = PRF[h]
            results.at[start_test,f"Reg_{h}"] = Reg[start_test+timedelta(hours=h)]
            results.at[start_test,f"RegF_{h}"] = RegF[h]
        
    return(results)

# simulations
start_sim = datetime.strptime('2023-02-10 00:00:00', '%Y-%m-%d %H:%M:%S')
end_sim = datetime.strptime('2023-02-12 00:00:00', '%Y-%m-%d %H:%M:%S')

rollingstep = 1
lookback = 24*7
lookahead = 3

results = Skytte_rolling_horizon(rollingstep,lookback,lookahead,start_sim,end_sim)
results = Skytte_rolling_horizon(rollingstep,lookback,lookahead,start_sim,end_sim,arima=True)

# Graphs PR vs PR* vs Regulation

for la in range(lookahead):
    gra.PRvsPRfitted(results[f"PR_{la}"],results[f"PRF_{la}"],results[f"Reg_{la}"],results[f"RegF_{la}"],start_sim,end_sim,la)
    
    # PR
    R2 = r2_score(results[f"PR_{la}"], results[f"PRF_{la}"]).round(3)
    residuals = (results[f"PRF_{la}"]-results[f"PR_{la}"]) / max(PR)
    gra.pdf(residuals,"Residuals / max(PR)",f"PR forecast {la} R\u00B2 = {R2}")
    
    # Reg
    R2 = r2_score(results[f"RegF_{la}"], results[f"Reg_{la}"]).round(3)
    residuals = (results[f"RegF_{la}"]-results[f"Reg_{la}"]) / max(Reg)
    gra.pdf(residuals,"Residuals / max(Reg)",f"Reg forecast {la} R\u00B2 = {R2}")    
    

#%%###################################################### model statistics (only one model)

start = datetime.strptime('2023-02-10 00:00:00', '%Y-%m-%d %H:%M:%S')
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
start_sim = datetime.strptime('2023-03-01 22:00:00', '%Y-%m-%d %H:%M:%S')
end_sim = datetime.strptime('2024-01-25 03:00:00', '%Y-%m-%d %H:%M:%S')

rollingstep = 1
lookahead = 6
R22 = {}

lookback = np.arange(3*24,40*24,24*2)
for lb in lookback:
    R22[lb] = []
    results = Skytte_rolling_horizon(rollingstep,int(lb),lookahead,start_sim,end_sim)
    for la in range(lookahead):
        R2 = r2_score(results[f"PR_{la}"], results[f"PRF_{la}"]).round(3)
        R22[lb].append(R2)
    
gra.R2vsDays(R22,lookback,lookahead,'Sensitivety analysis 01/10/23 - 28/01/24')







