"""
PR distribution clustering
"""

#%%

import numpy as np
import pandas as pd
import Database_management as dbm
import Graphs as gra
import matplotlib.pyplot as plt


#%% 

# Unbalances: PR and Reg
folder = "C:/Users/pasqui/Desktop/Sbilanciamenti"
unbalances = dbm.concat_xlsx_Terna(folder,2023,1,2023,12,"Riepilogo_Mensile_Orario")
unbalances = unbalances.loc[unbalances['Macrozona']=='SUD']
df = pd.DataFrame(columns=['PR'], index=unbalances.index)
df['PR'] = unbalances['Prezzo di sbilanciamento']
df['PR'] = dbm.cut_smart(df['PR'],0,350)
folder = "C:/Users/pasqui/Desktop/MercatiElettrici/MGP_Prezzi"
dam = dbm.concat_xml_GME(folder,2023,1,1,2024,1,31,"MGPPrezzi")
df['P'] = pd.to_numeric(dam['CNOR'].str.replace(',', '.'), errors='coerce').round(3)
df['P-PR'] = df['P']-df['PR']
df['hour'] = df.index.hour
df['weekday'] = df.index.weekday
festivities_2023 = ['2023-01-01','2023-01-06', '2023-04-09','2023-04-10','2023-04-25','2023-05-01', '2023-06-02', '2023-08-15', '2023-11-01', '2023-11-02', '2023-12-08', '2023-12-25',  '2023-12-26' ]
df.loc[df.index.isin(festivities_2023), 'weekday'] = 6
df['month'] = df.index.month
map_season = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4, 12: 1}
df['season'] = df['month'].map(map_season)
df['sign'] = df['P-PR'].apply(lambda x: 1 if x <= 0 else -1)
df['P-PR up'] = np.where(df['sign'] == 1, df['P-PR'], np.nan)
df['P-PR down'] = np.where(df['sign'] == -1, df['P-PR'], np.nan)

df['sign-1'] = df['sign'].shift(periods=1)
df['sign-2'] = df['sign'].shift(periods=2)
df['sign-3'] = df['sign'].shift(periods=3)
df['sign-4'] = df['sign'].shift(periods=4)
df['sign-5'] = df['sign'].shift(periods=5)
df['sign-6'] = df['sign'].shift(periods=6)

df.describe()
df = df.drop('PR',axis=1)
df = df.drop('P',axis=1)
corr = df.corr()

gra.pdf(df['P-PR'],'P-PR [€/MWh]','')
gra.pdf(abs(df['P-PR']),'abs(P-PR) [€/MWh]','')
gra.pdf(df['P-PR up'],'up [€/MWh]','')
gra.pdf(df['P-PR down'],'down [€/MWh]','')

print(df['P-PR up'].mean())
print(df['P-PR down'].mean())

print((df['sign']>0).sum())
print((df['sign']<0).sum())
print(corr)

(df['sign'] == df['sign-1']).sum() / 8760
(df['sign'] == df['sign-2']).sum() / 8760
(df['sign'] == df['sign-3']).sum() / 8760
(df['sign'] == df['sign-4']).sum() / 8760
(df['sign'] == df['sign-5']).sum() / 8760
(df['sign'] == df['sign-6']).sum() / 8760


#%% scatterplots !!!

regressors = ['hour','month','season','weekday']

for r in regressors:
    
    plt.figure(dpi=1000)
    plt.scatter(df[r],df['P-PR'],s=2)
    plt.xlabel(r)
    plt.ylabel('P-PR [€/MWh]')
    plt.grid(axis='y')
    plt.show()
    
#%% boxplots !!!

for r in regressors:
    fig, ax = plt.subplots(dpi=1000)
    num_valori_r = len(df[r].unique())
    spaziatura = 1.5
    medie_boxplot = []  # Lista per memorizzare le medie dei boxplot
    for i, v in enumerate(df[r].unique()):       
        posizione = (i + 1) * spaziatura  # Calcola la posizione del boxplot
        boxplot_data = df[df[r]==v]['P-PR']
        ax.boxplot(boxplot_data, positions=[posizione], widths=0.5)
        # Calcola la media del boxplot corrente e aggiungila alla lista delle medie
        medie_boxplot.append(np.mean(boxplot_data))
    ax.grid(True)
    ax.set_xticks([(i + 1) * spaziatura for i in range(num_valori_r)])  # Imposta i ticks dell'asse x
    ax.set_xticklabels(df[r].unique())  # Etichetta gli ticks dell'asse x con i valori di 'r'
    ax.set_xlabel(r)  # Etichetta dell'asse x
    ax.set_ylabel("P-PR [€/MWh]")  # Etichetta dell'asse y
    # Traccia la linea continua che congiunge i valori medi dei boxplot
    plt.plot([(i + 1) * spaziatura for i in range(num_valori_r)], medie_boxplot, color='r', linestyle='-')
    plt.tight_layout()
    plt.show()


#%% forecast the sign

from Reg_arima import sign_arima
from datetime import datetime, timedelta

df['sign'] = df['sign'].replace(-1,0)

start_sim = datetime.strptime('2023-03-01 00:00:00', '%Y-%m-%d %H:%M:%S')
end_sim = datetime.strptime('2023-03-7 23:00:00', '%Y-%m-%d %H:%M:%S')

lookahead = 6
lookback = 24*30

# initialise results database
date_list = []
date = start_sim
while date <= end_sim:
    date_list.append(date)
    date += timedelta(hours=1)
results = pd.DataFrame(index=date_list)  

# fitting, forecast and save results!    
for start_test in results.index:
    end_test = start_test + timedelta(hours=lookahead-1)
    end_trial = start_test - timedelta(hours=1)
    start_trial = end_trial - timedelta(hours=lookback)
    sign_forecast = sign_arima(df['sign'][start_trial:end_trial],df['sign'][start_test:end_test],lookahead,0.05,3)
    
    for h in range(lookahead):
        results.at[start_test,f"sign_{h}"] = df['sign'][start_test+timedelta(hours=h)]
        results.at[start_test,f"signF_{h}"] = sign_forecast[h]
          
results['signF_0'] = results['signF_0'].apply(lambda x: 1 if x > 0.5 else 0)
(results['sign_0'] == results['signF_0']).sum() / len(results)

# same results as sign = sign-1


