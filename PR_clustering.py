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
    plt.scatter(df[r],df['P-PR'])
    plt.xlabel(r)
    plt.ylabel('P-PR [€/MWh]')
    plt.grid(axis='y')
    plt.show()
    














