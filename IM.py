"""
MI vs DAM
"""


#%%
import numpy as np
import pandas as pd
import Database_management as dbm
import matplotlib.pyplot as plt
from datetime import datetime


#%%##################################################################### import databases

y1,m1,d1 = 2023,1,1
y2,m2,d2 = 2024,2,28
zone = 'CNOR'

PdamDB = dbm.concat_xml_GME("C:/Users/pasqui/Desktop/MercatiElettrici/MGP_Prezzi",y1,m1,d1,y2,m2,d2,"MGPPrezzi")
VdamDB = dbm.concat_xml_GME("C:/Users/pasqui/Desktop/MercatiElettrici/MGP_Quantita",y1,m1,d1,y2,m2,d2,"MGPQuantita")

Pim1DB = dbm.concat_xml_GME("C:/Users/pasqui/Desktop/MercatiElettrici/MI-A1_Prezzi",y1,m1,d1,y2,m2,d2,"MI-A1Prezzi")
Vim1DB = dbm.concat_xml_GME("C:/Users/pasqui/Desktop/MercatiElettrici/MI-A1_Quantita",y1,m1,d1,y2,m2,d2,"MI-A1Quantita")

Pim2DB = dbm.concat_xml_GME("C:/Users/pasqui/Desktop/MercatiElettrici/MI-A2_Prezzi",y1,m1,d1,y2,m2,d2,"MI-A2Prezzi")
Vim2DB = dbm.concat_xml_GME("C:/Users/pasqui/Desktop/MercatiElettrici/MI-A2_Quantita",y1,m1,d1,y2,m2,d2,"MI-A2Quantita")

Pim3DB = dbm.concat_xml_GME("C:/Users/pasqui/Desktop/MercatiElettrici/MI-A3_Prezzi",y1,m1,d1,y2,m2,d2,"MI-A3Prezzi")
Vim3DB = dbm.concat_xml_GME("C:/Users/pasqui/Desktop/MercatiElettrici/MI-A3_Quantita",y1,m1,d1,y2,m2,d2,"MI-A3Quantita")

xbidDB = dbm.concat_xbid_GME("C:/Users/pasqui/Desktop/MercatiElettrici/XBID_EsitiTotali",y1,m1,d1,y2,m2,d2,"XBIDEsitiTotali",zone)


#%% ###################################################################### Prices

df = pd.DataFrame(columns=['DAM','IM-1','IM-2','IM-3','XBID','XBID-LH'], index=PdamDB.index)

df['DAM'] = pd.to_numeric(PdamDB[zone].str.replace(',', '.'), errors='coerce').round(3)
df['IM-1'] = pd.to_numeric(Pim1DB[zone].str.replace(',', '.'), errors='coerce').round(3)
df['IM-2'] = pd.to_numeric(Pim2DB[zone].str.replace(',', '.'), errors='coerce').round(3)
df['IM-3'] = pd.to_numeric(Pim3DB[zone].str.replace(',', '.'), errors='coerce').round(3)
df['XBID'] = pd.to_numeric(xbidDB['Riferimento']).round(3)
df['XBID-LH'] = pd.to_numeric(xbidDB['LastHour']).round(3)

# =============================================================================
# start = datetime.strptime('2023-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
# end = datetime.strptime('2023-04-30 00:00:00', '%Y-%m-%d %H:%M:%S')
# df = df[start:end]
# =============================================================================

df['hour'] = df.index.hour
dfh = df.groupby(['hour']).mean()
dfh['IM-3'][0], dfh['IM-3'][11] = np.nan, np.nan

plt.figure(dpi=1000)
for y in dfh:    
    plt.plot(dfh.index,dfh[y],label=y)
plt.legend()
plt.grid()
plt.xlabel('Hour')
plt.ylabel('Average energy price [€/MWh]')
plt.xlim(0,23)
plt.ylim(0,250)
plt.xticks([0,2,4,6,8,10,12,14,16,18,20,22],[0,2,4,6,8,10,12,14,16,18,20,22])
plt.show()


#%% #################################################################### Volumes

df = pd.DataFrame(columns=['DAM','IM-1','IM-2','IM-3','XBID'], index=PdamDB.index)

df['DAM'] = pd.to_numeric(VdamDB[f"{zone}_ACQUISTI"].str.replace(',', '.'), errors='coerce').round(3)
df['IM-1'] = pd.to_numeric(Vim1DB[f"{zone}_ACQUISTI"].str.replace(',', '.'), errors='coerce').round(3)
df['IM-2'] = pd.to_numeric(Vim2DB[f"{zone}_ACQUISTI"].str.replace(',', '.'), errors='coerce').round(3)
df['IM-3'] = pd.to_numeric(Vim3DB[f"{zone}_ACQUISTI"].str.replace(',', '.'), errors='coerce').round(3)
df['XBID'] = pd.to_numeric(xbidDB['Acquisti']).round(3)
df['IM'] = df['XBID'] + df['IM-1'] + df['IM-2'] + df['IM-3']
 
start = datetime.strptime('2023-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
end = datetime.strptime('2023-04-30 00:00:00', '%Y-%m-%d %H:%M:%S')

# =============================================================================
# start = datetime.strptime('2023-05-01 00:00:00', '%Y-%m-%d %H:%M:%S')
# end = datetime.strptime('2023-08-31 00:00:00', '%Y-%m-%d %H:%M:%S')
# 
# =============================================================================
# =============================================================================
# start = datetime.strptime('2023-09-01 00:00:00', '%Y-%m-%d %H:%M:%S')
# end = datetime.strptime('2023-12-31 00:00:00', '%Y-%m-%d %H:%M:%S')
# =============================================================================
# =============================================================================
# start = datetime.strptime('2024-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
# end = datetime.strptime('2024-02-28 00:00:00', '%Y-%m-%d %H:%M:%S')
# =============================================================================
df = df[start:end]
nd = len(df['IM'])/(24) 

print(df['IM'].sum()/nd)
print(df['DAM'].sum()/nd)
print(df['IM'].sum()/(df['DAM'].sum())*100)

labels = ['IM-1','IM-2','IM-3','XBID']
colors = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple']
plt.figure(dpi=1000)
plt.bar(range(len(labels)),[df['IM-1'].sum()/nd,df['IM-2'].sum()/nd,df['IM-3'].sum()/nd,df['XBID'].sum()/nd],color=colors,label=labels)
plt.legend()
plt.xticks([])
plt.tick_params(axis='x', which='both', bottom=False, top=False)
plt.ylabel('Average daily exchange [MWh/day]')
plt.ylim(0,3000)
plt.grid()
plt.show()





#%% DAM prezzo medio 2024

y1,m1,d1 = 2024,1,1
y2,m2,d2 = 2024,6,24
zone = 'PUN'
PdamDB = dbm.concat_xml_GME("C:/Users/pasqui/Desktop/MercatiElettrici/MGP_Prezzi",y1,m1,d1,y2,m2,d2,"MGPPrezzi")[zone]
PdamDB.index = pd.to_datetime(PdamDB.index)
PdamDB = PdamDB.str.replace(',', '.').astype(float)
hourly_mean = PdamDB.groupby(PdamDB.index.hour).mean()
hourly_mean.plot()

# =============================================================================
# full_year = pd.date_range(start='2024-01-01 00:00:00', end='2024-12-31 23:00:00', freq='H')
# repeated_hourly_mean = pd.Series( data=[hourly_mean[hour] for hour in full_year.hour], index=full_year)
# repeated_hourly_mean.to_csv('PUN_24a.csv')
# 
# =============================================================================

