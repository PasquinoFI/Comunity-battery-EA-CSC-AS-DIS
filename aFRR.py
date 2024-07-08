"""
aFRRR
"""


#%%
import numpy as np
import pandas as pd
import Database_management as dbm
import matplotlib.pyplot as plt
from datetime import datetime


#%%##################################################################### import databases

# MB_PTotali # MB_PRiservaSecondaria # MB_PAltriServizi questi tre hanno tutti la stessa struttura e tot è la somma degli altri due
# forse per altri servizi intende tutto cio che non è secondaria??? controllare se la somma torna


# MB_OffertePubbliche
# AFRR_OffertePubbliche

y1,m1,d1 = 2023,1,1
y2,m2,d2 = 2024,2,28
zone = 'CNOR'

RS = dbm.concat_xml_GME("C:/Users/pasqui/Desktop/MercatiElettrici/MB_PRiservaSecondaria",y1,m1,d1,y2,m2,d2,"MBPRiservaSecondaria")
AS = dbm.concat_xml_GME("C:/Users/pasqui/Desktop/MercatiElettrici/MB_PAltriServizi",y1,m1,d1,y2,m2,d2,"MBPAltriServizi")
T = dbm.concat_xml_GME("C:/Users/pasqui/Desktop/MercatiElettrici/MB_PTotali",y1,m1,d1,y2,m2,d2,"MBPTotali")

MSD = dbm.concat_xml_GME("C:/Users/pasqui/Desktop/MercatiElettrici/MSD_ServiziDispacciamento",y1,m1,d1,y2,m2,d2,"MSDServiziDispacciamento")


a = read_xml_GME("C:/Users/pasqui/Desktop/MercatiElettrici/MB_OffertePubbliche",2023,12,20,"MBOffertePubbliche")
#d = read_xml_GME("C:/Users/pasqui/Desktop/MercatiElettrici/MSD_OffertePubbliche",2023,12,20,"MSDOffertePubbliche")


df = pd.DataFrame(index=RS.index)

#df['Pur'] = pd.to_numeric(RS[f"{zone}_AcquistiMWh_norev"].str.replace(',', '.'), errors='coerce').round(3)
#df['PurPmin'] = pd.to_numeric(RS[f"{zone}_PrezzoMinimoAcquisto"].str.replace(',', '.'), errors='coerce').round(3)

df['MB:RS_down'] = pd.to_numeric(RS[f"{zone}_PrezzoMedioAcquisto"].str.replace(',', '.'), errors='coerce').round(3)
df['MB:AS_down'] = pd.to_numeric(AS[f"{zone}_PrezzoMedioAcquisto"].str.replace(',', '.'), errors='coerce').round(3)
df['MSD_down'] = pd.to_numeric(MSD[f"{zone}_PrezzoMedioAcquisto"].str.replace(',', '.'), errors='coerce').round(3)
#df['Sel'] = pd.to_numeric(RS[f"{zone}_VenditeMWh_norev"].str.replace(',', '.'), errors='coerce').round(3)
#df['SelPmax'] = pd.to_numeric(RS[f"{zone}_PrezzoMassimoVendita"].str.replace(',', '.'), errors='coerce').round(3)
df['MB:RS_up'] = pd.to_numeric(RS[f"{zone}_PrezzoMedioVendita"].str.replace(',', '.'), errors='coerce').round(3)
df['MB:AS_up'] = pd.to_numeric(AS[f"{zone}_PrezzoMedioVendita"].str.replace(',', '.'), errors='coerce').round(3)
df['MSD_up'] = pd.to_numeric(MSD[f"{zone}_PrezzoMedioVendita"].str.replace(',', '.'), errors='coerce').round(3)



df_des = df.describe()

df['hour'] = df.index.hour
dfh = df.groupby(['hour']).mean()

plt.figure(dpi=1000)
for y in dfh:    
    ls = '-'
    if "up" in str(y):
        ls = '--'
    plt.plot(dfh.index,dfh[y],label=y,ls=ls)
plt.legend()
plt.grid()
plt.xlabel('Hour')
plt.ylabel('Average energy price [€/MWh]')
plt.xlim(0,23)
plt.ylim(0,250)
plt.xticks([0,2,4,6,8,10,12,14,16,18,20,22],[0,2,4,6,8,10,12,14,16,18,20,22])
plt.show()
