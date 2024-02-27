"""
Auxiliary functions
"""

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET


def read_xml_GME(folder,year,month,day,what):
    
    # formatting
    if month < 10:
        month = f"0{month}"
    if day < 10:
        day = f"0{day}"

    # read xml and creat dictionary
    tree = ET.parse(f"{folder}/{year}{month}{day}{what}.xml")    
    root = tree.getroot()
    data = []
    for child in root:
        row = {}
        for subchild in child:
            row[subchild.tag] = subchild.text
        data.append(row)
    
    # from dictionary to clean database
    data = pd.DataFrame(data)
    data = data.drop(data.columns[0], axis=1)
    data = data.drop(data.index[0])
    
    return(data)


def concat_xml_GME(folder,year1,month1,day1,year2,month2,day2,what):
    
    # initialise df with the first day
    year = year1
    month = month1
    day = day1
    df = read_xml_GME(folder,year,month,day,what)
    
    # add days
    days_in_month = {1: 31,2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    
    go = 1
    while go:
        
        day += 1
        
        if day > days_in_month[month]:
            day = 1
            month = month +1
            if month > 12:
                month = 1
                year += 1
        
        df = pd.concat([df, read_xml_GME(folder,year,month,day,what)], ignore_index=True)
        if year == year2 and month == month2 and day == day2:
            go = 0
            
    df['Timeindex'] = pd.to_datetime(df['Data'])
    df['Timeindex'] = pd.to_datetime(df['Timeindex']) + pd.to_timedelta(pd.to_numeric(df['Ora'])-1, unit='h')
    
    df['Giorno'] = df['Timeindex'].dt.day
    df['Mese'] = df['Timeindex'].dt.month
    df['Ora'] = df['Timeindex'].dt.hour
    df.index = df['Timeindex']
    
    return(df)
    

#df = concat_xml_GME("C:/Users/pasqui/Desktop/MercatiElettrici/MGP_Prezzi",2023,1,1,2024,2,10,"MGPPrezzi")
    
def concat_xlsx_Terna(folder,year1,month1,year2,month2,what):
    
    # initialise df with the first month
    year = year1
    month = month1
    
    # formatting
    if month < 10:
        monthf = f"0{month}"
    else:
        monthf = month
    
    df = pd.read_excel(f"{folder}/{what}_{year}{monthf}.xlsx",header=1)
    go = 1
    while go:
        month = month +1
        if month > 12:
            month = 1
            year += 1
        
        # formatting
        if month < 10:
            monthf = f"0{month}"
        else:
            monthf = month
        df = pd.concat([df, pd.read_excel(f"{folder}/{what}_{year}{monthf}.xlsx",header=1)], ignore_index=True)
        if year == year2 and month == month2:
            go = 0
            
    df['Timeindex'] = pd.to_datetime(df['Data Riferimento'])
    df['Giorno'] = df['Timeindex'].dt.day
    df['Mese'] = df['Timeindex'].dt.month
    df['Ora'] = df['Timeindex'].dt.hour
    df.index = df['Timeindex']
    
    return(df)
    
#sb =  concat_xlsx_Terna("C:/Users/pasqui/Desktop/Sbilanciamenti",2023,1,2023,12,"Riepilogo_Mensile_Orario")

    