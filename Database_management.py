"""
Data peparation
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



def concat_xbid_GME(folder,year1,month1,day1,year2,month2,day2,what,zone):
      
      # initialise df with the first day
      year = year1
      month = month1
      day = day1
      df = read_xml_GME(folder,year,month,day,what)
      df = df[df['Zone']==zone]
      
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
                  
          dfd = read_xml_GME(folder,year,month,day,what)
          dfd = dfd[dfd['Zone']==zone]
          df = pd.concat([df, dfd], ignore_index=True)
          if year == year2 and month == month2 and day == day2:
              go = 0
              
      df.index = pd.to_datetime(df['FlowDate']) + pd.to_timedelta(pd.to_numeric(df['Hour'])-1, unit='h') # 0 - 23 
      
      # summer/solar time correction: delete duplicates and reach for skips 
      df = df[~df.index.duplicated(keep='first')]
      new_row = df.loc[pd.to_datetime('2023-03-26 22:00:00')].copy()
      new_row.name = pd.to_datetime('2023-03-26 23:00:00')
      df.loc[pd.to_datetime('2023-03-26 23:00:00')] = new_row
      df = df.sort_index()
      
      return(df)
    

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
            
    df.index = pd.to_datetime(df['Data']) + pd.to_timedelta(pd.to_numeric(df['Ora'])-1, unit='h') # 0 - 23 
    
    # summer/solar time correction: delete duplicates and reach for skips 
    df = df[~df.index.duplicated(keep='first')]
    new_row = df.loc[pd.to_datetime('2023-03-26 22:00:00')].copy()
    new_row.name = pd.to_datetime('2023-03-26 23:00:00')
    df.loc[pd.to_datetime('2023-03-26 23:00:00')] = new_row
    df = df.sort_index()
    
    return(df)
    
    
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
            
    df.index =  pd.to_datetime(df['Data Riferimento'])
    
    # summer/solar time correction: delete duplicates and reach for skips 
    df.drop(pd.to_datetime('2023-10-29 02:01:00'), inplace=True)
    new_rows = df.loc[pd.to_datetime('2023-03-26 01:00:00')].copy()
    new_row1 = new_rows.iloc[0]
    new_row2 = new_rows.iloc[1]
    new_row1.name = pd.to_datetime('2023-03-26 02:00:00')
    new_row2.name = pd.to_datetime('2023-03-26 02:00:00')
    df.loc[pd.to_datetime('2023-03-26 02:00:00')] = new_row1
    df.loc[pd.to_datetime('2023-03-26 02:00:00')] = new_row2
    df = df.sort_index()

    return(df)
    

def cut(serie,alpha):
    
    lc = np.percentile(serie,alpha) # lower cut
    uc = np.percentile(serie,100-alpha) # upper cut
    serie[serie>uc] = uc
    serie[serie<lc] = lc
    
    return(serie)

def cut_smart(serie,lb,ub):
    index_tocut = serie[serie > ub].index
    while len(index_tocut) > 0:
        serie.loc[index_tocut] = serie.shift(1).loc[index_tocut]
        index_tocut = serie[serie > ub].index
    index_tocut = serie[serie < lb].index
    while len(index_tocut) > 0:
        serie.loc[index_tocut] = serie.shift(1).loc[index_tocut]
        index_tocut = serie[serie < lb].index
    return(serie)

def cut_smart3(serie1,lb1,ub1,serie2,lb2,ub2,serie3,lb3,ub3):
    
    index_tocut = serie1[serie1 > ub1].index
    while len(index_tocut) > 0:
        serie1.loc[index_tocut] = serie1.shift(1).loc[index_tocut]
        serie2.loc[index_tocut] = serie2.shift(1).loc[index_tocut]
        serie3.loc[index_tocut] = serie3.shift(1).loc[index_tocut]        
        index_tocut = serie1[serie1 > ub1].index
    index_tocut = serie1[serie1 < lb1].index
    while len(index_tocut) > 0:
        serie1.loc[index_tocut] = serie1.shift(1).loc[index_tocut]
        serie2.loc[index_tocut] = serie2.shift(1).loc[index_tocut]
        serie3.loc[index_tocut] = serie3.shift(1).loc[index_tocut]
        index_tocut = serie1[serie1 < lb1].index
        
    index_tocut = serie2[serie2 > ub2].index
    while len(index_tocut) > 0:
        serie1.loc[index_tocut] = serie1.shift(1).loc[index_tocut]
        serie2.loc[index_tocut] = serie2.shift(1).loc[index_tocut]
        serie3.loc[index_tocut] = serie3.shift(1).loc[index_tocut]        
        index_tocut = serie2[serie2 > ub2].index
    index_tocut = serie2[serie2 < lb2].index
    while len(index_tocut) > 0:
        serie1.loc[index_tocut] = serie1.shift(1).loc[index_tocut]
        serie2.loc[index_tocut] = serie2.shift(1).loc[index_tocut]
        serie3.loc[index_tocut] = serie3.shift(1).loc[index_tocut]
        index_tocut = serie2[serie2 < lb2].index
        
    index_tocut = serie3[serie3 > ub3].index
    while len(index_tocut) > 0:
        serie1.loc[index_tocut] = serie1.shift(1).loc[index_tocut]
        serie2.loc[index_tocut] = serie2.shift(1).loc[index_tocut]
        serie3.loc[index_tocut] = serie3.shift(1).loc[index_tocut]        
        index_tocut = serie3[serie3 > ub3].index
    index_tocut = serie3[serie3 < lb3].index
    while len(index_tocut) > 0:
        serie1.loc[index_tocut] = serie1.shift(1).loc[index_tocut]
        serie2.loc[index_tocut] = serie2.shift(1).loc[index_tocut]
        serie3.loc[index_tocut] = serie3.shift(1).loc[index_tocut]
        index_tocut = serie3[serie3 < lb3].index
        
    return(serie1,serie2,serie3)
    
def cut_smart2(serie1,lb1,ub1,serie2,lb2,ub2):
    
    index_tocut = serie1[serie1 > ub1].index
    while len(index_tocut) > 0:
        serie1.loc[index_tocut] = serie1.shift(1).loc[index_tocut]
        serie2.loc[index_tocut] = serie2.shift(1).loc[index_tocut]        
        index_tocut = serie1[serie1 > ub1].index
    index_tocut = serie1[serie1 < lb1].index
    
    while len(index_tocut) > 0:
        serie1.loc[index_tocut] = serie1.shift(1).loc[index_tocut]
        serie2.loc[index_tocut] = serie2.shift(1).loc[index_tocut]
        index_tocut = serie1[serie1 < lb1].index
        
    index_tocut = serie2[serie2 > ub2].index
    while len(index_tocut) > 0:
        serie1.loc[index_tocut] = serie1.shift(1).loc[index_tocut]
        serie2.loc[index_tocut] = serie2.shift(1).loc[index_tocut]        
        index_tocut = serie2[serie2 > ub2].index
    index_tocut = serie2[serie2 < lb2].index
    
    while len(index_tocut) > 0:
        serie1.loc[index_tocut] = serie1.shift(1).loc[index_tocut]
        serie2.loc[index_tocut] = serie2.shift(1).loc[index_tocut]
        index_tocut = serie2[serie2 < lb2].index
        
    return(serie1,serie2) 
    
    
    
