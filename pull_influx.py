"""
test Grafana influx pull
"""


import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
import pandas as pd
from datetime import datetime,timedelta

def pull_influx(time,ahead,what,pvname):
    
    start = str(time)[:10]+'T'+str(time)[11:]+'Z'
    time = time + timedelta(hours=ahead-1)
    stop = str(time)[:10]+'T'+str(time)[11:]+'Z'
    
    ###########################################################################

    org = "DESL-EPFL"
    bucket = "microgrid_ST"
    token = "xRZfjgfLcEkGbU8ZxDtnbaV2kMbObWN_Q-fg8ZPAw_68R7VNuCbFb6x284eBuPowtRxOF1mba13VRMZ_0StMnQ=="
    IP = '128.179.34.35'
    PORT = 52000
    url = f"http://{IP}:{PORT}"
    client = influxdb_client.InfluxDBClient(
        url=url,
        token=token,
        org=org
    )
    # Query script
    query_api = client.query_api()
    
    if what=='irradiance':
        query = f'''from(bucket: "microgrid_ST")
          |> range(start: {start}, stop: {stop})
          |> filter(fn: (r) => r["_measurement"] == "microgrid")
          |> filter(fn: (r) => r["Resource"] == "meteobox_roof")
          |> filter(fn: (r) => r["_field"] == "GHI")
          |> aggregateWindow(every: 2m0s, fn: mean, createEmpty: false)
          |> yield(name: "mean")'''
      
    if what=='power':
          query = f'''from(bucket: "microgrid_ST")
            |> range(start: {start}, stop: {stop})
            |> filter(fn: (r) => r["Resource"] == "{pvname}")
            |> filter(fn: (r) => r["_field"] == "P")
            |> aggregateWindow(every: 2m0s, fn: mean, createEmpty: false)
            |> yield(name: "mean")'''
      
 
    result = query_api.query(org=org, query=query)
    # Initialize an empty list to store data
    data = []
    # Iterate through the result tables and records
    for table in result:
        for record in table.records:
            data.append({
                "Time": record.get_time(),
                "Measurement": record.get_measurement(),
                "Field": record.get_field(),
                "Value": record.get_value()
                # You can add more fields here as needed
            })
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)
    
    # Convert the Time column to datetime format for better manipulation
    df['Time'] = pd.to_datetime(df['Time'])
    df.index = df['Time']
    df = df.drop(['Time', 'Measurement','Field'], axis=1)
    #df = df.resample('15T').ffill()
    
    if what=='irradiance':
        df.columns=(['ghi [W/m2]']) 
        df['ghi [W/m2]'].iloc[0] = 0
    if what=='power':
        df.columns=(['power [W]']) 
        df['power [W]'].iloc[0] = 0
    
    
    return(df)






