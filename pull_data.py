"""
pull/push data from/to microgrid
"""
import socket, json, time, threading
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

class microgrid:
    
    def __init__(self,load_profile):
        
        self.ip_from, self.port_from = '128.178.46.54', 35210  # Matti
        self.ip_to, self.port_to = '192.168.1.86', 33003 # Franci        

        self.P = pd.DataFrame(columns=[0,1,2,3,4,5,6,7,8]) # busses active power
        self.Q = pd.DataFrame(columns=[0,1,2,3,4,5,6,7,8]) # busses active power
        self.bus_index = [0,1,2,3,4,7,8,9,10] # microgrid bus index
        self.load = pd.read_csv(load_profile,index_col=0)
        self.load.index = pd.to_datetime(self.load.index)
        self.load = self.load.resample('1S').ffill()
        self.bess = None
        self.bess_q = []
    
    def start_pulling(self):
        # Read and save messages sent to self.ip_from and self.port_from
        self.sock_from = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_from.bind((self.ip_from, self.port_from)) # ip and port
        self.pulling = True
        
        while self.pulling: # continuosly
            data, server = self.sock_from.recvfrom(20000) 
            self.data = json.loads(data.decode('utf-8'))
            P = []
            Q = []
            for bus in self.P:
                P += [0]
                Q += [0]
                for phase in range(3):
                    P[bus] += json.loads(data.decode('utf-8'))['SCADA']['Data']['Buses'][self.bus_index[bus]]['Terminals'][phase]['P'] ### potenza attiva sulla fase 1 al GCP [W]
                    Q[bus] += json.loads(data.decode('utf-8'))['SCADA']['Data']['Buses'][self.bus_index[bus]]['Terminals'][phase]['Q'] ### potenza attiva sulla fase 1 al GCP [W]

            self.P.loc[json.loads(data.decode('utf-8'))['SCADA']['Info']['Timestamp']] = P
            self.Q.loc[json.loads(data.decode('utf-8'))['SCADA']['Info']['Timestamp']] = Q
            self.bess = json.loads(data.decode('utf-8'))['Samsung']
            
    def stop_pulling(self):
        self.pulling = False
        time.sleep(1)
        self.sock_from.close()
        
    def stop_pushing(self):
        self.pushing = False
        time.sleep(1)
        self.sock_to.close()
        
    def get_avg_power(self):
        P = self.P.mean()
        Q = self.Q.mean()
        self.P = pd.DataFrame(columns=[0,1,2,3,4,5,6,7,8]) # clean memory
        self.Q = pd.DataFrame(columns=[0,1,2,3,4,5,6,7,8]) # clean memory
        return(P,Q)   
        
    def start_pushing(self):
        # Send messages to self.ip_to and self.port_to
        self.sock_to = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.pushing = True
        
        while self.pushing:
            timenow = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            
# =============================================================================
#             Franci = { 
#                 'ZENONE' : {"P_set [kW]": self.load.loc[timenow]['Active Power [kW]'],
#                             "Q_set [kVAr]": self.load.loc[timenow]['Reactive Power [kVAr]']}, # (cosphi=0.9),
#                  'BESS' : {"P_set [kW]": 0, 
#                            "Q_set [kVAr]": 0.1}}
# =============================================================================
            
            
            ### charging BESS and zenone off
            Franci = {  
                'ZENONE' : {"P_set [kW]": 0,
                            "Q_set [kVAr]": 0}, # (cosphi=0.9),
                 'BESS' : {"P_set [kW]": 5, 
                           "Q_set [kVAr]": 0}}
            
            json_string = json.dumps(Franci)
            
            self.sock_to.sendto(json_string.encode('utf-8'), (self.ip_to, self.port_to))
            
            #print('\n')
            #print(json_string)
            time.sleep(1)
            
    def test_QP(self,serie_P):
        # Send messages to self.ip_to and self.port_to
        self.sock_to = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.pushing = True
        self.sock_from = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_from.bind((self.ip_from, self.port_from)) # ip and port
        self.pulling = True
        
        for P in serie_P:
            
            timenow = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            
            Franci = { 
                'ZENONE' : {"P_set [kW]": self.load.loc[timenow]['P [W]']/1000,
                            "Q_set [kVAr]": self.load.loc[timenow]['P [W]']/1000 *0.484}, # (cosphi=0.9),
                 'BESS' : {"P_set [kW]": P, 
                           "Q_set [kVAr]": 0}}
            
            # send P
            json_string = json.dumps(Franci)
            self.sock_to.sendto(json_string.encode('utf-8'), (self.ip_to, self.port_to))

            # read Q
            data, server = self.sock_from.recvfrom(20000) 
            self.bess_q += [json.loads(data.decode('utf-8'))['Samsung']['Data']['Q']]
            time.sleep(1)
                

                
if __name__=='__main__':
    
    
    ### push and pull i continuos
    grid = microgrid("12_residential_load_profiles_s1_5minutes.csv")
    pull_thread = threading.Thread(target=grid.start_pulling)
    push_thread = threading.Thread(target=grid.start_pushing)
    pull_thread.start()
    push_thread.start()
    
    
    
    t0=time.time()
    while time.time()-t0 < 5: # while True per andare a diritto
        
        time.sleep(1)
        p,q = grid.get_avg_power()
        print(p)
        print(grid.bess)
        print('\n')
    
    grid.stop_pulling()
    grid.stop_pushing()
    
    
    
    
    
    
    ##### Q = f(P) test ###################################################
# =============================================================================
#     serie_P = np.repeat(np.arange(0,20.1,0.1),3)
#     grid = microgrid("12_residential_load_profiles_s1_seconds_xFranci.csv")
#     grid.test_QP(serie_P)
#     grid.stop_pulling()
#     grid.stop_pushing()
#     test = pd.DataFrame({'P':serie_P, 'Q':np.array(grid.bess_q)})
#     test.to_csv('test_QP')
# =============================================================================
    

# =============================================================================
#     test = pd.read_csv("EA_converter_QfP.csv")
#     #test.to_csv("EA_converter_QfP.csv")
#     
#     import matplotlib.pyplot as plt
#     fig, (ax1) = plt.subplots(1,1,dpi=1000)
#     ax1.scatter(test['P']/1000,test['Q']/1000,alpha=1,s=2)
#     ax1.set_xlabel("Active power [kW]")
#     ax1.set_ylabel("Reactive power [kVAr]")
#     plt.grid()
#     plt.title("EA Converter Q = f(P) ")
#     plt.show()
# =============================================================================







    
### a inizio simulazione creo l'oggetto e inizio a farlo pullare i dati
### ad ogni controllo lancio get_avg_power e moltiplicandolo per il timestep avro' l'energia realizzata


# =============================================================================
#     timestamp = t[0]
#     date = datetime.fromtimestamp(timestamp)
#     date = date - relativedelta(years=66,month=5,day=31) 
#     print(date.strftime("%d/%m/%Y, %H:%M:%S"))
# =============================================================================

import matplotlib.pyplot as plt
# test pullind dwd
# Read and save messages sent to self.ip_from and self.port_from


# =============================================================================
# time.sleep(60*50)
# while True:
#     ip_from, port_from = '128.178.46.54', 35211  # Matti
#     sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     sock.bind((ip_from, port_from)) # ip and port
#     data, server = sock.recvfrom(60000) 
#     a = json.loads(data.decode('utf-8'))
#     sock.close()
#     # Convert data to lists
#     x = list(a['0'].keys())
#     y = list(a['0'].values())
#     # Plot the data
#     plt.figure(figsize=(10, 6),dpi=1000)
#     plt.plot(x, y, marker='o')
#     plt.xticks(rotation=90)
#     plt.xlabel('Time')
#     plt.ylabel('GHI')
#     plt.title('DWD forecast')
#     plt.grid(True)
#     plt.tight_layout()
#     # Show the plot
#     plt.show()
#     time.sleep(60*60)
# =============================================================================
    


