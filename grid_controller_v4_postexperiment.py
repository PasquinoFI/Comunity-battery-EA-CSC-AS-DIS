# -*- coding: utf-8 -*-
"""
grid controller per le simulationi e per l'esperimento
"""

import socket, json, time, threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from PV_forecast2 import pv_production2, empirical_model, pv_production3
from scheduling_iDistFlow import scheduling_iDistFlow
from plots_controller import scheduling_plot, control_plot, control_plot_final
from control_iDistFlow import control_iDistFlow, fake_DPr
from LoadFlow import solve_Load_flow
import PerUnit

class controller:
    
    def __init__(self,grid,bess,load,PVs,economic,simulation):
        
        # grid and bess (only one bess is possible at the moment) (a virtual line is added to the grid to simulate battery-related losses)
        self.grid = pd.DataFrame(pd.read_csv(f"input/{grid['filename']}",delim_whitespace=True, skipinitialspace=True)).rename_axis("line")
        self.bess = bess
        self.grid.loc[len(self.grid)] = {'busup':self.bess['bus'],'busdown':len(self.grid)+1,'r[ohm]':self.bess['r'],'x[ohm]':0,'B(S)':0,'/':1,'ampacity[A]':1e3,'length[km]':0}
        self.bess['bus_real'] = self.bess['bus']
        self.bess['bus'] = len(self.grid)
        self.L = len(self.grid) # number of lines
        self.B = self.L+1  # number of busses
        self.Ab = grid['Ab'] # per unit power base [W]
        self.Eb = grid['Eb'] # per unit voltage base and GCP constrain [V]
        self.bess['SoE_max'] = self.bess['SoE_max']/self.Ab
        self.bess['P_max'] = self.bess['P_max']/self.Ab
        self.grid = PerUnit.to_pu_grid(self.grid, self.Ab, self.Eb)
        
        # simulation
        self.S = simulation['scenarios']   
        self.time_start = datetime.strptime(simulation['start'], '%Y-%m-%d %H:%M:%S')
        self.time_end = datetime.strptime(simulation['end'], '%Y-%m-%d %H:%M:%S')
        self.Ks = simulation['Ks']
        self.Kc = simulation['Kc']
        self.Hor = simulation['horizon']
        self.rtc_step = simulation['rtc_step']
        self.bidding_lag = simulation['bidding_lag']
        
        # load (only one load is possibile at the moment)
        self.loadbus = load['bus']
        self.load_h = pd.read_csv(f"input/{load['filename']}",index_col=0)/self.Ab
        self.load_h.index = pd.to_datetime(self.load_h.index)
        
        self.load_xFranci = pd.read_csv(f"input/{simulation['load_fake_realisation_filename']}",index_col=0)
        self.load_xFranci.index = pd.to_datetime(self.load_xFranci.index)
        self.load_xFranci = self.load_xFranci.resample('1S').ffill() # [W]
        self.load_rtcstep = self.load_xFranci/self.Ab # [pu]
        self.load_rtcstep.index = pd.to_datetime(self.load_rtcstep.index)
        self.load_rtcstep = self.load_rtcstep.resample(f"{int(self.rtc_step)}S").mean()

        # PVs
        self.PVs = PVs
        for pv in PVs:
            self.PVs[pv]['lt_forecast_model'] = empirical_model(PVs[pv]['filename_data_h'])
            self.PVs[pv]['fake_realisation'] = pd.read_csv(f"input/{PVs[pv]['filename_data']}",index_col=0)
            self.PVs[pv]['fake_realisation'].index = pd.to_datetime(self.PVs[pv]['fake_realisation'].index).tz_localize(None)
            if self.rtc_step > 120:
                self.PVs[pv]['fake_realisation'] = self.PVs[pv]['fake_realisation'].resample(f"{int(self.rtc_step)}S").mean()
            else:
                self.PVs[pv]['fake_realisation'] = self.PVs[pv]['fake_realisation'].resample(f"{int(self.rtc_step)}S").ffill()
        
        # economic
        self.ep = pd.read_csv(f"input/{economic['ep_filename']}",index_col=0)/1e6 # [€/MWh] -> [€/Wh]
        self.ep.index = pd.to_datetime(self.ep.index)
        self.acost = economic['acost']/1e3 # [€/kWh] -> [€/Wh]
        self.cscinc = economic['cscinc']/1e3 # [€/kWh] -> [€/Wh]
        self.unbcost = economic['unbcost']/1e3 # [€/kWh] -> [€/Wh]
        
        # scheduling iDistFlow correction parameters initializations
        self.pco = np.ones((self.S,self.Hor,self.L))*0 # pu
        self.qco = np.ones((self.S,self.Hor,self.L))*0 # pu
        self.vco = np.ones((self.S,self.Hor,self.L))*0 # pu
        self.vap = np.ones((self.S,self.Hor,self.B)) # pu
        
        # control iDistFlow correction parameters initializations
        self.pco_c = np.ones(self.L)*0 # pu
        self.qco_c = np.ones(self.L)*0 # pu
        self.vco_c = np.ones(self.L)*0 # pu
        self.vap_c = np.ones(self.B) # pu
        
        # variable to save initialisation
        self.DP_bidded = pd.DataFrame(columns=["pu"])
        self.DP_real = pd.DataFrame(columns=["pu"])
        self.p_bess_set = pd.DataFrame(columns=["pu"]) # on the virtual bus
        self.p_bess_set_real = pd.DataFrame(columns=["pu"]) # on the virtual line / real set point to push
        self.q_bess_set = pd.DataFrame(columns=["pu"])
        self.p_bess_set.loc[self.time_start] = [[0 for b in range(self.B)]] # setpoint before real-time control starts
        self.p_bess_set_real.loc[self.time_start] = 0
        self.q_bess_set.loc[self.time_start] = [[0 for b in range(self.B)]] # setpoint before real-time control starts
        self.bess['SoE'] = pd.DataFrame([self.bess['SoE']/self.Ab], index=[self.time_start], columns=["pu"])
        
        # grid state to save realisation (pulling from microgrid PMU)
        self.bus_index = [0,1,2,3,4,7,8,9,10] # microgrid bus index
        self.P_real = pd.DataFrame(columns=[0,1,2,3,4,5,6,7,8]) # busses active power: continuously updated and reset at each control step
        self.Q_real = pd.DataFrame(columns=[0,1,2,3,4,5,6,7,8]) # busses reactive power: continuously updated and reset at each control step
        self.P_bess_real = pd.DataFrame(columns=[0]) # bess active power: continuously updated and reset at each control step
        self.Q_bess_real = pd.DataFrame(columns=[0]) # bess reactive power: continuously updated and reset at each control step
        self.P_rtcstep = pd.DataFrame(columns=[0,1,2,3,4,5,6,7,8]) # busses active power: updated at each control step
        self.Q_rtcstep = pd.DataFrame(columns=[0,1,2,3,4,5,6,7,8]) # busses reactive power: updated at each control step
        self.P_bess_rtcstep = pd.DataFrame(columns=[0]) # bess active power: updated at each control step
        self.Q_bess_rtcstep = pd.DataFrame(columns=[0]) # bess reactive power: updated at each control step
        self.SoE_min = pd.DataFrame(columns=['pu'])
        self.SoE_max = pd.DataFrame(columns=['pu'])
        self.SoE_mean = pd.DataFrame(columns=['pu'])
        
        # manual_control
        self.manual_control = False   
        self.mcc = True
    
        
    def pv_lt_forecast_sim(self):
        self.p_pv = pd.DataFrame(index=self.scheduling_time_index, columns=[f"prod_{i} [W]" for i in range(self.S)])
        self.q_pv = pd.DataFrame(index=self.scheduling_time_index, columns=[f"prod_{i} [W]" for i in range(self.S)])
        self.p_pv = self.p_pv.map(lambda x: [0]*self.B)
        self.q_pv = self.q_pv.map(lambda x: [0]*self.B)
        for pv in self.PVs:
            for s in range(self.S):
                if s == 0:
                    self.PVs[pv]['pro'] = -pv_production2(s,self.time_scheduling,self.Hor,self.PVs[pv]['lt_forecast_model'])
                else:
                    self.PVs[pv]['pro'] = pd.concat([self.PVs[pv]['pro'],-pv_production2(s,self.time_scheduling,self.Hor,self.PVs[pv]['lt_forecast_model'])],axis=1)                                     
        for s in range(self.S):
            for h in range(self.Hor):
                for pv in self.PVs:
                    self.p_pv.iloc[h,s][self.PVs[pv]['bus']] += self.PVs[pv]['pro'].iloc[h,s] / self.Ab # set pv bus injection [pu]
                    self.q_pv.iloc[h,s][self.PVs[pv]['bus']] += self.p_pv.iloc[h,s][self.PVs[pv]['bus']]*0 # (cosphi=1)    
    
    
    def pv_lt_forecast_exp(self):
        self.p_pv = pd.DataFrame(index=self.scheduling_time_index, columns=[f"prod_{i} [W]" for i in range(self.S)])
        self.q_pv = pd.DataFrame(index=self.scheduling_time_index, columns=[f"prod_{i} [W]" for i in range(self.S)])
        self.p_pv = self.p_pv.map(lambda x: [0]*self.B)
        self.q_pv = self.q_pv.map(lambda x: [0]*self.B)
        
        
        # pullind dwd most updated forecast from linux pc (open and close sock_dwd)
        sock_dwd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock_dwd.bind(('128.178.46.54', 35211)) # ip and port
        sock_dwd.settimeout(0.2)  # Timeout di 1 secondo
        data, server = sock_dwd.recvfrom(60000) 
        try:
            self.dwd = json.loads(data.decode('utf-8'))
        except socket.timeout:
            print("DWD pulling is not working !!!")
        sock_dwd.close()
        
        for pv in self.PVs:
            for s in range(self.S):
                if s == 0:
                    self.PVs[pv]['pro'] = -pv_production3(pd.DataFrame.from_dict(self.dwd[str(s)], orient='index', columns=['ghi [W/m2]']),self.PVs[pv]['lt_forecast_model'],s)
                else:
                    self.PVs[pv]['pro'] = pd.concat([self.PVs[pv]['pro'],-pv_production3(pd.DataFrame.from_dict(self.dwd[str(s)], orient='index', columns=['ghi [W/m2]']),self.PVs[pv]['lt_forecast_model'],s)],axis=1)                                     
        for s in range(self.S):
            for h in range(self.Hor):
                for pv in self.PVs:
                    self.p_pv.iloc[h,s][self.PVs[pv]['bus']] += self.PVs[pv]['pro'].iloc[h,s] / self.Ab# set pv bus injection [pu]
                    self.q_pv.iloc[h,s][self.PVs[pv]['bus']] += self.p_pv.iloc[h,s][self.PVs[pv]['bus']]*0 # (cosphi=1)  
    
    
    def pv_st_forecast_sim(self):
        self.p_pv_st = [0]*self.B
        self.q_pv_st = [0]*self.B
        
        for pv in self.PVs:
            #self.p_pv_st[self.PVs[pv]['bus']] += - self.PVs[pv]['fake_realisation'].loc[self.time_scheduling:self.time_control]['power [W]'].mean() / self.Ab # moving average
            
            if self.time_control > self.time_control_start + timedelta(seconds=self.rtc_step*2):
                self.time_st_forecast = self.time_control - timedelta(seconds=self.rtc_step*2)
            else:
                self.time_st_forecast = self.time_control_start
            self.p_pv_st[self.PVs[pv]['bus']] += - self.PVs[pv]['fake_realisation'].loc[self.time_st_forecast:self.time_control]['power [W]'].mean() / self.Ab # moving average

        
        
    def pv_st_forecast_exp(self):
        self.p_pv_st = [0]*self.B
        self.q_pv_st = [0]*self.B
        
        for pv in self.PVs:
            
            if self.time_control > self.time_control_start + timedelta(seconds=self.rtc_step*2):
                self.time_st_forecast = self.time_control - timedelta(seconds=self.rtc_step*2)
            else:
                self.time_st_forecast = self.time_control_start
            
            self.p_pv_st[self.PVs[pv]['bus']] = self.P_rtcstep[self.time_st_forecast:].mean()[self.PVs[pv]['bus']]  # moving average 2 steps
            self.q_pv_st[self.PVs[pv]['bus']] = self.Q_rtcstep[self.time_st_forecast:].mean()[self.PVs[pv]['bus']]  # moving average 2 steps
        
            if self.PVs[pv]['bus'] == self.bess['bus_real']: # bess e pv on the same bus correction (PMU read the sum)
                self.p_pv_st[self.PVs[pv]['bus']] += - self.P_bess_rtcstep.loc[self.time_st_forecast:].mean()[0] 
                self.q_pv_st[self.PVs[pv]['bus']] += - self.Q_bess_rtcstep.loc[self.time_st_forecast:].mean()[0]

        
    def pv_realisation_sim(self):
        self.p_pv_re = [0]*self.B
        self.q_pv_re = [0]*self.B
        for pv in self.PVs:
            self.p_pv_re[self.PVs[pv]['bus']] += - self.PVs[pv]['fake_realisation'].loc[self.time_control]['power [W]'] / self.Ab
    
        
    def load_lt_forecast(self):
        self.p_load = pd.DataFrame(index=self.scheduling_time_index, columns=[f"load_{i} [W]" for i in range(self.S)])
        self.q_load = pd.DataFrame(index=self.scheduling_time_index, columns=[f"load_{i} [W]" for i in range(self.S)])
        self.p_load = self.p_load.map(lambda x: [0]*self.B)   
        self.q_load = self.q_load.map(lambda x: [0]*self.B) 
        for s in range(self.S):
            for h in range(self.Hor):
                self.p_load.iloc[h,s][self.loadbus] = self.load_h.loc[self.scheduling_time_index].iloc[h,s] # set load bus injection [pu]
                self.q_load.iloc[h,s][self.loadbus] = self.p_load.iloc[h,s][self.loadbus]*0.484 # (cosphi=0.9)
        
    
    def load_st_forecast_sim(self):
        self.p_load_st = [0]*self.B
        self.q_load_st = [0]*self.B
        self.p_load_st[self.loadbus] = self.load_rtcstep.loc[self.time_control_start:self.time_control]['P [W]'].mean() # moving average
        self.q_load_st[self.loadbus] = self.p_load_st[self.loadbus]*0.484 # (cosphi=0.9)
        
        
    def load_st_forecast_exp(self):
        self.p_load_st = [0]*self.B
        self.q_load_st = [0]*self.B
        self.p_load_st[self.loadbus] = self.P_rtcstep[self.time_control_start:].mean()[self.loadbus]  # moving average 
        self.q_load_st[self.loadbus] = self.Q_rtcstep[self.time_control_start:].mean()[self.loadbus]  # moving average
        
        
    def load_realisation_sim(self):
        self.p_load_re = [0]*self.B
        self.q_load_re = [0]*self.B
        self.p_load_re[self.loadbus] = self.load_rtcstep.loc[self.time_control]['P [W]'] # already in pu
        self.q_load_re[self.loadbus] = self.p_load_re[self.loadbus]*0.484 # (cosphi=0.9)
        
      
    def gcp_realisation_sim(self):
        # fake realisation
        self.DP_real.loc[self.time_control] = fake_DPr(self.grid, self.p_pv_re, self.q_pv_re, self.p_load_re, self.q_load_re, self.p_bess_set.loc[self.time_control]['pu'], self.q_bess_set.loc[self.time_control]['pu']) # DP realised at step-1
        self.DPr += self.DP_real.loc[self.time_control]['pu']*self.rtc_step
        
        
    def gcp_realisation_exp(self):
        self.DP_real.loc[self.time_control] = self.P_rtcstep.loc[self.time_control][0]
        self.DPr += self.P_rtcstep.loc[self.time_control][0]*self.rtc_step
    
    
    def average_realisation(self):
        #save average realisation in the control step and clean the memory
        self.P_rtcstep.loc[self.time_control] = self.P_real.mean()
        self.Q_rtcstep.loc[self.time_control] = self.Q_real.mean()
        self.P_bess_rtcstep.loc[self.time_control] = self.P_bess_real.mean()
        self.Q_bess_rtcstep.loc[self.time_control] = self.Q_bess_real.mean()
        self.P_real = pd.DataFrame(columns=[0,1,2,3,4,5,6,7,8]) # clean memory
        self.Q_real = pd.DataFrame(columns=[0,1,2,3,4,5,6,7,8]) # clean memory
        self.P_bess_real = pd.DataFrame(columns=[0]) 
        self.Q_bess_real = pd.DataFrame(columns=[0]) 
        
        self.SoE_mean.loc[self.time_control] = (self.SoE_min.iloc[-1]+self.SoE_max.iloc[-1])/2
        if self.SoE_mean.loc[self.time_control]['pu'] > self.bess['SoE_max']/2:
            self.bess['SoE'].loc[self.time_control] = self.SoE_max.iloc[-1]
        else:
            self.bess['SoE'].loc[self.time_control] = self.SoE_min.iloc[-1]
        self.SoE_min = pd.DataFrame(columns=['pu']) # reset
        self.SoE_max = pd.DataFrame(columns=['pu']) # reset
    
    
    def start_pulling(self):
        """
        GCP realisation --> DP realised
        PV realisation --> PV st forecast --> control
        Load realisation --> Load st forecast --> control
        BESS SoE --> control
        
        optional:
            P, Q and V of all bussess: all the grid state
        """
        
        self.sock_pulling = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_pulling.bind(('128.178.46.54', 35210)) # ip and port from which microgrid measurements are received
        self.sock_pulling.settimeout(0.2)
        self.pulling = True
        
        while self.pulling: # continuosly
            try:
                self.timenow = datetime.fromtimestamp(time.time())
                data, server = self.sock_pulling.recvfrom(20000) 
                P = []
                Q = []
                for bus in self.P_real:
                    P += [0]
                    Q += [0]
                    for phase in range(3):
                        if bus == 0:
                            P[bus] +=  json.loads(data.decode('utf-8'))['SCADA']['Data']['Buses'][self.bus_index[bus]]['Terminals'][phase]['P'] / self.Ab ### Active power in each bus in each phase
                            Q[bus] +=  json.loads(data.decode('utf-8'))['SCADA']['Data']['Buses'][self.bus_index[bus]]['Terminals'][phase]['Q'] / self.Ab ### Reactive power in each bus in each phase
                        else:
                            P[bus] += - json.loads(data.decode('utf-8'))['SCADA']['Data']['Buses'][self.bus_index[bus]]['Terminals'][phase]['P'] / self.Ab ### Active power in each bus in each phase
                            Q[bus] += - json.loads(data.decode('utf-8'))['SCADA']['Data']['Buses'][self.bus_index[bus]]['Terminals'][phase]['Q'] / self.Ab ### Reactive power in each bus in each phase
                self.P_real.loc[self.timenow] = P
                self.Q_real.loc[self.timenow] = Q
                self.P_bess_real.loc[self.timenow] = - json.loads(data.decode('utf-8'))['Samsung']['Data']['P'] / self.Ab
                self.Q_bess_real.loc[self.timenow] = - json.loads(data.decode('utf-8'))['Samsung']['Data']['Q'] / self.Ab 
                soemin = json.loads(data.decode('utf-8'))['Samsung']['Data']['SOC_min']/100 * self.bess['SoE_max']
                soemax = json.loads(data.decode('utf-8'))['Samsung']['Data']['SOC_max']/100 * self.bess['SoE_max']
                self.SoE_min.loc[self.timenow] = [soemin]
                self.SoE_max.loc[self.timenow] = [soemax]               
                
            except socket.timeout:
                print("Pulling is not working !!!")
    
    
    def start_pushing(self):
        """"
        Load setpoint (P and Q)
        BESS setpoint (P and Q)
        """
        # Send messages to self.ip_to and self.port_to
        self.sock_pushing = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_pushing.settimeout(0.2)
        self.pushing = True
        
        while self.pushing:
            try:
                timenow = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
                
                if not self.manual_control: # resources contolled by optimisations 
                    Franci = { 
                        'ZENONE' : {"P_set [kW]": self.load_xFranci.loc[timenow]['P [W]']/1000, 
                                    "Q_set [kVAr]": self.load_xFranci.loc[timenow]['P [W]']/1000 *0.484}, # (cosphi=0.9),
                         'BESS' : {"P_set [kW]": -self.p_bess_set_real.iloc[-1,0]*self.Ab/1000, 
                                   "Q_set [kVAr]": 0}}
                else: # resources contolled by manually
                    Franci = { 
                        'ZENONE' : {"P_set [kW]": self.manual['ZENONE']["P_set [kW]"]/1000, 
                                    "Q_set [kVAr]": self.manual['ZENONE']["Q_set [kVAr]"]/1000},
                         'BESS' : {"P_set [kW]": -self.manual['BESS']["P_set [kW]"]/1000, 
                                   "Q_set [kVAr]": -self.manual['BESS']["Q_set [kVAr]"]/1000}}                    
                    
                json_string = json.dumps(Franci)
                #print(json_string)
                self.sock_pushing.sendto(json_string.encode('utf-8'), ('192.168.1.86', 33003 ))
                time.sleep(1) # send every second
            except socket.timeout:
                print("Pushing is not working !!!")
            
            
    def manual_control_check(self):
        
        while self.mcc:
            ### manual control check
            with open('manual_control.json', 'r') as file: manual = json.load(file)    
            self.manual = manual
            self.manual_control = self.manual['TrueFalse']
            time.sleep(10)
            
    
    def stop_pushing(self):
        self.pushing = False
        time.sleep(0.5)
        self.sock_pushing.close()
        
        
    def stop_pulling(self):
        self.pulling = False
        time.sleep(0.5)
        self.sock_pulling.close()      
    
    
    def scheduling(self): 
        self.p_bess,self.q_bess,DP, self.P, self.SoE_s = scheduling_iDistFlow(self.pco,self.qco,self.vco,self.vap,self.grid,self.p_pv,self.p_load,self.q_pv,self.q_load,self.bess['bus'],self.bess['SoE'].loc[self.time_scheduling]['pu'],self.bess['SoE_max'],self.bess['P_max'],self.ep.loc[self.scheduling_time_index],self.acost,self.cscinc,self.unbcost,self.Ab)
        for ks in range(1,self.Ks+1): 
            for s in range(self.S):
                for h in range(self.Hor):
                    self.pco[s,h],self.qco[s,h],self.vco[s,h],self.vap[s,h] = solve_Load_flow(self.grid,self.p_pv.iloc[h,s],self.q_pv.iloc[h,s],self.p_load.iloc[h,s],self.q_load.iloc[h,s],[self.p_bess[(s, h, b)] for b in range(self.B)],[self.q_bess[(s, h, b)] for b in range(self.B)],coDistFlow=True)
            self.p_bess,self.q_bess,DP, self.P, self.SoE_s = scheduling_iDistFlow(self.pco,self.qco,self.vco,self.vap,self.grid,self.p_pv,self.p_load,self.q_pv,self.q_load,self.bess['bus'],self.bess['SoE'].loc[self.time_scheduling]['pu'],self.bess['SoE_max'],self.bess['P_max'],self.ep.loc[self.scheduling_time_index],self.acost,self.cscinc,self.unbcost,self.Ab)
        ## update DP
        try: 
            self.DP = pd.concat([self.DP.loc[:self.time_scheduling-timedelta(hours=1-self.bidding_lag)],pd.DataFrame(list(DP.values())[self.bidding_lag:], index=self.scheduling_time_index[self.bidding_lag:], columns=['pu'])])
        except: # first scheduling
            self.DP = pd.DataFrame(list(DP.values()), index=self.scheduling_time_index, columns=['pu'])
        #print(self.DP)
        
        
    def scheduling_exp(self):
        
        # long term forecast 
        self.scheduling_time_index = pd.date_range(start=self.time_scheduling, periods=self.Hor, freq='H')
        self.pv_lt_forecast_exp()
        self.load_lt_forecast()
        
        # scheduling
        self.scheduling()
        scheduling_plot(f"{self.time_scheduling} scheduling",self.S,self.Hor,self.SoE_s,self.bess['SoE_max'],self.P,self.DP.loc[self.scheduling_time_index]['pu'],self.PVs,self.load_h.loc[self.scheduling_time_index],self.bess['bus'],self.Ab,self.ep.loc[self.scheduling_time_index])
        
        # set next scheduling time in one hour and wait for it
        self.time_scheduling += timedelta(hours=1) # prossimo scheduling tra un'ora
        
    
    def rt_control(self):
        
        #self.DP.loc[self.time_control_start]['pu'] = -0.09812 # fake tappabuchi Will
        
        self.DP_bidded.loc[self.time_control] = self.DP.loc[self.time_control_start]['pu']
        self.Tc = int(((self.time_control_start+timedelta(hours=1)-self.time_control).seconds)/self.rtc_step) # problem size
        self.p_bess_c,self.q_bess_c,self.SoE_c,self.P_c = control_iDistFlow(self.DP.loc[self.time_control_start]['pu']*3600,self.DPr,self.Tc,self.rtc_step,self.pco_c,self.qco_c,self.vco_c,self.vap_c,self.grid,self.p_pv_st,self.p_load_st,self.q_pv_st,self.q_load_st,self.bess['bus'],self.bess['SoE'].loc[self.time_control]['pu'],self.bess['SoE_max'],self.bess['P_max'],self.Ab)
        for kc in range(1,self.Kc+1):
            self.pco_c,self.qco_c,self.vco_c,self.vap_c = solve_Load_flow(self.grid,self.p_pv_st,self.q_pv_st,self.p_load_st,self.q_load_st,self.p_bess_c,self.q_bess_c,coDistFlow=True)
            self.p_bess_c,self.q_bess_c,self.SoE_c,self.P_c = control_iDistFlow(self.DP.loc[self.time_control_start]['pu']*3600,self.DPr,self.Tc,self.rtc_step,self.pco_c,self.qco_c,self.vco_c,self.vap_c,self.grid,self.p_pv_st,self.p_load_st,self.q_pv_st,self.q_load_st,self.bess['bus'],self.bess['SoE'].loc[self.time_control]['pu'],self.bess['SoE_max'],self.bess['P_max'],self.Ab)
            
            
    def update_bess_sim(self):
        
        # update SoE and bess set point
        self.p_bess_set.loc[self.time_control] = [[self.p_bess_c[b] for b in range(self.B)]]
        self.p_bess_set_real.loc[self.time_control] = self.P_c[self.bess['bus']-1]
        self.q_bess_set.loc[self.time_control] = [[self.q_bess_c[b] for b in range(self.B)]]
        self.bess['SoE'].loc[self.time_control+timedelta(seconds=self.rtc_step)] = [self.SoE_c[1,self.bess['bus']]]
        
        
    def update_bess_exp(self):
                
        # update SoE and bess set point --> che poi vanno al push push push
        self.p_bess_set.loc[self.time_control] = [[self.p_bess_c[b] for b in range(self.B)]]
        self.p_bess_set_real.loc[self.time_control] = self.P_c[self.bess['bus']-1]
        self.q_bess_set.loc[self.time_control] = [[self.q_bess_c[b] for b in range(self.B)]]
        
        # devo aggiornare lo SoE a fine dell'ora (che mi servirà come previsione per fare lo scheduling in anticipo)
        self.bess['SoE'].loc[self.time_scheduling] = [self.SoE_c[self.Tc,self.bess['bus']]]
              
        
    def run_simulation(self):
        
        self.time_scheduling = self.time_start
        while self.time_scheduling <= self.time_end:
            print(f"\n{self.time_scheduling} scheduling")
            
            # long term forecast 
            self.scheduling_time_index = pd.date_range(start=self.time_scheduling, periods=self.Hor, freq='H')
            self.pv_lt_forecast_sim()
            self.load_lt_forecast()
            
            # scheduling
            self.scheduling()
            scheduling_plot(f"{self.time_scheduling} scheduling",self.S,self.Hor,self.SoE_s,self.bess['SoE_max'],self.P,self.DP.loc[self.scheduling_time_index]['pu'],self.PVs,self.load_h.loc[self.scheduling_time_index],self.bess['bus'],self.Ab,self.ep.loc[self.scheduling_time_index])
            
            # control
            self.time_control = self.time_scheduling
            self.time_control_start = self.time_scheduling
            self.DPr = 0
            while self.time_control < self.time_scheduling+timedelta(hours=1): # one hour of real time control
                #print(f"\n{self.time_control} control") 
                
                # realisation and st forecast
                self.load_realisation_sim()
                self.load_st_forecast_sim()
                self.pv_st_forecast_sim() 
                self.pv_realisation_sim() 
                
                # control
                self.rt_control()
                self.update_bess_sim()
                
                # gcp realisation
                self.gcp_realisation_sim()
                
                # go to next control step
                self.time_control += timedelta(seconds=self.rtc_step) 
            
            # go to next scheduling step
            self.time_scheduling += timedelta(hours=1)
                
          
    def pull_push_test(self,seconds,p_bess):
        # start pulling and pushing data from/to microgrid
        time.sleep(1)
        print("\n")
        print(f"Star pulling and pushing test: p_bess = {p_bess}")
        
        self.p_bess_set_real.iloc[-1,0] = p_bess/self.Ab
        
        self.mcc = True
        manual_control_thread = threading.Thread(target=self.manual_control_check) 
        manual_control_thread.start()
       
        # reset values
        self.P_real = pd.DataFrame(columns=[0,1,2,3,4,5,6,7,8]) # busses active power: continuously updated and reset at each control step
        self.Q_real = pd.DataFrame(columns=[0,1,2,3,4,5,6,7,8]) # busses reactive power: continuously updated and reset at each control step
        self.P_bess_real = pd.DataFrame(columns=[0]) # bess active power: continuously updated and reset at each control step
        self.Q_bess_real = pd.DataFrame(columns=[0]) # bess reactive power: continuously updated and reset at each control step
        self.SoE_min = pd.DataFrame(columns=['pu'])
        self.SoE_max = pd.DataFrame(columns=['pu'])
        
        pull_thread = threading.Thread(target=self.start_pulling)
        pull_thread.start()
        push_thread = threading.Thread(target=self.start_pushing)
        push_thread.start()
        sock_dwd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock_dwd.bind(('128.178.46.54', 35211)) # ip and port
        sock_dwd.settimeout(0.2)  # Timeout di 1 secondo
        data, server = sock_dwd.recvfrom(60000) 
        try:
            self.dwd = json.loads(data.decode('utf-8'))
        except socket.timeout:
            print("DWD pulling is not working !!!")
        sock_dwd.close()
        # stop pulling and pushing data from/to microgrid
        time.sleep(seconds)
        
        print((self.P_real.mean()*self.Ab).round(1))
        self.stop_pulling()
        self.stop_pushing()
        time.sleep(1)
        
        # reset values
        self.P_real = pd.DataFrame(columns=[0,1,2,3,4,5,6,7,8]) # busses active power: continuously updated and reset at each control step
        self.Q_real = pd.DataFrame(columns=[0,1,2,3,4,5,6,7,8]) # busses reactive power: continuously updated and reset at each control step
        self.P_bess_real = pd.DataFrame(columns=[0]) # bess active power: continuously updated and reset at each control step
        self.Q_bess_real = pd.DataFrame(columns=[0]) # bess reactive power: continuously updated and reset at each control step
        self.SoE_min = pd.DataFrame(columns=['pu'])
        self.SoE_max = pd.DataFrame(columns=['pu'])
        self.p_bess_set_real.iloc[-1,0] = 0
        self.mcc = False
        
        
    def save_csv(self,tcs):
        
        tcs = tcs.strftime("%Y-%m-%d_%H-%M-%S")
        
        self.DP_bidded.to_csv(f"results/{tcs}_DP_bidded.csv")
        self.DP_real.to_csv(f"results/{tcs}_DP_real.csv")
        self.p_bess_set.to_csv(f"results/{tcs}_p_bess_set.csv")
        self.p_bess_set_real.to_csv(f"results/{tcs}_p_bess_set_real.csv")        
        self.q_bess_set.to_csv(f"results/{tcs}_q_bess_set.csv")
        self.bess['SoE'].to_csv(f"results/{tcs}_bess_SoE.csv")
        self.P_rtcstep.to_csv(f"results/{tcs}_P_rtcstep.csv")
        self.Q_rtcstep.to_csv(f"results/{tcs}_Q_rtcstep.csv")
        self.P_bess_rtcstep.to_csv(f"results/{tcs}_P_bess_rtcstep.csv")
        self.SoE_mean.to_csv(f"results/{tcs}_SoE_mean.csv")
        
        ### data for scheduling graph
        SoE_df = pd.DataFrame.from_dict(a.SoE_s, orient='index', columns=['value'])
        SoE_df.to_csv(f"results/{tcs}_SoE_s.csv")
        P_df = pd.DataFrame.from_dict(a.SoE_s, orient='index', columns=['value'])
        P_df.to_csv(f"results/{tcs}_P.csv")
        for pv in PVs:
            PVs[pv]['pro'].to_csv(f"results/{tcs}_PVs_{pv}.csv")
        self.DP.loc[self.scheduling_time_index]['pu'].to_csv(f"results/{tcs}_DP.csv")
        self.load_h.loc[self.scheduling_time_index].to_csv(f"results/{tcs}_load_h.csv")
        self.ep.loc[self.scheduling_time_index].to_csv(f"results/{tcs}_ep.csv")

        
    def goto_SoC(self,set_soc):
        
        self.sock_pulling = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_pulling.bind(('128.178.46.54', 35210)) # ip and port from which microgrid measurements are received
        self.sock_pulling.settimeout(0.2)
        self.sock_pushing = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_pushing.settimeout(0.2)
        
        reached = False
        while not reached: # continuosly
            time.sleep(1) # push and pull every 1 second
            
            # pull
            try:
                data, server = self.sock_pulling.recvfrom(20000) 
                soemin = json.loads(data.decode('utf-8'))['Samsung']['Data']['SOC_min']
                soemax = json.loads(data.decode('utf-8'))['Samsung']['Data']['SOC_max']
                soemean = (soemin+soemax)/2
            except socket.timeout:
                print("Pulling is not working !!!")
        
            # control SoC
            if set_soc-1 < soemean < set_soc+1:
                reached = True
            if soemean < set_soc:
                p_set = -20
            else:
                p_set = 20
        
            # push
            try:
                Franci = { 
                    'ZENONE' : {"P_set [kW]": 0,
                                "Q_set [kVAr]": 0},
                     'BESS' : {"P_set [kW]": p_set, 
                               "Q_set [kVAr]": 0}}
                json_string = json.dumps(Franci)
                self.sock_pushing.sendto(json_string.encode('utf-8'), ('192.168.1.86', 33003 )) # ip_to port_to
            except socket.timeout:
                print("Pushing is not working !!!")
                
        self.stop_pulling()
        self.stop_pushing()
        print(f"SoC {set_soc} reached")

          
    def run_rescheduling(self,SoE_realisations):
        
        self.time_scheduling = self.time_start
        
        while self.time_scheduling <= self.time_end:
            
            # long term forecast 
            self.scheduling_time_index = pd.date_range(start=self.time_scheduling, periods=self.Hor, freq='H')
            
            self.pv_lt_forecast_sim()
            self.load_lt_forecast()
            
            # SoE realisation
            self.bess['SoE'].loc[self.time_scheduling] = SoE_realisations.loc[self.time_scheduling]
            
            # scheduling            
            self.scheduling()
            scheduling_plot(f"{self.time_scheduling} scheduling",self.S,self.Hor,self.SoE_s,self.bess['SoE_max'],self.P,self.DP.loc[self.scheduling_time_index]['pu'],self.PVs,self.load_h.loc[self.scheduling_time_index],self.bess['bus'],self.Ab,self.ep.loc[self.scheduling_time_index])
           
            ### save data for scheduling graph
            tcs = self.time_scheduling
            tcs = tcs.strftime("%Y-%m-%d_%H-%M-%S")
            SoE_df = pd.DataFrame.from_dict(a.SoE_s, orient='index', columns=['value'])
            SoE_df.to_csv(f"results/{tcs}_SoE_s.csv")
            P_df = pd.DataFrame.from_dict(a.P, orient='index', columns=['value'])
            P_df.to_csv(f"results/{tcs}_P.csv")
            for pv in self.PVs:
                self.PVs[pv]['pro'].to_csv(f"results/{tcs}_PVs_{pv}.csv")
            self.DP.loc[self.scheduling_time_index]['pu'].to_csv(f"results/{tcs}_DP.csv")
            self.load_h.loc[self.scheduling_time_index].to_csv(f"results/{tcs}_load_h.csv")
            self.ep.loc[self.scheduling_time_index].to_csv(f"results/{tcs}_ep.csv")
            
            
            self.time_scheduling += timedelta(hours=1)
            

            
            


    def run_experimentation(self,hours_of_simulation,lagsched):
        
        # start pulling and pushing data from/to microgrid
        pull_thread = threading.Thread(target=self.start_pulling)
        push_thread = threading.Thread(target=self.start_pushing)
        self.mcc = True
        manual_control_thread = threading.Thread(target=self.manual_control_check) 
        manual_control_thread.start()
        
        ### si parte 5 minuti prima dello scoccare dell'ora per lo scheduling e allo scoccare per il controllo  
        self.timenow = datetime.fromtimestamp(time.time())
        
        self.time_scheduling = self.timenow.replace(minute=0,second=0,microsecond=0) + timedelta(hours=1) # primo scheduling quando scocca l'ora ma lanciato in anticipo di 5 minuti
        self.time_next_scheduling = self.time_scheduling - timedelta(minutes=lagsched)
        
        self.time_control = self.time_scheduling # primo control quando scocca l'ora
        self.time_control_start = self.time_scheduling # aggiornato ogni ora
        
        self.DPr = 0
        
        
        print(f"\nIl primo scheduling partità alle {self.time_next_scheduling}, si ripete ogni ora")
        print(f"Il real-time control partità alle {self.time_control} , si ripete ogni {self.rtc_step} s")
        timestart = self.time_scheduling
        started = False
        while self.timenow <= timestart + timedelta(hours=hours_of_simulation): # per tutta la durata dell'esperimento
            
            time.sleep(0.1) # per non sovraccaricare
            self.timenow = datetime.fromtimestamp(time.time())
            
            
            ### start pulling and pushing
            if not started and self.timenow > self.time_control - timedelta(minutes=lagsched):
                print("\nLet's start pulling and pushing")
                started = True
                pull_thread.start()
                push_thread.start()
                time.sleep(30)
                self.average_realisation() # it will use it as SoC start point for the first scheduling

            ### è passata un'ora! nuovo DP, salva i dati e grafico controllo
            if self.timenow > self.time_control_start + timedelta(hours=1):
                
                print('\nAn hour has passed: new DP to follow!')
                
                # salvataggio dati (comunque quasi tutti sono su Grafana)
                save_thread = threading.Thread(target=self.save_csv, args=(self.time_control_start,))
                save_thread.start()
            
                # grafico controllo 
                cp_thread = threading.Thread(target=control_plot, args=(self.DP_bidded, self.DP_real, self.Ab, self.rtc_step, self.p_bess_set_real.iloc[1:], self.bess['SoE'], self.bess['SoE_max']))
                cp_thread.start()
           
                self.DPr = 0
                self.time_control_start += timedelta(hours=1) 
                
                
            ### fai lo scheuling in thread mentre gira il controllo 3 minuti prima dello scocco dell'ora
            if self.timenow > self.time_next_scheduling: # scheduling 3 minuti prima
                print(f"\n{self.time_next_scheduling} scheduling")
                
                scheduling_thread = threading.Thread(target=self.scheduling_exp)
                scheduling_thread.start()
                self.time_next_scheduling += timedelta(hours=1)
                ## ripartirà tra un'ora 
                
                
            ### fai il control al momento giusto
            if self.timenow > self.time_control:
                
                print(f"\n{self.time_control} control")
                print((self.P_real.mean()*self.Ab).round(1))
                
                # PV and Load realisation and st forecast
                self.average_realisation()
                self.pv_st_forecast_exp() 
                self.load_st_forecast_exp() 
             
                # control
                try:
                    self.rt_control()
                except:
                    "control doesn't work"
                self.update_bess_exp()
                
                # gcp realisation
                self.gcp_realisation_exp()
            
                # set next control time and wait for it
                self.time_control += timedelta(seconds=self.rtc_step) 
                
                

                
        # stop pulling and pushing data from/to microgrid
        time.sleep(2)
        self.mcc = False
        self.stop_pulling()
        self.stop_pushing()
        


    
#%% ###################################################################################################
if __name__=='__main__': ###########################################################################
    
    # INPUT per simulatione ##########################################################################
    
# =============================================================================
#     grid = {'filename':'linedata_AC_test.txt', 'Ab':60000, 'Eb':400}
#     bess = {'bus':6, 'r':0.017, 'SoE_max':60000, 'SoE':30000, 'P_max':20000}
#     load = {'bus':2, 'filename':'12_residential_load_profiles_40s_hourly.csv'}
#     PVs = {'PVFacade': {'bus':8, 'filename_data_h':'PV_PVFacade realisation_h.csv', 'filename_data':'PV_PVFacade realisation.csv'},
#            'Solarmax': {'bus':8, 'filename_data_h':'PV_Solarmax realisation_h.csv', 'filename_data':'PV_Solarmax realisation.csv'},
#            'Perun':    {'bus':6, 'filename_data_h':'PV_Perun realisation_h.csv', 'filename_data':'PV_Perun realisation.csv'}}
#     economic = {'ep_filename':'PUN24a.csv', 'acost':0.05, 'cscinc':0.12, 'unbcost':10}
#     simulation = {'scenarios':40, 'load_fake_realisation_filename':'12_residential_load_profiles_s1_5minutes.csv',
#                   'horizon':24, 'Ks':1, 'Kc':1, 'rtc_step':30, 'start':'2024-06-01 00:00:00', 'end':'2024-06-01 02:00:00'}
#     
# =============================================================================
    grid = {'filename':'linedata_AC_test.txt', 'Ab':60000, 'Eb':400}
    bess = {'bus':6, 'r':0, 'SoE_max':60000, 'SoE':30000, 'P_max':30000}
    load = {'bus':2, 'filename':'12_residential_load_profiles_40s_hourly.csv'}
    PVs = {'PVFacade': {'bus':8, 'filename_data_h':'PV_PVFacade realisation_h.csv', 'filename_data':'PV_PVFacade realisation.csv'},
           'Solarmax': {'bus':8, 'filename_data_h':'PV_Solarmax realisation_h.csv', 'filename_data':'PV_Solarmax realisation.csv'}}
    economic = {'ep_filename':'PUN24a.csv', 'acost':0.15, 'cscinc':0.12, 'unbcost':10}
    simulation = {'scenarios':40, 'load_fake_realisation_filename':'12_residential_load_profiles_s1_5minutes.csv', 'bidding_lag':1,
                  'horizon':24, 'Ks':1, 'Kc':1, 'rtc_step':30, 'start':'2024-06-18 00:00:00', 'end':'2024-06-18 00:00:00'}
    
    
    # RAN ##########################################################################################
    
    a = controller(grid,bess,load,PVs,economic,simulation) # class initialisation
    
    
    
    #### RESCHEDULIN
    # carica gli soe da usare per il rescheduling (quelli dell'esperimento)
# =============================================================================
#     tcs = datetime.strptime('2024-06-27 07:00:00', '%Y-%m-%d %H:%M:%S')
#     tcs = tcs.strftime("%Y-%m-%d_%H-%M-%S")
#     soes = pd.read_csv(f"results/TEST_4final/{tcs}_bess_SoE.csv",index_col=0)
#     soes.index = pd.to_datetime(soes.index)
#     soe = []
#     data = datetime.strptime('2024-06-26 08:00:00', '%Y-%m-%d %H:%M:%S')
#     while data <= datetime.strptime('2024-06-27 07:00:00', '%Y-%m-%d %H:%M:%S'):
#         soe += [soes.loc[data]['pu']]
#         data += timedelta(hours=1)         
#     SoE_realisations = pd.DataFrame(soe,index=pd.date_range(start=a.time_start, end=a.time_end, freq='H'),columns=['pu'])
#     a.run_rescheduling(SoE_realisations)
# =============================================================================
    

    ### SIMULATIONS
   # a.run_simulation() # simulation
    #control_plot(a.DP_bidded,a.DP_real,a.Ab,a.rtc_step,a.p_bess_set_real,a.bess['SoE'],a.bess['SoE_max'],sim=True)
    #a.save_csv(a.time_control_start)
    




#%% simulationi varie per sistemare i buchi!!! non dovrebbero piu servire
# =============================================================================
#     tcs = datetime.strptime('2024-06-27 04:00:00', '%Y-%m-%d %H:%M:%S')
#     tcs = tcs.strftime("%Y-%m-%d_%H-%M-%S")
#     DP_bidded = pd.read_csv(f"results/TEST_4/{tcs}_DP_bidded.csv",index_col=0)
#     DP_bidded.index = pd.to_datetime(DP_bidded.index)
#     DP_real = pd.read_csv(f"results/TEST_4/{tcs}_DP_real.csv",index_col=0)
#     DP_real.index = pd.to_datetime(DP_real.index)
#     p_bess_set_real = pd.read_csv(f"results/TEST_4/{tcs}_p_bess_set_real.csv",index_col=0)    
#     p_bess_set_real.index = pd.to_datetime(p_bess_set_real.index)    
#     soes = pd.read_csv(f"results/TEST_4/{tcs}_bess_SoE.csv",index_col=0)
#     soes.index = pd.to_datetime(soes.index)
#     soes.sort_index(inplace=True)
#     DP_bidded.sort_index(inplace=True)
#     DP_real.sort_index(inplace=True)
#     p_bess_set_real.sort_index(inplace=True)
#     
#     
#     
#     # tappo1 Will
#     tcs = datetime.strptime('2024-06-26 14:00:00', '%Y-%m-%d %H:%M:%S')
#     tcs = tcs.strftime("%Y-%m-%d_%H-%M-%S")
#     DP_real_tappo1 = pd.read_csv(f"results/TEST_4X/{tcs}_DP_real_tappo1.csv",index_col=0)
#     DP_real_tappo1.index = pd.to_datetime(DP_real_tappo1.index)
#     DP_real.update(DP_real_tappo1)
#     p_bess_set_real_tappo1 = pd.read_csv(f"results/TEST_4X/{tcs}_p_bess_set_real_tappo1.csv",index_col=0)    
#     p_bess_set_real_tappo1.index = pd.to_datetime(p_bess_set_real_tappo1.index)   
#     p_bess_set_real.update(p_bess_set_real_tappo1)
#     
#     
#     ### tappa 2 buchi interpolando linearmente in soes
#     # Converte la colonna timestamp in formato datetime
#     # Funzione per interpolare valori su un intervallo
#     def interpolate_interval(data, start_time, end_time):
#         # Ottieni i valori ai tempi di inizio e fine
#         start_value = data.loc[data.index == start_time, 'pu'].values[0]
#         end_value = data.loc[data.index == end_time, 'pu'].values[0]
#         
#         # Crea una maschera per l'intervallo
#         mask = (data.index >= start_time) & (data.index <= end_time)
#         
#         # Calcola il numero di passi nell'intervallo
#         steps = mask.sum()
#         
#         # Genera valori interpolati linearmente
#         interpolated_values = np.linspace(start_value, end_value, steps)
#         
#         # Sostituisci i valori nell'intervallo con i valori interpolati
#         data.loc[mask, 'pu'] = interpolated_values
#     
#     # Definisci gli intervalli
#     intervals = [
#         ("2024-06-26 14:00:00", "2024-06-26 15:00:00")
#     ]
#     
#     # Applica l'interpolazione per ogni intervallo
#     for start_time, end_time in intervals:
#         interpolate_interval(soes, pd.to_datetime(start_time), pd.to_datetime(end_time))
#         
#     control_plot(DP_bidded,DP_real,a.Ab,a.rtc_step,p_bess_set_real.iloc[1:],soes,a.bess['SoE_max'])
#         
#         
#     ### ultime 3 ore simulate
#     tcs = datetime.strptime('2024-06-27 07:00:00', '%Y-%m-%d %H:%M:%S')
#     tcs = tcs.strftime("%Y-%m-%d_%H-%M-%S")
#     DP_bidded3 = pd.read_csv(f"results/TEST_4X/{tcs}_DP_bidded.csv",index_col=0)
#     DP_bidded3.index = pd.to_datetime(DP_bidded3.index)
#     DP_real3 = pd.read_csv(f"results/TEST_4X/{tcs}_DP_real.csv",index_col=0)
#     DP_real3.index = pd.to_datetime(DP_real3.index)
#     p_bess_set_real3 = pd.read_csv(f"results/TEST_4X/{tcs}_p_bess_set_real.csv",index_col=0)    
#     p_bess_set_real3.index = pd.to_datetime(p_bess_set_real3.index)    
#     soes3 = pd.read_csv(f"results/TEST_4X/{tcs}_bess_SoE.csv",index_col=0)
#     soes3.index = pd.to_datetime(soes3.index)
#     
#     soes = soes3.combine_first(soes)
#     DP_bidded = DP_bidded3.combine_first(DP_bidded)
#     DP_real = DP_real3.combine_first(DP_real)
#     p_bess_set_real = p_bess_set_real3.combine_first(p_bess_set_real)
#     
#     control_plot(DP_bidded,DP_real,a.Ab,a.rtc_step,p_bess_set_real.iloc[1:],soes,a.bess['SoE_max'],sim=2)
#     
#     
#     soes.to_csv
#     DP_bidded = DP_bidded3.combine_first(DP_bidded)
#     DP_real = DP_real3.combine_first(DP_real)
#     p_bess_set_real = p_bess_set_real3.combine_first(p_bess_set_real)
#     
#     
#     DP_bidded.to_csv(f"results/{tcs}_DP_bidded.csv")
#     DP_real.to_csv(f"results/{tcs}_DP_real.csv")
#     p_bess_set_real.to_csv(f"results/{tcs}_p_bess_set_real.csv")        
#     soes.to_csv(f"results/{tcs}_bess_SoE.csv")
# =============================================================================







#%% POST PROCESSING CON I DATI SALVATIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII


    # grafico control
    import pandas as pd
    tcs = datetime.strptime('2024-06-27 07:00:00', '%Y-%m-%d %H:%M:%S')
    tcs = tcs.strftime("%Y-%m-%d_%H-%M-%S")
    DP_bidded = pd.read_csv(f"results/TEST_4final/{tcs}_DP_bidded.csv",index_col=0)
    DP_real = pd.read_csv(f"results/TEST_4final/{tcs}_DP_real.csv",index_col=0)
    p_bess_set_real = pd.read_csv(f"results/TEST_4final/{tcs}_p_bess_set_real.csv",index_col=0)        
    soes = pd.read_csv(f"results/TEST_4final/{tcs}_bess_SoE.csv",index_col=0)
    
    load = pd.read_csv("input/12_residential_load_profiles_s1_5minutes.csv",index_col=0)
    pv = pd.read_csv("input/PV_PVFacade realisation.csv",index_col=0)
    pv2 = pd.read_csv("input/PV_Solarmax realisation.csv",index_col=0)
    load.index = pd.to_datetime(load.index)
    pv.index = pd.to_datetime(pv.index)
    pv2.index = pd.to_datetime(pv2.index)
    pv = pv + pv2
    pv=-pv
    pv = pv.shift(freq='1H')
    load = load.loc[datetime.strptime('2024-06-26 08:00:00', '%Y-%m-%d %H:%M:%S'):datetime.strptime('2024-06-27 07:00:00', '%Y-%m-%d %H:%M:%S')]
    pv = pv.loc[pd.to_datetime('2024-06-26 08:00:00').tz_localize('UTC'):pd.to_datetime('2024-06-27 07:00:00').tz_localize('UTC')]
    control_plot_final(DP_bidded,DP_real,a.Ab,a.rtc_step,p_bess_set_real.iloc[1:],soes,a.bess['SoE_max'],pv,load,sim=2)
    
    
    # grafico scheduling
    tcs = datetime.strptime('2024-06-26 08:00:00', '%Y-%m-%d %H:%M:%S')
    tcs = tcs.strftime("%Y-%m-%d_%H-%M-%S")
    SoE_s = pd.read_csv(f"results/TEST_4final/{tcs}_SoE_s.csv",index_col=0)
    SoE_s.index = SoE_s.index.map(eval)
    P = pd.read_csv(f"results/TEST_4final/{tcs}_P.csv",index_col=0)
    P.index = P.index.map(eval)
    PVss = {}
    for pv in a.PVs:
        PVss[pv] = {}
        PVss[pv]['pro'] = pd.read_csv(f"results/TEST_4final/{tcs}_PVs_{pv}.csv",index_col=0)
    DP = pd.read_csv(f"results/TEST_4final/{tcs}_DP.csv",index_col=0)
    load_h = pd.read_csv(f"results/TEST_4final/{tcs}_load_h.csv",index_col=0)
    load_h.index = pd.to_datetime(load_h.index)
    ep = pd.read_csv(f"results/TEST_4final/{tcs}_ep.csv",index_col=0)
    scheduling_plot(f"{tcs} scheduling",a.S,a.Hor,SoE_s,a.bess['SoE_max'],P,DP,PVss,load_h,a.bess['bus'],a.Ab,ep,fromdatasaved=True)


    
    from datetime import datetime, timedelta
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Initialize starting time
    tcs = datetime.strptime('2024-06-26 08:00:00', '%Y-%m-%d %H:%M:%S')
    
    plt.figure(dpi=1000, figsize=(9, 3))
    step = 3
    for t in range(24):
        tcss = tcs.strftime("%Y-%m-%d_%H-%M-%S")
        dp = pd.read_csv(f"results/TEST_4final/{tcss}_DP.csv", index_col=0, parse_dates=True)
        tcs += timedelta(hours=1)
        if t % step == 0:
            plt.plot(dp.index, dp['pu'] * 60000, label=dp.index[0].strftime('%H'))
    
    # Customize x-axis
    xticks = pd.date_range(start='2024-06-26 08:00', end='2024-06-28 08:00', freq=f"{step}H")
    xticks_labels = [d.strftime('%H') for d in xticks]
    
    plt.gca().set_xticks(xticks)
    plt.gca().set_xticklabels(xticks_labels)
    
    # Additional plot settings
    plt.legend()
    plt.grid()
    plt.xlim(xticks[0],xticks[-1])
    plt.ylabel("DP [kWh]")
    plt.show()







    

    
    
   
