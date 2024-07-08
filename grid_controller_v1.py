# -*- coding: utf-8 -*-
"""
grid controller per le simulationi
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from PV_forecast2 import pv_production2, empirical_model
from scheduling_iDistFlow import scheduling_iDistFlow
from plots_controller import scheduling_plot, control_plot
from control_iDistFlow import control_iDistFlow, fake_DPr
from LoadFlow import solve_Load_flow
import PerUnit 

class controller:
    
    def __init__(self,grid,bess,load,PVs,economic,simulation):
        
        # grid and bess (only one bess is possible at the moment) (a virtual line is added to the grid to simulate battery-related losses)
        self.grid = pd.DataFrame(pd.read_csv(grid['filename'],delim_whitespace=True, skipinitialspace=True)).rename_axis("line")
        self.bess = bess
        self.grid.loc[len(self.grid)] = {'busup':self.bess['bus'],'busdown':len(self.grid)+1,'r[ohm]':self.bess['r'],'x[ohm]':0,'B(S)':0,'/':1,'ampacity[A]':1e3,'length[km]':0}
        self.bess['bus'] = len(self.grid)
        self.L = len(self.grid) # number of lines
        self.B = self.L+1  # number of busses
        self.Ab = grid['Ab'] # per unit power base [W]
        self.Eb = grid['Eb'] # per unit voltage base and GCP constrain [V]
        self.bess['SoE_max'] = self.bess['SoE_max']/self.Ab
        self.grid = PerUnit.to_pu_grid(self.grid, self.Ab, self.Eb)
        
        # simulation
        self.S = simulation['scenarios']   
        self.time_start = datetime.strptime(simulation['start'], '%Y-%m-%d %H:%M:%S')
        self.time_end = datetime.strptime(simulation['end'], '%Y-%m-%d %H:%M:%S')
        self.Ks = simulation['Ks']
        self.Kc = simulation['Kc']
        self.Hor = simulation['horizon']
        self.rtc_step = simulation['rtc_step']
        
        # load (only one load is possibile at the moment)
        self.loadbus = load['bus']
        self.load_h = pd.read_csv(load['filename'],index_col=0)/self.Ab
        self.load_h.index = pd.to_datetime(self.load_h.index)
        self.load_rtcstep = pd.read_csv(simulation['load_fake_realisation_filename'],index_col=0)/self.Ab
        self.load_rtcstep.index = pd.to_datetime(self.load_rtcstep.index)
        self.load_rtcstep = self.load_rtcstep.resample(f"{int(self.rtc_step)}S").mean()

        # PVs
        self.PVs = PVs
        for pv in PVs:
            self.PVs[pv]['lt_forecast_model'] = empirical_model(PVs[pv]['filename_data_h'])
            self.PVs[pv]['fake_realisation'] = pd.read_csv(PVs[pv]['filename_data'],index_col=0)
            self.PVs[pv]['fake_realisation'].index = pd.to_datetime(self.PVs[pv]['fake_realisation'].index).tz_localize(None)
            if self.rtc_step > 120:
                self.PVs[pv]['fake_realisation'] = self.PVs[pv]['fake_realisation'].resample(f"{int(self.rtc_step)}S").mean()
            else:
                self.PVs[pv]['fake_realisation'] = self.PVs[pv]['fake_realisation'].resample(f"{int(self.rtc_step)}S").ffill()
        
        # economic
        self.ep = pd.read_csv(economic['ep_filename'],index_col=0)/1e6 # [€/MWh] -> [€/Wh]
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
        self.p_bess_set = pd.DataFrame(columns=["pu"])
        self.q_bess_set = pd.DataFrame(columns=["pu"])
        self.bess['SoE'] = pd.DataFrame([self.bess['SoE']/self.Ab], index=[self.time_start], columns=["pu"])
        
        
    def pv_lt_forecast(self):
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
    
    
    def pv_st_forecast(self):
        self.p_pv_st = [0]*self.B
        self.q_pv_st = [0]*self.B
        for pv in self.PVs:
            self.p_pv_st[self.PVs[pv]['bus']] += - self.PVs[pv]['fake_realisation'].loc[self.time_scheduling:self.time_control]['power [W]'].mean() / self.Ab # moving average
        
        
    def load_lt_forecast(self):
        self.p_load = pd.DataFrame(index=self.scheduling_time_index, columns=[f"load_{i} [W]" for i in range(self.S)])
        self.q_load = pd.DataFrame(index=self.scheduling_time_index, columns=[f"load_{i} [W]" for i in range(self.S)])
        self.p_load = self.p_load.map(lambda x: [0]*self.B)   
        self.q_load = self.q_load.map(lambda x: [0]*self.B) 
        for s in range(self.S):
            for h in range(self.Hor):
                self.p_load.iloc[h,s][self.loadbus] = self.load_h.loc[self.scheduling_time_index].iloc[h,s] # set load bus injection [pu]
                self.q_load.iloc[h,s][self.loadbus] = self.p_load.iloc[h,s][self.loadbus]*0.484 # (cosphi=0.9)
        
    
    def load_st_forecast(self):
        self.p_load_st = [0]*self.B
        self.q_load_st = [0]*self.B
        self.p_load_st[self.loadbus] = self.load_rtcstep.loc[self.time_scheduling:self.time_control]['P [W]'].mean() # moving average
        self.q_load_st[self.loadbus] = self.p_load_st[self.loadbus]*0.484 # (cosphi=0.9)
        
    
    def pv_realisation(self):
        self.p_pv_re = [0]*self.B
        self.q_pv_re = [0]*self.B
        for pv in self.PVs:
            self.p_pv_re[self.PVs[pv]['bus']] += - self.PVs[pv]['fake_realisation'].loc[self.time_control]['power [W]'] / self.Ab
        
        
    def load_realisation(self):
        self.p_load_re = [0]*self.B
        self.q_load_re = [0]*self.B
        self.p_load_re[self.loadbus] = self.load_rtcstep.loc[self.time_control]['P [W]'] # already in pu
        self.q_load_re[self.loadbus] = self.p_load_re[self.loadbus]*0.484 # (cosphi=0.9)
        
        
    def scheduling(self): 
        self.p_bess,self.q_bess,self.DP, self.P, self.SoE_s = scheduling_iDistFlow(self.pco,self.qco,self.vco,self.vap,self.grid,self.p_pv,self.p_load,self.q_pv,self.q_load,self.bess['bus'],self.bess['SoE'].loc[self.time_scheduling]['pu'],self.bess['SoE_max'],self.ep.loc[self.scheduling_time_index],self.acost,self.cscinc,self.unbcost)
        for ks in range(1,self.Ks+1): 
            for s in range(self.S):
                for h in range(self.Hor):
                    self.pco[s,h],self.qco[s,h],self.vco[s,h],self.vap[s,h] = solve_Load_flow(self.grid,self.p_pv.iloc[h,s],self.q_pv.iloc[h,s],self.p_load.iloc[h,s],self.q_load.iloc[h,s],[self.p_bess[(s, h, b)] for b in range(self.B)],[self.q_bess[(s, h, b)] for b in range(self.B)],coDistFlow=True)
            self.p_bess,self.q_bess,self.DP, self.P, self.SoE_s = scheduling_iDistFlow(self.pco,self.qco,self.vco,self.vap,self.grid,self.p_pv,self.p_load,self.q_pv,self.q_load,self.bess['bus'],self.bess['SoE'].loc[self.time_scheduling]['pu'],self.bess['SoE_max'],self.ep.loc[self.scheduling_time_index],self.acost,self.cscinc,self.unbcost)
    
    
    def rt_control(self):
        self.DP_bidded.loc[self.time_control] = self.DP[0]
        Tc = int(((self.time_scheduling+timedelta(hours=1)-self.time_control).seconds)/self.rtc_step) # problem size
        self.p_bess_c,self.q_bess_c,SoE_c = control_iDistFlow(self.DP[0]*3600,self.DPr,Tc,self.rtc_step,self.pco_c,self.qco_c,self.vco_c,self.vap_c,self.grid,self.p_pv_st,self.p_load_st,self.q_pv_st,self.q_load_st,self.bess['bus'],self.bess['SoE'].loc[self.time_control]['pu'],self.bess['SoE_max'])
        for kc in range(1,self.Kc+1):
            self.pco_c,self.qco_c,self.vco_c,self.vap_c = solve_Load_flow(self.grid,self.p_pv_st,self.q_pv_st,self.p_load_st,self.q_load_st,self.p_bess_c,self.q_bess_c,coDistFlow=True)
            self.p_bess_c,self.q_bess_c,SoE_c = control_iDistFlow(self.DP[0]*3600,self.DPr,Tc,self.rtc_step,self.pco_c,self.qco_c,self.vco_c,self.vap_c,self.grid,self.p_pv_st,self.p_load_st,self.q_pv_st,self.q_load_st,self.bess['bus'],self.bess['SoE'].loc[self.time_control]['pu'],self.bess['SoE_max'])

        # update SoE and bess set point
        self.p_bess_set.loc[self.time_control] = [[self.p_bess_c[b] for b in range(self.B)]]
        self.q_bess_set.loc[self.time_control] = [[self.q_bess_c[b] for b in range(self.B)]]
        self.bess['SoE'].loc[self.time_control+timedelta(seconds=self.rtc_step)] = [SoE_c[1,self.bess['bus']]]
       
        # see realisation
        self.DP_real.loc[self.time_control] = fake_DPr(self.grid, self.p_pv_re, self.q_pv_re, self.p_load_re, self.q_load_re, self.p_bess_set.loc[self.time_control]['pu'], self.q_bess_set.loc[self.time_control]['pu']) # DP realised at step-1
        self.DPr += self.DP_real.loc[self.time_control]['pu']*self.rtc_step
              
        
    def run_simulation(self):
        
        self.time_scheduling = self.time_start
        while self.time_scheduling <= self.time_end:
            print(f"\n{self.time_scheduling} scheduling")
            
            # long term forecast 
            self.scheduling_time_index = pd.date_range(start=self.time_scheduling, periods=self.Hor, freq='H')
            self.pv_lt_forecast()
            self.load_lt_forecast()
            
            # scheduling
            self.scheduling()
            scheduling_plot(f"{self.time_scheduling} scheduling",self.S,self.Hor,self.SoE_s,self.P,self.DP,self.PVs,self.load_h.loc[self.scheduling_time_index],self.bess['bus'],self.Ab,self.ep.loc[self.scheduling_time_index])
            
            self.time_control = self.time_scheduling
            self.DPr = 0
            while self.time_control < self.time_scheduling+timedelta(hours=1): # one hour of real time control
                print(f"{self.time_control} control") 
                
                # realisation and st forecast
                self.load_realisation()
                self.load_st_forecast()
                self.pv_st_forecast() 
                self.pv_realisation() 
                
                # control
                self.rt_control()
                self.time_control += timedelta(seconds=self.rtc_step) 
                
            self.time_scheduling += timedelta(hours=1)
                
                
    def run_experimentation(self):
        pass
    
    def pulling(self):
        pass
    
    def pushing(self):
        pass
    
    
    
####################################################################################################
if __name__=='__main__': ###########################################################################
    
    # INPUT ########################################################################################
    
    grid = {'filename':'linedata_AC_test.txt', 'Ab':40000, 'Eb':400}
    bess = {'bus':4, 'r':0.017, 'SoE_max':60000, 'SoE':30000}
    load = {'bus':2, 'filename':'12_residential_load_profiles_50s_hourly.csv'}
    PVs = {'PVFacade': {'bus':8, 'filename_data_h':'PV_PVFacade realisation_h.csv', 'filename_data':'PV_PVFacade realisation.csv'},
           'Solarmax': {'bus':8, 'filename_data_h':'PV_Solarmax realisation_h.csv', 'filename_data':'PV_Solarmax realisation.csv'},
           'Perun':    {'bus':6, 'filename_data_h':'PV_Perun realisation_h.csv', 'filename_data':'PV_Perun realisation.csv'}}
    economic = {'ep_filename':'PUN234.csv', 'acost':0.10, 'cscinc':0.12, 'unbcost':0.3}
    simulation = {'scenarios':10, 'load_fake_realisation_filename':'12_residential_load_profiles_s1_seconds.csv',
                  'horizon':24, 'Ks':1, 'Kc':1, 'rtc_step':10, 'start':'2024-06-01 00:00:00', 'end':'2024-06-01 00:00:00'}
    
    # RAN ##########################################################################################
    
    a = controller(grid,bess,load,PVs,economic,simulation) # class initialisation
    a.run_simulation() # simulation
    
    control_plot(a.DP_bidded,a.DP_real,a.Ab,a.rtc_step,a.p_bess_set,a.bess['SoE'])
    