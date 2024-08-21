"""
scheduling iDistFlow 
"""

import pandas as pd 
import numpy as np
import gurobipy as gp
import math

def scheduling_iDistFlow(pco,qco,vco,vap,grid,p_pv,p_load,q_pv,q_load,bess_bus,SoE_0,SoE_max,bess_Pmax,EP,acost,cscinc,unbcost,Ab,CoDistFlow=False):
    ## all input should be in [pu]
    ## all variables are in [pu]
    
    # Adjacency matrix for grid modelling
    G = np.zeros((len(grid), len(grid)))
    for l, row in grid.iterrows():
        if l != 0:
            G[row['busup']-1,l] = 1   
        
    # Model creation
    m = gp.Model("iDistFlow")
    
    # Problem size
    T = p_pv.shape[0] # Time horizon
    S = p_pv.shape[1] # Scenarios
    L = len(grid) # Lines
    B = L+1 # Busses

    # Controllable varaible definition
    DP = m.addVars(T, lb=-1, ub=1, vtype=gp.GRB.CONTINUOUS, name="DP") # Dispatch Plan at slack bus / GCP / bus 0 [W --> pu]
    p_bess = m.addVars(S,T,B, lb=-bess_Pmax, ub=bess_Pmax, vtype=gp.GRB.CONTINUOUS, name="p_bess") # Batteries scheduling [W --> pu]
    q_bess = m.addVars(S,T,B, lb=-bess_Pmax, ub=bess_Pmax, vtype=gp.GRB.CONTINUOUS, name="q_bess") # Batteries scheduling [W --> pu]

    # Auxiliary variables definition
    v =  m.addVars(S,T,B, lb=0.9025, ub=1.1025, vtype=gp.GRB.CONTINUOUS, name="v") # square of bus voltage [V**2 --> pu]
    #v =  m.addVars(S,T,B, lb=0.5025, ub=1.7025, vtype=gp.GRB.CONTINUOUS, name="v") # square of bus voltage [V**2 --> pu] # relax to test
    ia =  m.addVars(S,T,L, lb=-max(grid['ampacity[pu]']), ub=max(grid['ampacity[pu]']), vtype=gp.GRB.CONTINUOUS, name="ia") # active current in the line [A --> pu]
    ir =  m.addVars(S,T,L, lb=-max(grid['ampacity[pu]']), ub=max(grid['ampacity[pu]']), vtype=gp.GRB.CONTINUOUS, name="ir") # reactive current in the line [A --> pu]

    P = m.addVars(S,T,L, lb=-1, ub=1, vtype=gp.GRB.CONTINUOUS, name="P")  # Active power in each line
    P0a = m.addVars(S,T, lb=0, vtype=gp.GRB.CONTINUOUS, name="P0a")  # varaible to relax Power Factor constraint
    P0b =  m.addVars(S,T, lb=0, vtype=gp.GRB.CONTINUOUS, name="P0b") # variable to relax Power Factor constraint
    Q = m.addVars(S,T,L, lb=-1, ub=1, vtype=gp.GRB.CONTINUOUS, name="Q")  # Reactive power in each line
    SoE = m.addVars(S,T+1,B, lb=SoE_max*0.1, ub=SoE_max*0.9, vtype=gp.GRB.CONTINUOUS, name="SoE") # BESS State of Energy [Wh  --> pu]
    abs_p_bess = m.addVars(S,T,B, lb=0, ub=1, vtype=gp.GRB.CONTINUOUS, name="abs_p_bess") # Absolute value of p_bess [Wh --> pu]
    ch_p_bess = m.addVars(S,T,B, lb=0, ub=1, vtype=gp.GRB.CONTINUOUS, name="ch_p_bess") # Positive value of p_bess [Wh --> pu]
    ch = m.addVars(S,T,B, vtype=gp.GRB.BINARY, name="ch") # auxiliary variable for ch_p_bess [1/0]
    csc = m.addVars(S,T, lb=0, ub=1, vtype=gp.GRB.CONTINUOUS, name="csc") # collective self-consumption [Wh --> puv]
    unb = m.addVars(S,T, lb=-1, ub=1, vtype=gp.GRB.CONTINUOUS, name="unb") # unbalance / dispatch error [Wh --> pu]
    abs_unb = m.addVars(S,T, lb=0, ub=1, vtype=gp.GRB.CONTINUOUS, name="abs_unb") # unbalance / dispatch error [Wh --> pu]
    p_pv_neg = m.addVars(S,T,B, lb=-1, ub=0, vtype=gp.GRB.CONTINUOUS, name="p_pv_neg") # negative value of p_pv [Wh --> pu]
    neg = m.addVars(S,T,B, vtype=gp.GRB.BINARY, name="neg") # auxiliary variable for p_pv_neg [1/0]

    # Objective funtion
    AC = acost * gp.quicksum(abs_p_bess[s,t,b] for s in range(S) for t in range(T) for b in range(B)) / 2 / S
    INC = cscinc * gp.quicksum(csc[s,t] for s in range(S) for t in range(T)) / S
    DIS = unbcost * gp.quicksum(abs_unb[s,t] for s in range(S) for t in range(T)) / S
    EA = -gp.quicksum(DP[t]*EP.iloc[t,0] for t in range(T))
    PF = 1e-20 * gp.quicksum((P0a[s,t]**2 + P0b[s,t]**2) for s in range(S) for t in range(T)) / S # Power Factor constraint
    
    m.setObjective((EA-AC+INC-DIS-PF), gp.GRB.MAXIMIZE)
    
    # Constraints
    for s in range(S):
        m.addConstr(SoE[s,0,bess_bus] == SoE_0) 
        for t in range(T):
            m.addConstr(abs_unb[s,t] >= unb[s,t])
            m.addConstr(abs_unb[s,t] >= -unb[s,t])
            m.addConstr(unb[s,t] == (P[s,t,0]-DP[t]))                 
            m.addConstr(csc[s,t] <= gp.quicksum(ch_p_bess[s,t,b] + p_load.iloc[t,s][b] for b in range(B)))
            m.addConstr(csc[s,t] <= -gp.quicksum(p_pv_neg[s,t,b] for b in range(B)))
            
            # bess definition constraints
            for b in range(B):
                m.addConstr(SoE[s,t+1,b] == (SoE[s,t,b] + p_bess[s,t,b])) 
                m.addConstr(abs_p_bess[s,t,b] >= p_bess[s,t,b])
                m.addConstr(abs_p_bess[s,t,b] >= -p_bess[s,t,b])
                m.addConstr(ch_p_bess[s,t,b] == p_bess[s,t,b]*ch[s,t,b])
                m.addConstr(p_pv_neg[s,t,b] == p_pv.iloc[t,s][l]*neg[s,t,b])
                m.addConstr(ch_p_bess[s,t,b] >= 0)
                m.addConstr(p_pv_neg[s,t,b] <= 0)
                
                if b != bess_bus:
                    m.addConstr(p_bess[s,t,b] == 0) # no BESS in that bus 
                    m.addConstr(q_bess[s,t,b] == 0) # no BESS in that bus 
                    
                if b == bess_bus: # q non controllable but converter depending
                    m.addConstr(q_bess[s,t,b] == 0.0417*p_bess[s,t,b] + 2395.827/Ab)
                
                    
            # grid constraints (eq 1-8)
            for l in range(L):
                m.addConstr( P[s,t,l] == p_pv.iloc[t,s][l+1] + p_load.iloc[t,s][l+1] + p_bess[s,t,l+1] + gp.quicksum(G[l][k]*P[s,t,k] for k in range(L)) + pco[s,t,l])
                m.addConstr( Q[s,t,l] == q_pv.iloc[t,s][l+1] + q_load.iloc[t,s][l+1] + q_bess[s,t,l+1] + gp.quicksum(G[l][k]*Q[s,t,k] for k in range(L)) + qco[s,t,l])
                m.addConstr( v[s,t,grid.loc[l]['busdown']] == v[s,t,grid.loc[l]['busup']] - 2*(grid.loc[l]['r[pu]']*P[s,t,l]+grid.loc[l]['x[pu]']*Q[s,t,l]) + vco[s,t,l])
                m.addConstr( (ia[s,t,l]**2+ir[s,t,l]**2) <= grid.loc[l]['ampacity[pu]']**2)
                m.addConstr(  ia[s,t,l] == P[s,t,l]/vap[s,t,grid.loc[l]['busup']])
                m.addConstr(  ir[s,t,l] == Q[s,t,l]/vap[s,t,grid.loc[l]['busup']])
            m.addConstr( v[s,t,0] == 1) 
            m.addConstr( P[s,t,0] == P0a[s,t]-P0b[s,t])
            #m.addConstr( P0a[s,t]+P0b[s,t] >= Q[s,t,0]*math.tan(math.pi/2-math.acos(0.1)))
            #m.addConstr( P0a[s,t]+P0b[s,t] >= -Q[s,t,0]*math.tan(math.pi/2-math.acos(0.1)))
            
    # Solving model
    m.setParam('LogToConsole',0)
    m.optimize()
    
    # Check convergence
    if m.status == gp.GRB.OPTIMAL:
        pass
        #print("scheduling: iDistFlow convergence")
    else:
        print("scheduling: iDistFlow non-convergence") 
        
    # Returns solutions
    p_bess = m.getAttr("x",p_bess)
    q_bess = m.getAttr("x",q_bess)
    SoE = m.getAttr("x",SoE)
    P = m.getAttr("x",P)
    Q = m.getAttr("x",Q)
    DP = m.getAttr("x",DP)
    v = m.getAttr("x",v)
    ia = m.getAttr("x",ia)
    ir = m.getAttr("x",ir)
    AC=AC.getValue()
    INC=INC.getValue()
    DIS=DIS.getValue()
    EA=EA.getValue()
    PF=PF.getValue()

    return(p_bess,q_bess,DP,P,SoE)


#%%#######################################################################################################
if __name__=="__main__": 
    pass
    
    
