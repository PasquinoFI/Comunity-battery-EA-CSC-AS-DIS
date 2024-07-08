"""
Real time control
"""

from datetime import datetime, timedelta
from LoadFlow import solve_Load_flow
import gurobipy as gp
import numpy as np
import math


def control_iDistFlow(DPb,DPr,T,step,pco,qco,vco,vap,grid,p_pv,p_load,q_pv,q_load,bess_bus,SoE_0,SoE_max,bess_Pmax,Ab):
    
    # Adjacency matrix for grid modelling
    G = np.zeros((len(grid), len(grid)))
    for l, row in grid.iterrows():
        if l != 0:
            G[row['busup']-1,l] = 1 
    
    # Model creation
    m = gp.Model("dispatch")
    
    # Problem size
    L = len(grid) # Lines
    B = L+1 # Busses

    # Controllable varaible definition
    p_bess = m.addVars(B, lb=-bess_Pmax, ub=bess_Pmax, vtype=gp.GRB.CONTINUOUS, name="p_bess") # Batteries scheduling [W --> pu]
    q_bess = m.addVars(B, lb=-bess_Pmax, ub=bess_Pmax, vtype=gp.GRB.CONTINUOUS, name="q_bess") # Batteries scheduling [W --> pu]

    # Auxiliary variables definition
    P = m.addVars(L, lb=-1, ub=1, vtype=gp.GRB.CONTINUOUS, name="P")  # Active power in each line
    Q = m.addVars(L, lb=-1, ub=1, vtype=gp.GRB.CONTINUOUS, name="Q")  # Reactive power in each line
    P0a = m.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="P0a")  # varaible to relax Power Factor constraint
    P0b =  m.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="P0b") # variable to relax Power Factor constraint
    SoE = m.addVars(T+1,B, lb=SoE_max*0.1, ub=SoE_max*0.9, vtype=gp.GRB.CONTINUOUS, name="SoE") # BESS State of Energy [Wh --> pu]
    v =  m.addVars(B, lb=0.9025, ub=1.1025, vtype=gp.GRB.CONTINUOUS, name="v") # square of bus voltage [V**2 --> pu]
    ia =  m.addVars(L, lb=-max(grid['ampacity[pu]']), ub=max(grid['ampacity[pu]']), vtype=gp.GRB.CONTINUOUS, name="ia") # active current in the line [A --> pu]
    ir =  m.addVars(L, lb=-max(grid['ampacity[pu]']), ub=max(grid['ampacity[pu]']), vtype=gp.GRB.CONTINUOUS, name="ir") # reactive current in the line [A] --> pu

    DPo = m.addVar(lb=-1e5, ub=1e5, vtype=gp.GRB.CONTINUOUS, name="DPo") # dispatch optimized [Wh]
    DP_error = m.addVar(lb=-1e5, ub=1e5, vtype=gp.GRB.CONTINUOUS, name="DP_error") # dispatch error [Wh]
    abs_DP_error = m.addVar(lb=0, ub=1e5, vtype=gp.GRB.CONTINUOUS, name="abs_DP_error") # abs dispatch error [Wh]
    
    # Objective funtion
    m.setObjective(abs_DP_error, gp.GRB.MINIMIZE)
    
    # Constraints
    m.addConstr(DPo == P[0]*T*step)
    m.addConstr(DP_error == DPb -DPr -DPo)
    m.addConstr(abs_DP_error >= DP_error)
    m.addConstr(abs_DP_error >= -DP_error)
    m.addConstr(SoE[0,bess_bus] == SoE_0) 

    for b in range(B):
        for t in range(T):
            m.addConstr(SoE[t+1,b] == (SoE[t,b] + p_bess[b]*step/(3600)))
     
        if b != bess_bus:
            m.addConstr(p_bess[b] == 0) # no BESS in that bus 
            
        if b == bess_bus: # q non controllable but converter depending
            m.addConstr(q_bess[b] == 0.0417*p_bess[b] + 2395.827/Ab)
                
    # grid constraints (eq 1-8)
    for l in range(L):
        m.addConstr( P[l] == p_pv[l+1] + p_load[l+1] + p_bess[l+1] + gp.quicksum(G[l][k]*P[k] for k in range(L)) + pco[l])
        m.addConstr( Q[l] == q_pv[l+1] + q_load[l+1] + q_bess[l+1] + gp.quicksum(G[l][k]*Q[k] for k in range(L)) + qco[l])
        m.addConstr( v[grid.loc[l]['busdown']] == v[grid.loc[l]['busup']] - 2*(grid.loc[l]['r[pu]']*P[l]+grid.loc[l]['x[pu]']*Q[l]) + vco[l])
        m.addConstr( (ia[l]**2+ir[l]**2) <= grid.loc[l]['ampacity[pu]']**2)
        m.addConstr(  ia[l] == P[l]/vap[grid.loc[l]['busup']])
        m.addConstr(  ir[l] == Q[l]/vap[grid.loc[l]['busup']])
    m.addConstr( v[0] == 1) 
    m.addConstr( P[0] == P0a-P0b)
    #m.addConstr( P0a+P0b >= Q[0]*math.tan(math.pi/2-math.acos(0.1)))
    #m.addConstr( P0a+P0b >= -Q[0]*math.tan(math.pi/2-math.acos(0.1)))
        

    # Solving model
    m.setParam('LogToConsole',0)
    m.optimize()

    # Check convergence
    if m.status == gp.GRB.OPTIMAL:
        pass
        #print("control: iDistFlow convergence")
    else:
        print(m.status)
        print("control: iDistFlow non-convergence") 
        m.computeIIS()  # Richiedi a Gurobi di calcolare l'IIS
        m.write("model.ilp")  # Scrive l'IIS in un file
        print("I vincoli seguenti sono in conflitto:")
        for c in m.getConstrs():
            if c.IISConstr:
                print(f"{c.constrName} è parte dell'IIS")
        
        
    # Returns solutions
    p_bess = {b: p_bess[b].X for b in range(B)}
    q_bess = {b: q_bess[b].X for b in range(B)}
    SoE = m.getAttr("x",SoE)
    P = {b: P[b].X for b in range(L)}
    DPo = DPo.X

    return(p_bess,q_bess,SoE,P)
    
def fake_DPr(grid,p_pv_c,q_pv_c,p_load_c,q_load_c,p_bess_set,q_bess_set):
    # sr scanario used as realisation
    DPr = solve_Load_flow(grid,p_pv_c,q_pv_c,p_load_c,q_load_c,p_bess_set,q_bess_set,fake_realisation=True)
    return(DPr)




#################################################################
if __name__=="__main__": 
    pass