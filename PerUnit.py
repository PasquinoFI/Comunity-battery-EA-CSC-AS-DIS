"""
from-to per unit conversion functions
"""

import pandas as pd
import numpy as np

def to_pu(grid,p_pv,q_pv,p_load,q_load,p_bess,q_bess,Ab,Eb):
    """
    from V, A and Ohm to pu
    Ab [kVa] base for the voltage
    Eb [V] base for the powers
    """
    
    Ib = Ab/Eb
    Zb = Eb/Ib
    
    if isinstance(p_pv, pd.DataFrame):
        p_pv = p_pv.applymap(lambda cell: np.array(cell) / Ab)
        q_pv = q_pv.applymap(lambda cell: np.array(cell) / Ab)
        p_load = p_load.applymap(lambda cell: np.array(cell) / Ab)
        q_load = q_load.applymap(lambda cell: np.array(cell) / Ab)
    else:
        p_pv = p_pv/Ab
        q_pv = q_pv/Ab
        p_load = p_load/Ab
        q_load = q_load/Ab
        
    p_bess = p_bess/Ab
    q_bess = q_bess/Ab
        
    grid['r[ohm]'] = grid['r[ohm]']/Zb
    grid['x[ohm]'] = grid['x[ohm]']/Zb
    grid['ampacity[A]'] = grid['ampacity[A]']/Ib
    grid = grid.rename(columns={'r[ohm]': 'r[pu]', 'x[ohm]': 'x[pu]', 'ampacity[A]': 'ampacity[pu]' })

    return(grid,p_pv,q_pv,p_load,q_load,p_bess,q_bess)

def from_pu(grid,p_pv,q_pv,p_load,q_load,p_bess,q_bess,Ab,Eb):
    """
    from pu to V, A and Ohm
    Ab [kVa] base for the voltage
    Eb [V] base for the powers
    """
    
    Ib = Ab/Eb
    Zb = Eb/Ib
    
    p_pv = p_pv*Ab
    q_pv = q_pv*Ab
    p_load = p_load*Ab
    q_load = q_load*Ab
    p_bess = p_bess*Ab
    q_bess = q_bess*Ab
    grid['r[pu]'] = grid['r[pu]']*Zb
    grid['x[pu]'] = grid['x[pu]']*Zb
    grid['ampacity[pu]'] = grid['ampacity[pu]']*Ib
    
    grid = grid.rename(columns={'r[pu]': 'r[ohm]', 'x[pu]': 'x[ohm]', 'ampacity[pu]': 'ampacity[A]' })
    
    return(grid,p_pv,q_pv,p_load,q_load,p_bess,q_bess)

def to_pu_grid(grid,Ab,Eb):
    
    Ib = Ab/Eb
    Zb = Eb/Ib    
    grid['r[ohm]'] = grid['r[ohm]']/Zb
    grid['x[ohm]'] = grid['x[ohm]']/Zb
    grid['ampacity[A]'] = grid['ampacity[A]']/Ib
    grid = grid.rename(columns={'r[ohm]': 'r[pu]', 'x[ohm]': 'x[pu]', 'ampacity[A]': 'ampacity[pu]' })
    
    return(grid)
    
def to_pu_pq(p_pv,q_pv,p_load,q_load,Ab,Eb):
    
    if isinstance(p_pv, pd.DataFrame):
        p_pv = p_pv.applymap(lambda cell: np.array(cell) / Ab)
        q_pv = q_pv.applymap(lambda cell: np.array(cell) / Ab)
        p_load = p_load.applymap(lambda cell: np.array(cell) / Ab)
        q_load = q_load.applymap(lambda cell: np.array(cell) / Ab)
    else:
        p_pv = p_pv/Ab
        q_pv = q_pv/Ab
        p_load = p_load/Ab
        q_load = q_load/Ab
        
    return(p_pv,q_pv,p_load,q_load)

def to_pu_pq2(p_load,q_load,Ab,Eb):
    
    p_load = p_load.applymap(lambda cell: np.array(cell) / Ab)
    q_load = q_load.applymap(lambda cell: np.array(cell) / Ab)

    return(p_load,q_load)

