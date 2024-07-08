"""
Load forecast scenarios generaation
"""

import numpy as np
from scipy.special import gamma
import pandas as pd

def load_weibul(shape,loc,mean,nc):
    """
    shape : float>0 shape paremeter of Weibul distribution
    loc : float>0 loc paremeter of Weibul distribution (minimum value)
    mean : float>0 mean of the distribution
    nc : int > 0 number of customers 

    Returns
    -------
    nc values randomly extracted from a Weibul distribution (shape,loc,mean)
    """
    scale = (mean - loc) / (gamma(1 + 1/shape))
    random_loads = np.random.weibull(shape, nc) * scale + loc
    return(random_loads)

def load_forecast(profile,nc,S,shape=1.4,loc=0.01):
    """
    profile : list of array of average values (es. 8760 value of average load over the year)
    nc : see load_weibul
    S : number of scenarios to generate
    spahe : see load_weibul (1.4 from Pecan database analysis 3 years 300 residential users)
    loc : see load_weibul (0.1 from Pecan database analysis 3 years 300 residential users)

    Returns
    -------
    Load array len(profile) x scenarios

    """
    T = len(profile)
    Load = np.ones((T,S))
    for t in range(T):
        for s in range(S):
            Load[t,s] = load_weibul(shape,loc,profile[t],nc).sum()
    Load = pd.DataFrame(Load)
    return(Load)


################################################################################
if __name__=="__main__": 

    import matplotlib.pyplot as plt
    from scipy.stats import weibull_min
    
    nc = 10  # number of customers
    S = 20 # number of scenarios
    profile = pd.read_csv('ARERA Firenze 2700kWh.csv')['W']/1000
    shape = 1.4 # shape parameter Weibul
    loc = 0.01 # loc parameter Weibul
    Load_test = load_forecast(profile,nc,S,shape=shape,loc=loc)
    
    
    # Weibul distribution simulation
    # shape, loc, scale = weibull_min.fit(data)
    
    scale = 0.5
    x = np.linspace(0, 4, 1000)
    pdf = weibull_min.pdf(x,shape,loc=loc,scale=scale)
    
    plt.figure(dpi=1000)
    plt.plot(x, pdf, label=f"shape={shape} scale={scale} loc={loc}")
    plt.xlabel('[kWh]')
    plt.ylabel('Probability density')
    plt.title('')
    plt.grid()
    plt.xlim(0,3)
    plt.legend()
    plt.show()
    
    plt.figure(dpi=1000)
    for mean in [0.2,0.4,0.6,0.8,1]:
        scale = (mean - loc) / (gamma(1 + 1/shape))
        x = np.linspace(0, 4, 1000)
        pdf = weibull_min.pdf(x,shape,loc=loc,scale=scale)
        plt.plot(x, pdf,label=f"mean={mean} kWh --> scale={round(scale,2)}")
    plt.xlabel('[kWh]')
    plt.ylabel('Probability density')
    plt.title('')
    plt.grid()
    plt.legend()
    plt.xlim(0,3)
    plt.show()
    
    
    
    
    
    
    # serie for the experiment
    nc = 12 # number of customers
    S = 40 # number of scenarios
    
    # profile to use as hourly mean
    profile = pd.read_csv('ARERA Firenze 2700kWh.csv')['W']/1000 # standard average load profile of one customer
    feb_28_indices = range(1416, 1440)  # Indices for 28th February (24 hours)
    feb_28_values = profile.iloc[feb_28_indices]
    profile = pd.concat([profile.iloc[:1440], feb_28_values, profile.iloc[1440:]]).reset_index(drop=True)
    profile.index = pd.date_range(start='2024-01-01 00:00:00', end='2025-01-01 00:00:00', freq='H') 
    profile = profile['2024-05-01 00:00:00':'2024-08-01 00:00:00']
    profile = profile[:-1]
    
    # generazione profili da weibulls
    Loads_h = [] # profili orari di ogni customer 50 scenari
    Loads_5m = [] # profili ogni 5 minuti di ogni customer scenario 1
    for c in range(nc):    
        Load_h = load_forecast(profile, 1, S) # generation of Load scenarios
        Load_h.index = profile.index
        Loads_h += [Load_h]
        
        profile_5m = Load_h[1].resample('300S').ffill()
        Load_5m = load_forecast(profile_5m, 1, 1)
        Loads_5m += [Load_5m]
        Load_5m.index = profile_5m.index
        
    
    # sommatoria di 20 customers, serie oraria 50 scenari 
    Load_tot_h = Loads_h[0]
    Load_tot_5m = Loads_5m[0]
    for c in range(1,nc):
        Load_tot_h += Loads_h[c]
        Load_tot_5m += Loads_5m[c]
        
    # rescale e rename
    Load_tot_h = Load_tot_h*1000
    Load_tot_5m = Load_tot_5m*1000
    Load_tot_5m.rename(columns={0: 'P [W]'}, inplace=True)
    
    # prepare the one for Franci
# =============================================================================
#     Load_franci = Load_tot_5m/1000
#     Load_franci.rename(columns={'P [W]': 'Active Power [kW]'}, inplace=True)
#     Load_franci['Reactive Power [kVAr]'] = Load_franci['Active Power [kW]']*0.484 # (cosphi=0.9)
# =============================================================================
    
    # round 
    Load_tot_h = Load_tot_h.round(0)
    Load_tot_5m = Load_tot_5m.round(0)
    #Load_franci = Load_franci.round(3)

    # save
    Load_tot_h.to_csv('12_residential_load_profiles_40s_hourly.csv')
    Load_tot_5m.to_csv('12_residential_load_profiles_s1_5minutes.csv')
    #Load_franci.to_csv('12_residential_load_profiles_s1_seconds_xFranci.csv')
    
    

    

        



        
    
        
    
    
    


    
    
    





