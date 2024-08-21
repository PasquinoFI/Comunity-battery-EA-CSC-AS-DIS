"""
controller plots
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

def scheduling_plot(title,S,T,SoE,SoE_max,P,DP,PVs,Load,bess_bus,Ab,EP,fromdatasaved=False):
    
    check = 0
    for pv in PVs:
        if check == 0:
            prod = PVs[pv]['pro'].copy()
            check = 1
        else:
            prod+= PVs[pv]['pro'].copy()
            
    y1 = []
    P0 = []
    for s in range(S):
        y1.append([])
        P0.append([])
        for t in range(T):
            if fromdatasaved:
                y1[s].append(SoE.loc[s,t,bess_bus]*Ab)
                P0[s].append(P.loc[s,t,0]*Ab)
            else:
               y1[s].append(SoE[s,t,bess_bus]*Ab)
               P0[s].append(P[s,t,0]*Ab)
      
    
    hours = Load.index.hour
    
    fig, ((ax1,ax2,ax3,ax4,ax5,ax6)) = plt.subplots(6,1,figsize=(9,12),dpi=1000)
    x = range(T)
    for y in y1:
        ax4.plot(x,y)
    ax4.grid()
    ax4.set_ylabel('SoE [Wh]')
    ax4.set_xticks(x[::3], labels=hours[::3])
    ax4.axhline(y=SoE_max * Ab * 0.1, color='black', linestyle='--')
    ax4.axhline(y=SoE_max * Ab * 0.9, color='black', linestyle='--')
    ax4.set_ylim(0,SoE_max*Ab)
    ax4.set_xlim(x[0],x[-1])
    
    for y in prod:
        ax1.plot(x,prod[y])
    ax1.grid()
    ax1.set_ylabel('PV [Wh]')
    ax1.set_xticks(x[::3], labels=hours[::3])
    ax1.set_ylim(-18000,0)
    ax1.set_xlim(x[0],x[-1])
    
    for s in range(S):
        ax2.plot(x,Load[str(s)]*Ab)
    ax2.grid()
    ax2.set_ylabel('Load [Wh]')
    ax2.set_xticks(x[::3], labels=hours[::3])
    ax2.set_xlim(x[0],x[-1])
    
    for y in P0:
        ax5.plot(x,y)
    ax5.grid()
    ax5.set_ylabel('GCP [Wh]')
    ax5.set_xticks(x[::3], labels=hours[::3])
    ax5.set_xlim(x[0],x[-1])
    ax5.axhline(y=0, color='black', linestyle='--')

    
    ax6.plot(x,DP*Ab)
    ax6.set_ylabel('DP [Wh]')
    ax6.grid()
    ax6.set_xticks(x[::3], labels=hours[::3])
    ax6.set_xlim(x[0],x[-1])
    ax6.axhline(y=0, color='black', linestyle='--')
    
    ax3.plot(x,EP*1000)
    ax3.grid()
    ax3.set_ylabel('EP [€/kWh]')
    ax3.set_xticks(x[::3], labels=hours[::3])
    ax3.set_xlim(x[0],x[-1])
    
    #plt.suptitle(title)
    #plt.suptitle("cloudy day 02/06/24")
    #plt.suptitle("sunny day 18/06/24")
    plt.suptitle("variable weather day 17/06/24")
    #plt.suptitle("$act^{cost}$ = 0.13 €/kWh")
    #plt.suptitle("$act^{cost}$ = 0.10 €/kWh")
    #plt.suptitle("$dis^{cost}$ = 0.45 €/kWh")
    #plt.suptitle("26/06/24 07:57 scheduling")
    plt.tight_layout()
    plt.show()
    return()


def control_plot(DP_bidded,DP_real,Ab,step,p_bess_set,SoEs,SoE_max,sim=False):
    DP_bidded = DP_bidded['pu']*Ab
    DP_real = DP_real['pu']*Ab
    
    P_bess_sets = p_bess_set['pu']*Ab
    
    if sim == 2:
        SoEs = (SoEs['pu']*Ab).sort_index()[1:-1]
    elif sim:
        SoEs = (SoEs['pu']*Ab).sort_index()[1:]
    else:
        SoEs = (SoEs['pu']*Ab).sort_index()[1:-2]
    
    # dispatching # rimoltiplica tutto x AB!!!!
    xlim = [0,len(DP_bidded)]
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,figsize=(9,8),dpi=1000)
    x = np.arange(len(DP_bidded))
    ax1.plot(x,DP_bidded,label='DP bidded')
    ax1.plot(x,DP_real,label='DP realised')
    ax1.legend()
    ax1.set_ylabel("GCP [W]")
    ax1.grid()
    ax1.set_xlim(xlim[0],xlim[1])
    dpe = abs((np.array(DP_bidded)-np.repeat(np.mean(np.array(DP_real).reshape(-1,int(3600/step)),axis=1),3600/step)))
    ax2.plot(x,dpe)
    ax2.grid()
    ax2.set_ylabel("Hourly DP error [Wh]")
    ax2.set_xlim(xlim[0],xlim[1])
    ax3.plot(x,P_bess_sets)
    ax3.grid()
    ax3.set_ylabel("P bess [W]")
    ax3.set_xlim(xlim[0],xlim[1])
    ax4.plot(x,SoEs)
    ax4.grid()
    ax4.set_ylabel("SoE [Wh]")
    ax4.set_xlim(xlim[0],xlim[1])
    ax4.axhline(y=SoE_max * Ab * 0.1, color='black', linestyle='--')
    ax4.axhline(y=SoE_max * Ab * 0.9, color='black', linestyle='--')
    ax4.set_ylim(0,SoE_max*Ab)
    plt.suptitle(f"Control every {step} seconds")
    
    plt.tight_layout()
    plt.show()
    
def control_plot_final(DP_bidded, DP_real, Ab, step, p_bess_set, SoEs, SoE_max, pv, load, sim=False):
    DP_bidded = DP_bidded['pu'] * Ab
    DP_real = DP_real['pu'] * Ab
    
    P_bess_sets = p_bess_set['pu'] * Ab
    
    if sim == 2:
        SoEs = (SoEs['pu'] * Ab).sort_index()[1:-1]
    elif sim:
        SoEs = (SoEs['pu'] * Ab).sort_index()[1:]
    else:
        SoEs = (SoEs['pu'] * Ab).sort_index()[1:-2]
    
    # Create a datetime index starting at 08:00
    start_time = pd.Timestamp('2024-06-26 08:00')
    time_index = [start_time + pd.Timedelta(seconds=30 * i) for i in range(len(DP_bidded))]

    # dispatching # rimoltiplica tutto x AB!!!!
    fig, (ax5, ax6, ax3, ax4, ax1, ax2) = plt.subplots(6, 1, figsize=(9, 12), dpi=1000)
    ax1.plot(time_index, DP_bidded, label='DP bidded')
    ax1.plot(time_index, DP_real, label='DP realised')
    ax1.legend()
    ax1.set_ylabel("GCP [W]")
    ax1.grid()
    ax1.set_xlim(time_index[0], time_index[-1])
    dpe = abs((np.array(DP_bidded) - np.repeat(np.mean(np.array(DP_real).reshape(-1, int(3600 / step)), axis=1), 3600 / step)))
    ax2.plot(time_index, dpe)
    ax2.grid()
    ax2.set_ylabel("Hourly DP error [Wh]")
    ax2.set_xlim(time_index[0], time_index[-1])
    ax3.plot(time_index, P_bess_sets)
    ax3.grid()
    ax3.set_ylabel("P bess [W]")
    ax3.set_xlim(time_index[0], time_index[-1])
    ax4.plot(time_index, SoEs)
    ax4.grid()
    ax4.set_ylabel("SoE [Wh]")
    ax4.set_xlim(time_index[0], time_index[-1])
    ax4.axhline(y=SoE_max * Ab * 0.1, color='black', linestyle='--')
    ax4.axhline(y=SoE_max * Ab * 0.9, color='black', linestyle='--')
    ax4.set_ylim(0, SoE_max * Ab)
    
    # Plot pv data
    ax5.plot(pv.index, pv['power [W]'], label='PV power', color='orange')
    ax5.legend()
    ax5.set_ylabel("PV Power [W]")
    ax5.set_xlim(pv.index[0], pv.index[-1])
    ax5.grid()
    
    # Plot load data
    ax6.plot(load.index, load['P [W]'], label='Load', color='purple')
    ax6.legend()
    ax6.set_xlim(load.index[0], load.index[-1])
    ax6.set_ylabel("Load [W]")
    ax6.grid()
    
    # Set date formatters for x-axis to show only the time and labels every 3 hours starting from 09:00
    time_fmt = mdates.DateFormatter('%H')
    hour_locator = mdates.HourLocator(byhour=[8, 11, 14, 17, 20, 23, 2, 5])
    minor_hour_locator = mdates.HourLocator(interval=1)
    
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.xaxis.set_major_formatter(time_fmt)
        ax.xaxis.set_major_locator(hour_locator)
        ax.xaxis.set_minor_locator(minor_hour_locator)
        ax.tick_params(axis='x', which='major')
        ax.tick_params(axis='x', which='minor')

    plt.suptitle(f"Control every {step} seconds")
    plt.tight_layout()
    plt.show()