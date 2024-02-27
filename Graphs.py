"""
Graphs
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Graphs DAM vs Unbalances
def DAMvsRegulation(P,PR,Reg,start,end,step):

    y1 = P[start:end]
    y2 = PR[start:end]
    y3 = Reg[start:end]
    
    x=np.arange(len(y1))
    fig, (ax1,ax2) = plt.subplots(2,1,dpi=1000)
    ax1.plot(x,y1,label='P')
    ax1.plot(x,y2,label='PR')
    ax1.legend()
    ax1.set_ylabel('Price [€/MWh]')
    ax2.plot(x,y3,label='Regulation = - Unbalance',color='green')
    ax2.plot(x,np.zeros(len(y3)),color='k')
    ax2.legend()
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Energy [MWh]')
    ax1.set_xlim(0,len(y1))
    ax2.set_xlim(0,len(y1))
    ax1.set_xticks(range(x[0], x[-1], step))
    ax2.set_xticks(range(x[0], x[-1], step))
    ax1.grid()
    ax2.grid()
    ax1.set_xticklabels([])
    plt.suptitle(f"{start[:10]} to {end[:10]}")
    plt.tight_layout()
    plt.show()
    
# Graphs P vs PR vs Reg
def PvsPRvsReg(P,PR,Reg,start,end,step):

    y1 = P[start:end]
    y2 = PR[start:end]
    y3 = Reg[start:end]
    
    x=np.arange(len(y1))
    fig, (ax1,ax2,ax3) = plt.subplots(3,1,dpi=1000)
    ax1.plot(x,y1,label='P')
    ax2.plot(x,y2,label='PR',color='tab:orange')
    ax3.plot(x,y3,label='Regulation = - Unbalance',color='green')
    ax3.plot(x,np.zeros(len(y3)),color='k')    
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax1.set_ylabel('Price [€/MWh]')
    ax2.set_ylabel('Price [€/MWh]')
    ax3.set_ylabel('Energy [MWh]')
    ax3.set_xlabel('Hour')
    ax1.set_xlim(0,len(y1))
    ax2.set_xlim(0,len(y1))
    ax3.set_xlim(0,len(y1))
    
    ax1.set_xticks(range(x[0], x[-1], step))
    ax2.set_xticks(range(x[0], x[-1], step))
    ax3.set_xticks(range(x[0], x[-1], step))
    
    months = [0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016, 8760]
    months_label = ['        J', '        F', '        M', '        A', '        M', '        J', '        J', '        A', '        S', '        O', '        N', '        D', '        J']
    ax1.set_xticks(months,months_label)
    ax2.set_xticks(months,months_label)
    ax3.set_xticks(months,months_label)
    
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    plt.suptitle(f"{start[:10]} to {end[:10]}")
    plt.tight_layout()
    plt.show()
    
# Graphs PR vs PR* vs Unbalances
def PRvsPRfitted(PR,PR_fitted,Unb,start,end):
    y1 = PR_fitted
    y2 = PR[start:end]
    y3 = Unb[start:end]
    
    x=np.arange(len(y1))
    fig, (ax1,ax2) = plt.subplots(2,1,dpi=1000)
    ax1.plot(x,y1,label='PR fitted')
    ax1.plot(x,y2,label='PR')
    ax1.legend()
    ax1.set_ylabel('Price [€/MWh]')
    ax2.plot(x,-y3,label='Regulation = - Unbalance',color='green')
    ax2.plot(x,np.zeros(len(y3)),color='k')
    ax2.legend()
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Energy [MWh]')
    ax1.set_xlim(0,len(y1))
    ax2.set_xlim(0,len(y1))
    #ax1.set_ylim(0,200)
    #ax2.set_ylim(1000,200)
    ax1.set_xticks([i * 24  for i in range(len(x)//24)])
    ax2.set_xticks([i * 24  for i in range(len(x)//24)])
    ax1.grid()
    ax2.grid()
    ax1.set_xticklabels([])
    plt.suptitle(f"{str(start)[:10]} to {str(end)[:10]}")
    plt.tight_layout()
    plt.show()
    
    
# Regulating power price
def regulating_power_prirce(Reg_range,PR_fitted,P0):

    fig, ax1 = plt.subplots(dpi=1000)
    ax1.plot(Reg_range,PR_fitted,label='Regulation price')
    ax1.plot(Reg_range,np.ones(len(Reg_range))*P0,label='Spot price')
    ax1.legend()
    ax1.set_ylabel('Price [€/MWh]')
    ax1.set_xlabel('Amount of regulation [MWh]')
    ax1.grid()
    ax1.set_xlim(Reg_range[0],Reg_range[-1])
    plt.show()
    
# Graphs premium readiness
def premium_readiness(P_range,Premium_down,Premium_up):

    fig, ax1 = plt.subplots(dpi=1000)
    ax1.plot(P_range,Premium_down,label='Premium_down')
    ax1.plot(P_range,Premium_up,label='Premium_up')
    ax1.legend(loc=3)
    ax1.set_ylabel('Premium [€/MWh]')
    ax1.set_xlabel('Spot price [MWh]')
    ax1.grid()
    ax1.set_xlim(P_range[0],P_range[-1])
    plt.show()
    
# R2vsDays
def R2vsDays(R22,days,title):

    fig, ax1 = plt.subplots(dpi=1000)
    
    for la in R22:
        ax1.plot(days,R22[la],label=la)
    
    plt.legend(title='Looking ahead [days]')
    ax1.set_ylabel('R\u00B2 [-]')
    ax1.set_xlabel('Looking back [days]')
    ax1.grid()
    ax1.set_xlim(days[0],days[-1])
    plt.suptitle(title)
    plt.show()

def pdf(serie,xlabel,title):
    
    # Calcola la deviazione standard dei residui
    serie_std = np.std(serie)
    
    # Plot PDF of serie
    plt.figure(dpi=1000)
    plt.hist(serie, bins=30, density=True, alpha=0.6)
    
    # Calcola la distribuzione normale con media zero e deviazione standard dei residui
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p0 = norm.pdf(x, 0, serie_std)
    p = norm.pdf(x, np.mean(serie), serie_std)
    
    # Aggiungi la curva della distribuzione normale al plot
    plt.plot(x, p, label='Normal')
    plt.plot(x, p0, label='Normal 0')
    plt.legend()

    plt.xlabel(xlabel)
    plt.ylabel('Probability Density')
    plt.title(title)
    plt.grid(True)
    plt.show()
    
def duble_boxplot(serie1,serie2,title1,title2,whis=1.5):
    
    fig, (ax1,ax2) = plt.subplots(1, 2, dpi=1000)
    
    ax1.boxplot(serie1,whis=whis)
    ax2.boxplot(serie2,whis=whis)
    ax1.set_title(title1)
    ax2.set_title(title2)
    ax1.grid(True)
    ax2.grid(True)
    ax2.set_ylim(-200,500)
    ax1.set_ylim(-200,500)
    plt.suptitle(whis)
    plt.tight_layout()
    plt.show()
    
    
def rolling_reg_tot(serie,window_size,title):
    
    mean = serie.rolling(window=window_size).mean()
    std = serie.rolling(window=window_size).std()
    x=np.arange(len(serie))    
    fig, ax1 = plt.subplots(dpi=1000)
    ax1.plot(x,serie,label='values')
    ax1.plot(x,mean,label='rolling mean')
    ax1.plot(x,std,label='rolling std')
    plt.legend()
    plt.grid()
    plt.xlim(x[0],x[-1])
    
    months = [0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016, 8760]
    months_label = ['        J', '        F', '        M', '        A', '        M', '        J', '        J', '        A', '        S', '        O', '        N', '        D', '        J']
    ax1.set_xticks(months,months_label)
    plt.title(title)
    plt.show()
    
def rolling_reg_m(serie,window_size,step,title):

    mean = serie.rolling(window=window_size).mean()
    std = serie.rolling(window=window_size).std()
    x=np.arange(len(serie))    
    fig, ax1 = plt.subplots(dpi=1000)
    ax1.plot(x,serie,label='values')
    ax1.plot(x,mean,label='rolling mean')
    ax1.plot(x,std,label='rolling std')
    plt.legend()
    plt.grid()
    plt.xlim(x[0],x[-1])
    
    ax1.set_xticks(range(x[0], x[-1], step))
    plt.title(title)
    plt.show()
    
def acf_pacf(serie,nlag,step,title):
    
    fig, ax = plt.subplots(2, 1, dpi=1000)
    plot_acf(serie, lags=nlag, ax=ax[0])
    ax[0].set_title('ACF')
    plot_pacf(serie, lags=nlag, ax=ax[1])
    ax[1].set_title('PACF')
    plt.tight_layout()
    ax[0].grid()  
    ax[1].grid() 
    ax[0].set_xticks(range(0, nlag+1, step))
    ax[1].set_xticks(range(0, nlag+1, step))
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def ARIMAvsReal(serie_test,forecast,step,title):
    
    x=np.arange(len(serie_test))    
    fig, ax1 = plt.subplots(dpi=1000)
    ax1.plot(x,serie_test,label='Real')
    ax1.plot(x,forecast,label='Forecasts')
    plt.legend()
    plt.grid()
    plt.xlim(x[0],x[-1])
    
    ax1.set_xticks(range(x[0], x[-1], step))
    plt.title(title)
    plt.show()
    
    
    
    
    
    