"""
Load flow for a low voltage radial grid
"""

import pandas as pd 
import numpy as np
from scipy.optimize import fsolve

def LoadFlow(x,grid,p_pv,q_pv,p_load,q_load,p_bess,q_bess):
    """"
    Load Flow equations
    for input see solve_Load_flow
    """

    # Variables
    L = len(grid)
    P = x[:L]
    Q = x[L:L*2]
    f = x[L*2:L*3]
    v = x[L*3:]
    
    # Adjacency matrix
    G = np.zeros((len(grid), len(grid)))
    for l, line in grid.iterrows():
        if l != 0:
            G[line['busup']-1,l] = 1   
       
    # Equations
    eqs = []
    for l, line in grid.iterrows():
        eqs.append( - P[l] + p_pv[l+1] + p_load[l+1] + p_bess[l+1] + line['r[pu]']*f[l] + np.dot(G[l],P) )
        eqs.append( - Q[l] + q_pv[l+1] + q_load[l+1] + q_bess[l+1] + line['x[pu]']*f[l] + np.dot(G[l],Q) )
        eqs.append( - f[l] + (np.abs(P[l]+1j*Q[l])**2)/v[line['busup']] )
        eqs.append( - v[line['busdown']] + v[line['busup']] -  2*(line['r[pu]']*P[l]+line['x[pu]']*Q[l]) +  (np.abs(line['r[pu]']+1j*line['x[pu]'])**2)*f[l] )
    eqs.append(v[0]-1) # slack bus voltage
    
    return(eqs)

def solve_Load_flow(grid,p_pv,q_pv,p_load,q_load,p_bess,q_bess,coDistFlow=False,fake_realisation=False):
    """
    grid : grid topology database see linedata_AC_test as example: must be radial!
    p_pv : vector len = number of busses in the gird. Active power injected at the busses.
    q_pv : vector len = number of busses in the gird. Reactive power injected at the busses.
    p_load : vector len = number of busses in the gird. Active power injected at the busses.
    q_load : vector len = number of busses in the gird. Reactive power injected at the busses.
    p_bess : vector len = number of busses in the gird. Active power injected at the busses.
    q_bess : vector len = number of busses in the gird. Reactive power injected at the busses.
    
    coDistFlow : TYPE, optional. The default is False: to decide what to return

    Returns
    -------
    lines and busses status or coDistFlow correction and approximations coefficients

    """
    
    # assert input dimension
    L = len(grid) # lines
    B = L+1 # busses
    assert len(p_pv) == len(q_pv) == len(p_bess) == len(q_bess) == len(p_load) == len(q_load) == B, "check input lenght: pv,load,bess must have the same dimension i.e. the number of busses in the grid = number of lines +1"
    
    # starting points
    v0 = np.ones(B)
    f0 = np.ones(L)
    P0 = np.zeros(L)
    Q0 = np.zeros(L)
    x0 = np.concatenate((P0,Q0,f0,v0))
    
    # solve
    root = fsolve(LoadFlow, x0, args=(grid,p_pv,q_pv,p_load,q_load,p_bess,q_bess))
    
    # solutions
    P = root[:L]
    Q = root[L:L*2]
    f = root[L*2:L*3]
    v = root[L*3:]
    
    if coDistFlow: # to return the corrections and approximations needed for CoDistFlow
        pco = pd.DataFrame({'Pco':grid['r[pu]']*f})
        qco = pd.DataFrame({'Qco':grid['x[pu]']*f})
        vco = pd.DataFrame({'vco':(np.abs(grid['r[pu]']+1j*grid['x[pu]'])**2)*f})       
        vap = pd.DataFrame({'vap':v**(1/2)})
        pco = pco['Pco'].to_numpy()
        qco = qco['Qco'].to_numpy()
        vco = vco['vco'].to_numpy()
        vap = vap['vap'].to_numpy()
        
        return(pco,qco,vco,vap)
    
    elif fake_realisation:
        return(P[0])
    
    else: # to return the lines and busses status
        lines_status = pd.DataFrame({'Active power':P, 'Reactive power':Q, 'Current':f**(1/2)})
        lines_status.index.name = 'line'
        busses_status = pd.DataFrame({'Voltage':v**(1/2)})
        busses_status.index.name = 'bus'
        return(lines_status,busses_status)


#%%###################### TEST AND VALIDATION ########################################################

if __name__=="__main__": 
    
    from PerUnit import to_pu, from_pu
    
### Load Flow test ###################################################################################
    
    # Grid (must be radial)
    data = pd.read_csv('linedata_AC_test.txt',delim_whitespace=True, skipinitialspace=True)
    grid = pd.DataFrame(data)
    grid = grid.rename_axis("line")
    L = len(grid) # number of lines 
    B = L+1  # number of bus = number of line+1
    
    bess_bus = 4
    
    # initialize input vectors and starting point
    p_pv = np.zeros(B)
    q_pv = np.zeros(B)
    p_load = np.zeros(B)
    q_load = np.zeros(B)
    p_bess = np.zeros(B)
    q_bess = np.zeros(B)
    
    # define pv, bess and load injection on the busses [W]
    p_load[2] = 6000 
    q_load[2] = 1000
    p_bess[bess_bus] = -500 
    p_bess[bess_bus] = -100
    p_pv[6] = -1000 
    p_pv[6] = -100 
    p_pv[8] = -2000 
    q_pv[8] = -500
    
    # need pu input
    Ab = 20000 # base value for the powers [W]
    Eb = 400.0 # base value for the voltages and slack bus voltage [V]
    grid,p_pv,q_pv,p_load,q_load,p_bess,q_bess = to_pu(grid,p_pv,q_pv,p_load,q_load,p_bess,q_bess,Ab,Eb)
    
    print('\n\n my model results:')
    lines,busses = solve_Load_flow(grid,p_pv,q_pv,p_load,q_load,p_bess,q_bess)
    print('\n')
    print(lines)
    print(busses)
    
    #P0 = solve_Load_flow(grid,p_pv,q_pv,p_load,q_load,p_bess,q_bess,fake_realisation=True)
    
    # output for CoDistFlow
# =============================================================================
#     pco,qco,vco,vap = solve_Load_flow(grid,p_pv,q_pv,p_load,q_load,p_bess,q_bess,coDistFlow=True)
#     print(pco)
#     print(qco)
#     print(vco)
#     print(vap)
# =============================================================================
    

### Validation vs pypsa ##############################################################################

    import pypsa
    network = pypsa.Network()
    
    # input definition (need not pu)
    grid,p_pv,q_pv,p_load,q_load,p_bess,q_bess = from_pu(grid,p_pv,q_pv,p_load,q_load,p_bess,q_bess,Ab,Eb)
    for b in range(B):
        network.add("Bus", f"bus {b}", v_nom=Eb)
    for l, line in grid.iterrows():
        network.add("Line", f"line {l}", bus0=f"bus {line['busup']}", bus1=f"bus {line['busdown']}", x=line['x[ohm]'], r=line['r[ohm]'])
    network.add("Generator", "gen slack", bus="bus 0", p_set=Ab, control="PQ")
    network.add("Load", "load 2", bus="bus 2", p_set=p_load[2], q_set=q_load[2])
    network.add("Load", "bess 4", bus="bus 4", p_set=p_bess[4], q_set=q_bess[4])
    network.add("Load", "pv 6", bus="bus 6", p_set=p_pv[6], q_set=q_pv[6])
    network.add("Load", "pv 8", bus="bus 8", p_set=p_pv[8], q_set=q_pv[8])
    
    # solve
    network.pf()
    
    # solutions    
    P = network.lines_t.p0
    Q = network.lines_t.q0
    v = network.buses_t.v_mag_pu
    i = []
    for l in range(L):
        i.append((abs(P[f"line {l}"]+1j*Q[f"line {l}"])/Ab)/(v[f"bus {l}"]))
    print('\n\n pypsa model results:')
    print('\n Active power')
    print(P/Ab)
    print('\n Reactive power')
    print(Q/Ab)
    print('\n Voltage')
    print(v)
    
### validation vs EPFL-DESL model ###################################################################

    def loadflow(Y, S_star, E_star, E_0, idx, Parameters):
        # ! Validated through comparison with the matlab code of the course
        n_nodes = len(E_0)
        G = np.real(Y)
        B = np.imag(Y)
        # Initialization
        Ere = np.real(E_0)
        Eim = np.imag(E_0)
        J = None
        for n_iter in range(1,Parameters['n_max']):
            # Compute nodal voltages/currents/powers
            E = Ere + 1j * Eim
            I = Y @ E
            S = E * np.conj(I)
            ## Mismatch calculation
            # Compute the mismatches for the entire network
            dS = S_star - S
            dP = np.real(dS)
            dQ = np.imag(dS)
            dV2 = E_star**2 - np.abs(E)**2
            # Keep only the relevant mismatches
            dP = np.delete(dP, idx['slack'])
            dQ = np.delete(dQ, np.concatenate((idx['pv'], idx['slack'])).astype(int))
            dV2 = np.delete(dV2, np.concatenate((idx['pq'], idx['slack'])).astype(int))
            dF = np.concatenate((dP, dQ, dV2)) # mismatch of the power flow equations
            ## Convergence check
            if np.max(np.abs(dF)) < Parameters['tol']:
                #print('NR algorithm has converged to a solution!')
                break
            elif n_iter == Parameters['n_max']-1:
                print('NR algorithm reached the maximum number of iterations!')
            ## Jacobian construction
            # For the sake of simplicity, the blocks of J are constructed
            # for the whole network (i.e., with size n_nodes x n_nodes).
            # The unnecessary rows/columns are removed subsequently
            # Extract the real and imaginary parts of the voltages
            Ere = np.real(E_0)
            Eim = np.imag(E_0)
            # Initialization
            J_PR = np.zeros((n_nodes, n_nodes)) # derivative: P versus E_re
            J_PX = np.zeros((n_nodes, n_nodes)) # derivative: P versus E_im
            J_QR = np.zeros((n_nodes, n_nodes)) # derivative: Q versus E_re
            J_QX = np.zeros((n_nodes, n_nodes)) # derivative: Q versus E_im
            J_ER = np.zeros((n_nodes, n_nodes)) # derivative: E^2 versus E_re
            J_EX = np.zeros((n_nodes, n_nodes)) # derivative: E^2 versus E_im
            # Construction
            for i in range(n_nodes):
                # Diagonal elements (terms outside the sum)
                J_PR[i, i] = 2 * G[i, i] * Ere[i]
                J_PX[i, i] = 2 * G[i, i] * Eim[i]
                J_QR[i, i] = -2 * B[i, i] * Ere[i]
                J_QX[i, i] = -2 * B[i, i] * Eim[i]
                J_ER[i, i] = 2 * Ere[i]
                J_EX[i, i] = 2 * Eim[i]
                for j in range(n_nodes):
                    if j != i:
                        # Diagonal elements (terms inside the sum)
                        J_PR[i, i] += G[i, j] * Ere[j] - B[i, j] * Eim[j]
                        J_PX[i, i] += B[i, j] * Ere[j] + G[i, j] * Eim[j]
                        J_QR[i, i] -= B[i, j] * Ere[j] + G[i, j] * Eim[j]
                        J_QX[i, i] += G[i, j] * Ere[j] - B[i, j] * Eim[j]
                        # Off-diagonal elements
                        J_PR[i, j] = G[i, j] * Ere[i] + B[i, j] * Eim[i]
                        J_PX[i, j] = -B[i, j] * Ere[i] + G[i, j] * Eim[i]
                        J_QR[i, j] = -B[i, j] * Ere[i] + G[i, j] * Eim[i]
                        J_QX[i, j] = -G[i, j] * Ere[i] - B[i, j] * Eim[i]
            # Remove extra rows (i.e., unnecessary equations)
            # slack bus: P & Q & E^2, PV buses: Q, PQ buses: E^2
            J_PR = np.delete(J_PR, idx['slack'], axis=0)
            J_PX = np.delete(J_PX, idx['slack'], axis=0)
            J_QR = np.delete(J_QR, np.concatenate((idx['pv'], idx['slack'])).astype(int), axis=0)
            J_QX = np.delete(J_QX, np.concatenate((idx['pv'], idx['slack'])).astype(int), axis=0)
            J_ER = np.delete(J_ER, np.concatenate((idx['pq'], idx['slack'])).astype(int), axis=0)
            J_EX = np.delete(J_EX, np.concatenate((idx['pq'], idx['slack'])).astype(int), axis=0)
            # Remove extra columns (i.e., variables)
            # slack bus: E_re & E_im
            J_PR = np.delete(J_PR, idx['slack'], axis=1)
            J_QR = np.delete(J_QR, idx['slack'], axis=1)
            J_ER = np.delete(J_ER, idx['slack'], axis=1)
            J_PX = np.delete(J_PX, idx['slack'], axis=1)
            J_QX = np.delete(J_QX, idx['slack'], axis=1)
            J_EX = np.delete(J_EX, idx['slack'], axis=1)
            # Combination
            J = np.concatenate((np.concatenate((J_PR, J_PX), axis=1), np.concatenate((J_QR, J_QX), axis=1),
                                np.concatenate((J_ER, J_EX), axis=1)), axis=0)
            ## Solution update
            # Solve
            dx = np.real(np.linalg.solve(J, dF)) # Take real part to avoid warning casting complex to double (imaginary part is zero)
            # Reconstruct the solution
            dE_re = np.zeros(len(Ere))
            dE_re[np.concatenate((idx['pq'], idx['pv'])).astype(int)] = dx[:len(idx['pq']) + len(idx['pv'])]
            dE_im = np.zeros(len(Eim))
            dE_im[np.concatenate((idx['pq'], idx['pv'])).astype(int)] = dx[len(idx['pq']) + len(idx['pv']):]
            # Update
            Ere += dE_re
            Eim += dE_im
       
        E = Ere + 1j * Eim
        return S, E, J, n_iter
    

    # input definition (need pu)
    grid,p_pv,q_pv,p_load,q_load,p_bess,q_bess = to_pu(grid,p_pv,q_pv,p_load,q_load,p_bess,q_bess,Ab,Eb)
    Ptest = np.zeros(B)
    Qtest = np.zeros(B)
    Ptest[2] = -p_load[2] 
    Qtest[2] = -q_load[2] 
    Ptest[4] = -p_bess[4] 
    Qtest[4] = -q_bess[4] 
    Ptest[6] = -p_pv[6] 
    Qtest[6] = -q_pv[6] 
    Ptest[8] = -p_pv[8]
    Qtest[8] = -q_pv[8]
    S_star = Ptest + 1j*Qtest
    E_star = np.ones(B, dtype=complex)
    E_0 = np.ones(B, dtype=complex)
    idx = dict(); idx['slack'] = [0]; idx['pq'] = np.arange(1, B); idx['pv'] = [] # Define the indices of the different types of nodes
    Parameters = dict(); Parameters['n_max'] = 100; Parameters['tol'] = 1e-8;  # Parameters for the Newton-Raphson method
    
    # admittance matric Y
    Y = np.zeros((B, B), dtype=complex)
    for l, line in grid.iterrows():
        Y[line['busup'],line['busdown']] = - 1 / (line['r[pu]']+1j*line['x[pu]'])
        Y[line['busdown'],line['busup']] = - 1 / (line['r[pu]']+1j*line['x[pu]'])
        Y[line['busup'],line['busup']] += 1 / (line['r[pu]']+1j*line['x[pu]'])
        Y[line['busdown'],line['busdown']] += 1 / (line['r[pu]']+1j*line['x[pu]'])

    S,E,J,n=loadflow(Y, S_star, E_star, E_0, idx, Parameters)
    print('\n\n EPFL-DESL model results:')
    print('\n Active power')
    print(S.real)
    print('\n Reactive power')
    print(S.imag)
    print('\n Voltage')
    print(E.real)
    
    