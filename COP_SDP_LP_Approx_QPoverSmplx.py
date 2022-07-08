# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 06:00:37 2022

@author: Akshit
"""
import numpy as np
import cvxpy as cp
import time

def SDP_LP_Approx(Q):
    return_dict = {}
    n = np.shape(Q)[0]    
    E = np.ones((n,n))
    
    #% ================= LP Relaxation from Non-Negativity ===================
    lmbd_IA = cp.Variable()
    X_IA = cp.Variable((n,n), symmetric=True)
    N = cp.Variable((n,n), symmetric=True)
    
    obj_IA = lmbd_IA
    
    constraints_IA = [Q - lmbd_IA*E == X_IA]
    constraints_IA.append(X_IA == N)
    constraints_IA.append(N >= 0)
    
    prob_IA = cp.Problem(cp.Maximize(obj_IA), constraints_IA)
    start = time.time()
    prob_IA.solve(solver=cp.MOSEK, verbose=False)
    end = time.time()  

    print('\nLP IA obj: '+str(obj_IA.value))
    return_dict['LP'] = {}
    return_dict['LP']['LB'] = (obj_IA.value).tolist()
    return_dict['LP']['RunTime'] = (end-start)
    
    #% =============== Enhanced LP Relaxation from Non-Negativity ============
    lmbd_IA = cp.Variable()
    X_IA = cp.Variable((n,n), symmetric=True)
    M = {}
    for i in range(n):
        M[i] = cp.Variable((n,n), symmetric=True)
    
    obj_IA = lmbd_IA
    
    constraints_IA = [Q - lmbd_IA*E == X_IA]
    for i in range(n):
        constraints_IA.append(X_IA-M[i] >= 0)
        constraints_IA.append(M[i][i,i] == 0)    
    
    for i in range(n):
        for j in range(n):
            if i!=j:
                constraints_IA.append(M[j][i,i] + 2*M[i][i,j] == 0)            
    
    for k in range(n):
        for j in range(k):
            for i in range(j):
                constraints_IA.append(M[i][j,k] + M[j][i,k] + M[k][i,j] >= 0)
    
    prob_IA = cp.Problem(cp.Maximize(obj_IA), constraints_IA)
    start = time.time()
    prob_IA.solve(solver=cp.MOSEK, verbose=False)
    end = time.time()  

    print('\nEnhanced LP IA obj: '+str(obj_IA.value))
    return_dict['Enhanced_LP'] = {}
    return_dict['Enhanced_LP']['LB'] = (obj_IA.value).tolist()
    return_dict['Enhanced_LP']['RunTime'] = (end-start)
    
    #% =============== SDP Relaxation from PSD + Non-Negativity ==============
    lmbd_IA = cp.Variable()
    X_IA = cp.Variable((n,n), symmetric=True)
    S = cp.Variable((n,n), symmetric=True)
    N = cp.Variable((n,n), symmetric=True)
    
    obj_IA = lmbd_IA
    
    constraints_IA = [Q - lmbd_IA*E == X_IA]
    constraints_IA.append(X_IA == S + N)
    constraints_IA.append(S >> 0)
    constraints_IA.append(N >= 0)
    
    prob_IA = cp.Problem(cp.Maximize(obj_IA), constraints_IA)
    start = time.time()
    prob_IA.solve(solver=cp.MOSEK, verbose=False)
    end = time.time()  

    print('\nSDP IA obj: '+str(obj_IA.value))   
    return_dict['SDP'] = {}
    return_dict['SDP']['LB'] = (obj_IA.value).tolist()
    return_dict['SDP']['RunTime'] = (end-start)
    
    #% ========= Enhanced SDP Relaxation from PSD + Non-Negativity ==========
    lmbd_IA = cp.Variable()
    X_IA = cp.Variable((n,n), symmetric=True)
    M = {}
    for i in range(n):
        M[i] = cp.Variable((n,n), symmetric=True)
    
    obj_IA = lmbd_IA
    
    constraints_IA = [Q - lmbd_IA*E == X_IA]
    for i in range(n):
        constraints_IA.append(X_IA-M[i] >> 0)
        constraints_IA.append(M[i][i,i] == 0)    
    
    for i in range(n):
        for j in range(n):
            if i!=j:
                constraints_IA.append(M[j][i,i] + 2*M[i][i,j] == 0)            
    
    for k in range(n):
        for j in range(k):
            for i in range(j):
                constraints_IA.append(M[i][j,k] + M[j][i,k] + M[k][i,j] >= 0)
                    
    prob_IA = cp.Problem(cp.Maximize(obj_IA), constraints_IA)
    start = time.time()
    prob_IA.solve(solver=cp.MOSEK, verbose=False)
    end = time.time()  

    print('\nEnhanced SDP IA obj: '+str(obj_IA.value))
    return_dict['Enhanced_SDP'] = {}
    return_dict['Enhanced_SDP']['LB'] = (obj_IA.value).tolist()
    return_dict['Enhanced_SDP']['RunTime'] = (end-start)
    
    return return_dict