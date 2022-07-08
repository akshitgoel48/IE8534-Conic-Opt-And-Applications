# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 04:36:27 2022

@author: Akshit
"""

import numpy as np
import cvxpy as cp
import random
import time

def add_constrIA(Delta):
    expr = {}
    for key in Delta.keys():
        expr[key] = {}
        for i in range(n):
            for j in range(i+1):                
                tmp_expr = Delta[key][j] @ X_IA @ Delta[key][i]
                constraints_IA[key,j,i] = tmp_expr >= 0
                # constraints_IA.append(tmp_expr >= 0) 
                if i != j:      expr[key][j,i] = tmp_expr
    return expr

def add_constrOA(Delta):
    for key in Delta.keys():
        for i in range(n):
            constraints_OA.append(Delta[key][i] @ X_OA @ Delta[key][i] >= 0)   
            
def partition_simplice():
    simplice_index = None
    for key in Delta.keys():
        for vertices_index, constr_expr in expr_IA[key].items():
            if round(constr_expr.value, 5) == 0:                
                simplice_index = key
                break;
        if simplice_index != None:   break;
        
    # simplice_index = random.sample(range(0,len(Delta)),1)[0]
    # vertices_index = np.sort(random.sample(range(0,len(Delta[simplice_index])),2))        
       
    u = Delta[simplice_index][vertices_index[0]]
    v = Delta[simplice_index][vertices_index[1]]
    
    w = mu*u + mu*v    
    constraints_OA.append(w @ X_OA @ w >= 0)
    
    key_list = list(Delta.keys()) 
    for key in key_list:
        vertices_index = {}
        tmp_list = [list(vecs) for vecs in Delta[key].values()]
        if list(u) in tmp_list and list(v) in tmp_list:
            vertices_index[0] = tmp_list.index(list(u))                  
            vertices_index[1] = tmp_list.index(list(v))
            tmp1 = Delta[key].copy()
            tmp2 = Delta[key].copy()
            tmp1[vertices_index[1]] = w
            Delta[key] = tmp1
            tmp2[vertices_index[0]] = w
            key2 = len(Delta)
            Delta[key2] = tmp2
            expr_IA[key2] = {}
            for i in range(n):
                for j in range(i+1):                
                    tmp_expr = Delta[key][j] @ X_IA @ Delta[key][i]
                    constraints_IA[key,j,i] = tmp_expr >= 0
                    # constraints_IA.append(tmp_expr >= 0) 
                    if i != j:      expr_IA[key][j,i] = tmp_expr  
                    
                    tmp_expr = Delta[key2][j] @ X_IA @ Delta[key2][i]
                    constraints_IA[key2,j,i] = tmp_expr >= 0
                    # constraints_IA.append(tmp_expr >= 0) 
                    if i != j:      expr_IA[key2][j,i] = tmp_expr 
            
#------------------------------------------------------------------------------            
def algorithm_smplx_partition(Q, epsilon):
    print("**RUNNING Simplex Partitions Algorithm**")
    global Delta, n, mu
    global expr_IA, constraints_IA, X_IA, constraints_OA, X_OA
    
    Delta = {}
    basis_vec = {}
    
    n = np.shape(Q)[0]
    E = np.ones((n,n))
    
    for index in range(n):
        tmp = np.zeros((n,))
        tmp[index] = 1
        basis_vec[index] = tmp
        
    Delta[0] = basis_vec
    
    lmbd_IA = cp.Variable()
    X_IA = cp.Variable((n,n), symmetric=True)
    constraints_IA = {}
    constraints_IA[0] = Q - lmbd_IA*E == X_IA
    expr_IA = add_constrIA(Delta)                
    
    lmbd_OA = cp.Variable()
    X_OA = cp.Variable((n,n), symmetric=True)
    constraints_OA = [Q - lmbd_OA*E == X_OA]       
    add_constrOA(Delta) 
           
    obj_IA = lmbd_IA
    obj_OA = lmbd_OA
    
    mu = 0.5
    # epsilon = 1e-6
    
    iter_num = 0
    runtime = 0
    obj_IA_list = []
    obj_OA_list = []
    while True:        
        # '---------- Inner Approximation ----------'
        prob_IA = cp.Problem(cp.Maximize(obj_IA), list(constraints_IA.values()))
        start = time.time()
        prob_IA.solve(solver=cp.CPLEX, verbose=False)
        end = time.time()  
        runtime = runtime + end-start
        obj_IA_list.append((obj_IA.value).tolist())
        
        # '---------- Outer Approximation ----------'
        prob_OA = cp.Problem(cp.Maximize(obj_OA), constraints_OA)
        start = time.time()
        prob_OA.solve(solver=cp.CPLEX, verbose=False)
        end = time.time()  
        runtime = runtime + end-start
        obj_OA_list.append((obj_OA.value).tolist())
        
        iter_num = iter_num + 1
        if obj_OA.value != None :
            gap = obj_OA.value-obj_IA.value
            tol = gap/(1+abs(obj_OA.value)+abs(obj_IA.value))  
            print(str(iter_num)+'. IA obj: '+str(obj_IA.value)+', OA obj: '+str(obj_OA.value)+', Tol: '+str(tol))
            if tol < epsilon :   break
        else:
            print(str(iter_num)+'. IA obj: '+str(obj_IA.value)+', Tol: '+str(float('Inf')))
        
        partition_simplice()
    
    return_dict = {}
    return_dict['LB'] = (obj_IA.value).tolist()
    return_dict['UB'] = (obj_OA.value).tolist()
    return_dict['NumIters'] = iter_num
    return_dict['RunTime'] = runtime
    return_dict['Tol'] = tol

    return [return_dict, obj_IA_list, obj_OA_list]

