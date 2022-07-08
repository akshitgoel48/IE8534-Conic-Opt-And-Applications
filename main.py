# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 07:14:48 2022

@author: Akshit
"""

#%%
from datetime import date
import os, time
import numpy as np
from COP_IA_OA_QPoverSmplx import algorithm_smplx_partition
from COP_SDP_LP_Approx_QPoverSmplx import SDP_LP_Approx
import matplotlib.pyplot as plt

def plot_func(name):
    fig = plt.figure(figsize=(5,5))
    iter_num = soln_summary1['NumIters']
    xaxis = np.arange(iter_num)+1
    plt.plot(xaxis, IA_list, color = 'Red', label = "ILP (Alg.)", marker="+")
    plt.plot(xaxis, OA_list, color = 'Green', label = "OLP (Alg.)", marker="+")
    plt.plot(xaxis, soln_summary2['Enhanced_SDP']['LB']*np.ones(iter_num), color = 'Orange', label="SDP-1", linestyle="--")
    plt.plot(xaxis, soln_summary2['SDP']['LB']*np.ones(iter_num), color = 'Blue', label="SDP-0", linestyle=":")
    plt.plot(xaxis, soln_summary2['Enhanced_LP']['LB']*np.ones(iter_num), color = 'Brown', label="LP-1", linestyle="--")
    plt.plot(xaxis, soln_summary2['LP']['LB']*np.ones(iter_num), color = 'Olive', label="LP-0", linestyle=":")
    plt.xlim([1, iter_num])
    plt.ylabel('Objective value')
    plt.xlabel('Iteration No.')
    plt.legend(loc='lower right')
    fig.savefig(name, bbox_inches='tight')
    plt.close()

#%%
today = date.today()
t = time.localtime()
folder = today.strftime("%m_%d_%y") + "_" + time.strftime("%Hhr_%Mmin_%Ssec", t)
os.mkdir(folder)
folder = folder + "/"

#%%
Q1 = np.array([[1, 0, 1, 1, 0],
              [0, 1, 0, 1, 1],
              [1, 0, 1, 0, 1],
              [1, 1, 0, 1, 0],
              [0, 1, 1, 0, 1]])

[soln_summary1, IA_list, OA_list] = algorithm_smplx_partition(Q1, 1e-6)

soln_summary2 = SDP_LP_Approx(Q1)

plot_func(folder+'Q1.pdf')

#%%
Q2 = np.array([[0, 1, 1, 0, 0],
               [1, 0, 1, 0, 0],
               [1, 1, 0, 1, 1],
               [0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0]])

n = np.shape(Q2)[0]    
Q2 = Q2 + np.eye(n,n)
[soln_summary1, IA_list, OA_list] = algorithm_smplx_partition(Q2, 3e-4)

soln_summary2 = SDP_LP_Approx(Q2)

plot_func(folder+'Q2.pdf')

#%%
Q3 = np.array([[-14, -15,   -16,    0,     0 ],
              [-15, -14,   -12.5, -22.5, -15],
              [-16, -12.5, -10,   -26.5, -16],
              [ 0,  -22.5, -26.5,  0,     0 ],
              [ 0,  -15,   -16,    0,    -14]])

[soln_summary1, IA_list, OA_list] = algorithm_smplx_partition(Q3, 1e-6)

soln_summary2 = SDP_LP_Approx(Q3)

plot_func(folder+'Q3.pdf')

#%%
Q4 = np.array([[0.9044, 0.1054, 0.5140, 0.3322, 0],
               [0.1054, 0.8715, 0.7385, 0.5866, 0.9751],
               [0.5140, 0.7385, 0.6936, 0.5368, 0.8086],
               [0.3322, 0.5866, 0.5368, 0.5633, 0.7478],
               [0,      0.9751, 0.8086, 0.7478, 1.2932]])

[soln_summary1, IA_list, OA_list] = algorithm_smplx_partition(Q4, 1e-6)

soln_summary2 = SDP_LP_Approx(Q4)
plot_func(folder+'Q4.pdf')

#%%
import pandas as pd
column_list = ['n', 'Algorithm', 'LP-0', 'LP-1', 'SDP-0', 'SDP-1']
result_LB = []
result_time = []

for n in [5, 10, 15, 20]:
        
    for times in range(10):
        obj_vec = [n]
        time_vec = [n]
        
        np.random.seed(times)
        Q_rand = np.random.uniform(-n, n, (n,n))
        Q_rand = 0.5*Q_rand + 0.5*np.transpose(Q_rand)
        
        [soln_summary1, IA_list, OA_list] = algorithm_smplx_partition(Q_rand, 1e-6)
        obj_vec.append(soln_summary1['LB'])
        time_vec.append(soln_summary1['RunTime'])
        
        soln_summary2 = SDP_LP_Approx(Q_rand)
        obj_vec.append(soln_summary2['LP']['LB'])
        obj_vec.append(soln_summary2['Enhanced_LP']['LB'])
        obj_vec.append(soln_summary2['SDP']['LB'])
        obj_vec.append(soln_summary2['Enhanced_SDP']['LB'])
        time_vec.append(soln_summary2['LP']['RunTime'])
        time_vec.append(soln_summary2['Enhanced_LP']['RunTime'])
        time_vec.append(soln_summary2['SDP']['RunTime'])
        time_vec.append(soln_summary2['Enhanced_SDP']['RunTime'])
        
        result_LB.append(obj_vec)
        result_time.append(time_vec) 
        
    result_LB.append(['']*len(column_list))
    result_LB.append(['']*len(column_list))
    result_time.append(['']*len(column_list))
    result_time.append(['']*len(column_list))
    
    table_LB = pd.DataFrame(result_LB, columns = column_list) 
    table_time = pd.DataFrame(result_time, columns = column_list) 
    table_LB.to_excel(folder+'compare_obj.xlsx', index = False)
    table_time.to_excel(folder+'compare_time.xlsx', index = False)