# -*- coding: utf-8 -*-
"""
Created on Wed May 20 23:21:36 2020

@author: Magnus Frandsen
"""

#Required packages!
import copy
import numpy as np
from scipy.stats import norm
from torch.quasirandom import SobolEngine

def create_general_brownian_bridge(n_sim, n_steps, n_process, end_time):
    #Create instructions for the creation of the brownian bridge
    
    ex_list = [] #The instructions
    stage_list = []
    temp_high = n_steps
    temp_low = 0
    
    compute_mid_step = lambda low, high: int(np.ceil((high - low) / 2) + low)
    
    temp_step = compute_mid_step(temp_low, temp_high)
    temp_list = [temp_step,temp_low,temp_high]
    
    stage_list.append(temp_list)
    ex_list.append(temp_list)
    
    while len(ex_list) < n_steps - 1:
        new_stage_list = []
        for item in stage_list:
            temp_step, temp_low, temp_high = item
            
            if temp_step - temp_low > 1:
                new_step = compute_mid_step(temp_low, temp_step)
                new_list = [new_step, temp_low, temp_step]
                
                new_stage_list.append(new_list)
                ex_list.append(new_list)
        
            
            if temp_high - temp_step > 1:
                new_step = compute_mid_step(temp_step, temp_high)
                new_list = [new_step, temp_step, temp_high]
                
                new_stage_list.append(new_list)
                ex_list.append(new_list)
            
            stage_list = copy.deepcopy(new_stage_list)
    
    #Create Brownian bridge
    #Modified algorithm from Glasserman 2009
    times = np.linspace(0, end_time, n_steps + 1)
    W = np.zeros(shape = (n_sim, n_steps+1, n_process))
    
    #Generate normal variables
    Sobol = SobolEngine(n_process * n_steps,scramble=True,seed=None)
    Z = norm.ppf(np.array(Sobol.draw(n_sim))*(1-2e-7)+1e-7)
    split = [list(range(n_process * n_steps))[i::n_process] for i in range(n_process)]
    Z = np.dstack(tuple([Z[:,i] for i in split]))
    
    #Create bm via brownian bridge
    W[:,n_steps,:] = np.sqrt(times[n_steps]) * Z[:,0,:]
    
    if n_steps > 1:
        cur_z_nr = 1
        for item in ex_list:
            i, l, r = item
            a = ((times[r] - times[i])*W[:,l,:] + (times[i] - times[l])*W[:,r,:]) / (times[r] - times[l])
            b = np.sqrt((times[i] - times[l]) * (times[r] - times[i]) / (times[r] - times[l]))
            W[:,i,:] = a + b * Z[:,cur_z_nr,:]
            
            cur_z_nr += 1
    
    dw = W[:,1:,:] - W[:,:-1,:]
    return dw

