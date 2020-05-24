# -*- coding: utf-8 -*-
"""
Created on Wed May 20 23:21:36 2020

@author: Magnus Frandsen
"""

import numpy as np
import copy

n_steps = 200

ex_list = []
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
    
##### SQUIP!!!!
