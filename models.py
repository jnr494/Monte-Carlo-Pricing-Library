# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:46:52 2020

@author: Magnus Frandsen
"""

import numpy as np
import copy

class Model:
    def change_min_steps(self,min_n_steps):
        self.min_n_steps = lambda time: int(min_n_steps) * time
    
    def create_samples(self,n_sim,n_steps,time):
    
        n_sim = int(n_sim)
        n_steps = int(n_steps)
        
        if self.min_n_steps(time) <= n_steps:
            actual_steps = n_steps / time
            step_freq = 1
        else:  
            step_freq = int(np.ceil(self.min_n_steps(time) / n_steps))
            actual_steps = n_steps * step_freq / time
        
        total_steps = int(actual_steps * time)
        
        sample_paths = np.zeros(shape = (n_sim,n_steps+1))
        current_paths = np.array([np.float(self.spot)] * n_sim)
        sample_paths[:,0] = current_paths
        
        sample_vols = np.zeros(shape = (n_sim,n_steps+1))
        current_vols = np.array([np.float(self.init_vol)] * n_sim)
        sample_vols[:,0] = current_vols
        
        dw = self.bm_generator.generate_bm(n_process = self.n_random_var, 
                                           n_steps = total_steps,
                                           n_sim = n_sim,
                                           end_time = time)
        
        dt  = 1 / actual_steps
        
        for i in range(total_steps):
            temp_dw = dw[:,i] / np.sqrt(dt)
            
            current_paths, current_vols = self.move_samples_vols(current_paths,current_vols,dt,temp_dw)
            
            if (i+1) % step_freq == 0:
                sample_paths[:,int(i / step_freq) + 1] = current_paths
                sample_vols[:,int(i / step_freq) + 1] = current_vols       
            
        return sample_paths, sample_vols

class BlackScholesModel(Model):
    def __init__(self,spot,rate,vol, bm_generator):
        self.spot = spot
        self.rate = rate
        self.vol = vol
        
        self.min_n_steps = lambda time: 1
        self.init_vol = vol
        self.n_random_var = 1

        self.bm_generator = copy.deepcopy(bm_generator)
        
    def move_samples_vols(self,current_paths,current_vols,dt,temp_dw):
        
        current_paths *= np.array(np.exp((self.rate - current_vols**2/2)*dt \
                                            + current_vols * np.sqrt(dt) * temp_dw[:,0]))
        current_vols = current_vols
        return current_paths, current_vols
        
class HestonModel(Model):
    def __init__(self,spot,rate,nu,kappa,theta,sigma,rho, bm_generator, min_n_steps = 2**7):
        self.spot = spot
        self.rate = rate
        self.nu = nu
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        
        self.min_n_steps = lambda time: min_n_steps * time
        self.init_vol = np.sqrt(nu) 
        self.n_random_var = 2
    
        self.bm_generator = copy.deepcopy(bm_generator)
    
    def move_samples_vols(self,current_paths,current_vols,dt,temp_dw):
        current_vols = copy.deepcopy(current_vols)
        current_paths = copy.deepcopy(current_paths)
        
        
        current_var = current_vols**2
        
        current_paths *= np.array(np.exp((self.rate - current_var/2)*dt \
                                            + current_vols * np.sqrt(dt) * temp_dw[:,0]))
        current_var += self.kappa * (self.theta - current_var) * dt \
                    + self.sigma * current_vols * np.sqrt(dt) \
                    * (self.rho * temp_dw[:,0] + np.sqrt(1 - self.rho**2) * temp_dw[:,1])
        
        current_var = np.maximum(current_var,0)
        
        current_vols = np.sqrt(current_var)
        
        return current_paths, current_vols
    

class SABRModel(Model):
    def __init__(self,spot, rate, beta, sigma, alpha, rho, bm_generator, min_n_steps = 2**7):
        self.spot = spot
        self.rate = rate
        self.beta = beta
        self. alpha = alpha
        self.rho = rho
        
        self.min_n_steps = lambda time: min_n_steps * time
        self.init_vol = sigma
        self.n_random_var = 2
        
        self.bm_generator = copy.deepcopy(bm_generator)
    
    def move_samples_vols(self,current_paths,current_vols,dt,temp_dw):
        current_vols = copy.deepcopy(current_vols)
        current_paths = copy.deepcopy(current_paths)
        
        current_paths += self.rate * current_paths * dt \
            + current_vols * current_paths**self.beta * np.sqrt(dt) * temp_dw[:,0]
        
        current_paths = np.maximum(0,current_paths)
        
        current_vols *= np.exp(-0.5*self.alpha**2 * dt + self.alpha * current_vols * np.sqrt(dt) \
                               * (self.rho * temp_dw[:,0] + np.sqrt(1 - self.rho**2) * temp_dw[:,1]))
        
        return current_paths, current_vols