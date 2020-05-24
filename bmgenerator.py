# -*- coding: utf-8 -*-
"""
Created on Sat May 16 17:24:43 2020

@author: Magnus Frandsen
"""

import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from torch.quasirandom import SobolEngine
from scipy.stats import norm
import copy
import general_brownian_bridge

class BMGenerator:
    def __init__(self,method = 'normal'):
        methods = ['normal','a','mm','sobol','sobolpca','sobolbb']
        if str(method).lower() in methods:    
            self.method = method
        else:
            self.method = 'normal'
    
    def generate_bm(self,n_process, n_steps,n_sim,end_time):
        if self.method == 'normal':
            return self.generate_bm_normal(n_process, n_steps,n_sim,end_time)
        elif self.method == 'a':
            return self.generate_bm_a(n_process, n_steps,n_sim,end_time)
        elif self.method == 'mm':
            return self.generate_bm_mm(n_process, n_steps,n_sim,end_time)
        elif self.method == 'sobol':
            return self.generate_bm_sobol(n_process, n_steps,n_sim,end_time)
        elif self.method == 'sobolpca':
            return self.generate_bm_sobolpca(n_process, n_steps,n_sim,end_time)
        elif self.method == 'sobolbb':
            return self.generate_bm_sobolbb(n_process, n_steps,n_sim,end_time)
        
    def generate_bm_normal(self,n_process, n_steps,n_sim,end_time):
        dw = np.random.normal(size=(n_sim,n_steps,n_process)) / np.sqrt(n_steps / end_time) 
        return dw
        
    def generate_bm_a(self,n_process, n_steps,n_sim,end_time):
        temp_n_sim = int(np.ceil(n_sim/2))
        dw = np.random.normal(size=(temp_n_sim,n_steps,n_process)) / np.sqrt(n_steps / end_time) 
        dw = np.concatenate((dw,-dw),axis = 0)
        return dw

    def generate_bm_mm(self,n_process, n_steps,n_sim,end_time):
        dw = np.random.normal(size=(n_sim,n_steps,n_process))
        
        mean = np.mean(dw,axis = 0)
        std = np.std(dw,axis = 0)
        
        dw = (dw - mean) / std 
        dw = dw / np.sqrt(n_steps / end_time)
        return(dw)
    
    def generate_bm_sobol(self,n_process, n_steps,n_sim,end_time):
        #generate normals
        Sobol = SobolEngine(n_process * n_steps,scramble=True,seed=None)
        Z = norm.ppf(np.array(Sobol.draw(n_sim))*(1-2e-7)+1e-7)
        
        split = [list(range(n_process * n_steps))[i::n_process] for i in range(n_process)]
        Z = np.dstack(tuple([Z[:,i] for i in split]))
        
        dw = Z / np.sqrt(n_steps / end_time) 
        return dw
    
    def generate_bm_sobolpca(self,n_process, n_steps,n_sim,end_time):
        #generate normals
        Sobol = SobolEngine(n_process * n_steps,scramble=True,seed=None)
        Z = norm.ppf(np.array(Sobol.draw(n_sim))*(1-2e-7)+1e-7)
        
        split = [list(range(n_process * n_steps))[i::n_process] for i in range(n_process)]
        Z = np.dstack(tuple([Z[:,i] for i in split]))
        
        #pca computations
        V = np.ones(shape=(n_steps,n_steps))
        for i in range(1,n_steps+1):
            V[i:,i:] += np.ones(shape=(n_steps - i, n_steps - i))
        V = V / n_steps * end_time
        
        lambdas, W = linalg.eig(V)
        D = np.diag(np.real(lambdas))
        
        A = W @ D**0.5
        
        
        
        #pca brownian motion
        pca_bm = np.dstack(tuple([Z[:,:,i] @ A[:,:].T for i in range(n_process)]))
        pca_dw = pca_bm
        pca_bm = np.concatenate((np.zeros((n_sim,1,n_process)), pca_bm),axis = 1)
        pca_dw = pca_bm[:,1:,:] - pca_bm[:,:-1,:]
        
        return pca_dw
    
    def generate_bm_sobolbb(self,n_process, n_steps,n_sim,end_time):
        dw = general_brownian_bridge.create_general_brownian_bridge(n_sim,n_steps,n_process,end_time)
        return dw
        
    def generate_bm_sobolbb_legacy(self,n_process, n_steps,n_sim,end_time):
        m = int(np.ceil(np.log(n_steps) / np.log(2)))
        h = int(2 ** m)
        times = np.linspace(0, end_time, h + 1)
        W = np.zeros(shape = (n_sim, h+1, n_process))
        
        #Generate normal variables
        Sobol = SobolEngine(n_process * h,scramble=True,seed=None)
        Z = norm.ppf(np.array(Sobol.draw(n_sim))*(1-2e-7)+1e-7)
        split = [list(range(n_process * h))[i::n_process] for i in range(n_process)]
        Z = np.dstack(tuple([Z[:,i] for i in split]))
        
        #Z = np.random.normal(size=(n_sim, h, n_process))
        
        #Create bm via brownian bridge - see almgorithm from fig 3.2 in Glasserman2010
        j_max = 1
        W[:,h,:] = np.sqrt(times[h]) * Z[:,0,:]
        cur_z_nr = 1
        for k in range(1, m + 1):
            i_min = int(h / 2)
            i = i_min
            l = 0
            r = h
            
            for j in range(1,j_max + 1):
                a = ((times[r] - times[i])*W[:,l,:] + (times[i] - times[l])*W[:,r,:]) / (times[r] - times[l])
                b = np.sqrt((times[i] - times[l]) * (times[r] - times[i]) / (times[r] - times[l]))
                W[:,i,:] = a + b * Z[:,cur_z_nr,:]
                
                i += h
                l += h
                r += h
                
                cur_z_nr += 1
                
            j_max *= 2
            h = i_min
                
        dw = W[:,1:,:] - W[:,:-1,:]
        return dw
        
# =============================================================================
# B = BMGenerator(method = 'sobolbb')
# a = B.generate_bm(2,2**7-1,2**10,1)
# 
# print(np.mean(np.cumsum(a,axis=1)[:,2**6,0]))
# print(np.std(np.cumsum(a,axis=1)[:,2**6,0])**2)
# plt.plot(np.linspace(1/2**7-1,1,2**7-1),np.cumsum(a[0,:,0]))
# =============================================================================

if __name__ == "__main__":
    pass
# =============================================================================
#     d = 100
#     maturity = 1
#     
#     V = np.ones(shape=(d,d))
#     for i in range(1,d+1):
#         V[i:,i:] += np.ones(shape=(d-i,d-i))
#         
#     V = V / d * maturity
#     
#     lambdas, W = linalg.eig(V)
#     D = np.diag(np.real(lambdas))
#     
#     A = W @ D**0.5
#     
#     Z = np.random.normal(size = d)
#     
#     BM = np.cumsum(Z / np.sqrt(d) * maturity)
#     
#     #k = 20
#     #PCA_BM = A[:,:k] @ Z[:k]
#     
#     
#     ts = np.arange(1,d+1)/d*maturity
#     plt.plot(ts,BM)
#     
#     ks = [10,30,100]
#     for k in ks:
#         plt.plot(ts,A[:,:k] @ Z[:k])
# 
# =============================================================================
